import torch
from torch import nn, optim
import os
from torch.utils.data import DataLoader
from obddata import ObdData
from nets import AttentionBlk
import torch.nn.functional as F
import numpy as np
import csv
import time
import visdom
from tqdm import tqdm
#aaa

# batch size 512 BEST
batchsz = 512

# learning rate
lr = 1e-3

# number of training epochs
epochs = 3000
# epochs = 0

# random seed
seed = 1234
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
# dimension of the output: [fuel consumption, time]
output_dimension = 1

# dimension of the input numerical features:
# [speed limit, mass, elevation change, previous orientation, length, direction angle]
feature_dimension = 6
# there are also 7 categorical features:
# "road_type", "time_stage", "week_day", "lanes", "bridge", "endpoint_u", "endpoint_v"

# multi-head attention not influential
head_number = 1

# length of path used for training/validation
train_path_length = 10
# length of path used for test
test_path_length = 1

# window size 3 best
window_sz = 3

# pace between paths
pace_train = 5
pace_test = pace_train
if pace_train>train_path_length:
    pace_train = train_path_length
    print("pace train has been changed to:", pace_train)
if pace_test>test_path_length:
    pace_test = test_path_length
    print("pace test has been changed to:",pace_test)

use_cuda = torch.cuda.is_available()
if use_cuda:
    #print('Using GPU..')
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# in Google Colab
# root for the trained model
# ckpt_path = "/content/drive/MyDrive/Colab_Notebooks/DeepLearning/best.mdl"
# root for the data
# data_root = "/content/drive/MyDrive/Colab_Notebooks/DeepLearning/data_normalized"
# root for the estimation output file
# output_root = "/content/drive/MyDrive/Colab_Notebooks/DeepLearning/prediction_result.csv"
# local
ckpt_path = "best_13d_fuel.mdl"
#ckpt_path = "best_13d_time.mdl"
data_root = "normalized data"
#data_root = "DataDifferentiated"
output_root = "prediction_result.csv"

# load data
train_db = ObdData(root=data_root,mode = "train",percentage=20, window_size=window_sz,\
                   path_length=train_path_length, label_dimension=output_dimension, pace=pace_train)
val_db = ObdData(root=data_root,mode="val",percentage=20,window_size=window_sz,\
                 path_length=test_path_length, label_dimension=output_dimension, pace=pace_test)
test_db = ObdData(root=data_root,mode="test",percentage=20,window_size=window_sz,\
                  path_length=test_path_length, label_dimension=output_dimension, pace=pace_test)
train_loader = DataLoader(train_db, batch_size=batchsz, num_workers=0)
val_loader = DataLoader(val_db, batch_size=batchsz, num_workers=0)
test_loader = DataLoader(test_db, batch_size=batchsz, num_workers=0)


def denormalize(x_hat):
    fuel_mean = [0.205986075]
    fuel_std = [0.32661580545285]
    mean = torch.tensor(fuel_mean).unsqueeze(1).to(device)
    std = torch.tensor(fuel_std).unsqueeze(1).to(device)
    return x_hat*std + mean

def mape_loss(label, pred):
    """
    Calculate Mean Absolute Percentage Error of the energy consumption estimation
    labels with 0 value are masked
    :param pred: [batchsz, output dimension]
    :param label: [batchsz, output dimension]
    :return: MAPE
    """
    p = pred[:, 0]
    l = label[:, 0]
    mask = l >= 1e-4

    loss_energy = torch.mean(torch.abs((p[mask] - l[mask]) / l[mask]))
    # print(p, l, loss_energy)
    return loss_energy


def eval(model, loader, output = False):
    """
    Evaluate the model accuracy by MAPE of the energy consumption & average MSE of energy and time
    :param model: trained model
    :param loader: data loader
    :param output: output the estimation result or not
    :return: MAPE of energy, average MSE of energy and time
    """
    if output:
        csvFile = open(output_root, "w")
        writer = csv.writer(csvFile)
        writer.writerow(["id", "ground truth fuel(l)", "ground truth time (s)", "estimated fuel (l)", "estimated time (s)"])
    loss_mape = 0
    loss_mse = 0
    cnt = 0
    id = 0
    model.eval()
    for x, y,c in loader:
        #x, y, c = x.to(device), y.to(device), c.to(device)
        with torch.no_grad():
            label_segment = torch.zeros(y.shape[0], output_dimension).to(device)
            pred_segment = torch.zeros(y.shape[0], output_dimension).to(device)
            #label_segment_denormalized = torch.zeros(y.shape[0], output_dimension).to(device)
            #pred_segment_denormalized = torch.zeros(y.shape[0], output_dimension).to(device)

            # For each batch, predict fuel consumption/time for each segment in a path and sum them
            for i in range(x.shape[1]):
                # [batch size, window length, feature dimension]
                x_segment = x[:, i, :, :]
                # [batch, categorical_dim, window size]
                c_segment = c[:, :, i, :]
                # [batch size, output dimension]
                pred = model(x_segment, c_segment)

                # [batch size, output dimension]
                pred_segment += pred
                #pred_segment_denormalized += denormalize(pred)
                # [batch size, output dimension]
                if output_dimension == 1:
                    label = y[:, i, window_sz // 2].unsqueeze(-1)
                else:
                    t = torch.tensor([1, 0.01]).unsqueeze(0).to(device)
                    label = y[:, i, window_sz // 2] * t

                # [batch size, output dimension]
                label_segment += label
                #label_segment_denormalized += denormalize(label)

                if output:
                    if output_dimension == 1:
                        for j in range(y.shape[0]):
                            writer.writerow([id, np.array(label[j,0].cpu()), "-", np.array(pred[j,0].cpu()),\
                                             "-"])
                    if output_dimension == 2:
                        for j in range(y.shape[0]):
                            writer.writerow([id, np.array(label[j, 0].cpu()), np.array(label[j, 1].cpu()), np.array(pred[j, 0].cpu()), \
                                            np.array(pred[j, 1].cpu())])
                        id += 1
            loss_mse += F.mse_loss(label_segment, pred_segment)*y.shape[0]
            #print(label_segment, pred_segment, mape_loss(label_segment, pred_segment))
            loss_mape += mape_loss(label_segment, pred_segment)*y.shape[0]
            cnt += y.shape[0]
    if output:
        csvFile.close()
    return loss_mape/cnt, loss_mse/cnt


def train():
    viz = visdom.Visdom()
    # Create a new model or load an existing one.
    model = AttentionBlk(feature_dim=feature_dimension,embedding_dim=[4,2,2,2,2,4,4],num_heads=head_number,output_dimension=output_dimension)
    if os.path.exists(ckpt_path):
        print('Reloading model parameters..')
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
    else:
        print('Creating new model parameters..')
        # this code is very important! It initialises the parameters with a
        # range of values that stops the signal fading or getting too big.
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    model.to(device)
    print(model)
    print(next(model.parameters()).device)
    p = sum(map(lambda p: p.numel(), model.parameters()))
    print("number of parameters:", p)

    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9)
    schedule = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)

    criterion = nn.MSELoss()
    global_step = 0
    best_mape, best_mse, best_epoch = torch.tensor(float("inf")), torch.tensor(float("inf")), 0
    viz.line([0],[-1], win='train_mse', opts=dict(title='train_mse'))
    viz.line([0], [-1], win='val_mse', opts=dict(title='val_mse'))
    viz.line([0], [-1], win='train_mape', opts=dict(title='train_mape'))
    viz.line([0], [-1], win='val_mape', opts=dict(title='val_mape'))
    viz.line([0], [-1], win='learning rate', opts=dict(title='learning rate'))
    for epoch in range(epochs):
        model.train()
        for step, (x, y, c) in tqdm(enumerate(train_loader)):
            # x: numerical features [batch, path length, window size, feature dimension]
            # y: label [batch, path length, window size, (label dimension)]
            # c: categorical features [batch, number of categorical features, path length, window size]
            #x, y, c = x.to(device), y.to(device), c.to(device)

            # For each batch, predict fuel consumption/time for each segment in a path and sum them
            # [batch size, output dimension]

            label_segment = torch.zeros(y.shape[0], output_dimension).to(device)
            pred_segment = torch.zeros(y.shape[0], output_dimension).to(device)
            #label_segment_denormalized = torch.zeros(y.shape[0], output_dimension).to(device)
            #pred_segment_denormalized = torch.zeros(y.shape[0], output_dimension).to(device)

            # For each batch, predict fuel consumption/time for each segment in a path and sum them
            for i in range(x.shape[1]):
                # [batch size, window length, feature dimension]
                x_segment = x[:, i, :, :]
                # [batch, categorical_dim, window size]
                c_segment = c[:, :, i, :]
                # [batch size, output dimension]
                pred = model(x_segment, c_segment)

                # [batch size, output dimension]
                pred_segment += pred
                #pred_segment_denormalized += denormalize(pred)

                # [batch size, output dimension]
                if output_dimension == 1:
                    label = y[:, i, window_sz// 2].unsqueeze(-1)
                else:
                    t = torch.tensor([1, 0.01]).unsqueeze(0).to(device)
                    label = y[:, i, window_sz // 2]*t


                # [batch size, output dimension]
                label_segment += label
                #label_segment_denormalized += denormalize(label)
            loss = criterion(label_segment, pred_segment)
            #loss = mape_loss(label_segment, pred_segment)
            #print(loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        viz.line([loss.item()], [global_step], win='train_mse', update='append')
        loss_mape = mape_loss(label_segment, pred_segment)
        viz.line([loss_mape.item()], [global_step], win='train_mape', update='append')
        schedule.step(loss)
        learning_rate = optimizer.state_dict()['param_groups'][0]['lr']
        viz.line([learning_rate], [global_step], win='learning rate', update='append')
        # print("epoch:", epoch, "test_mse:", loss)
        if epoch % 1 == 0:
            val_mape, val_mse = eval(model, val_loader)
            # schedule.step(val_mse)
            print("epoch:", epoch, "val_mape(%):", np.array(val_mape.cpu())*100, "val_mse:", np.array(val_mse.cpu()))
            print("epoch:", epoch,  "train_mse:", loss)
            if val_mse < best_mse:
                best_epoch = epoch
                best_mse = val_mse
                torch.save(model.state_dict(), ckpt_path)
            viz.line([val_mse.item()], [global_step], win='val_mse', update='append')

            viz.line([val_mape.item()], [global_step], win='val_mape', update='append')
            #schedule.step(val_mse)
        global_step += 1
    print("best_epoch:", best_epoch, "best_mape(%):", np.array(best_mape.cpu())*100, "best_mse:", np.array(best_mse.cpu()))


def test(output = False):
    """

    :param output: "True" -> output the estimation results to output_root
    :return:
    """

    # load an existing model.
    model = AttentionBlk(feature_dim=feature_dimension,embedding_dim=[4,2,2,2,2,4,4],num_heads=head_number,output_dimension=output_dimension)
    if os.path.exists(ckpt_path):
        print('Reloading model parameters..')
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
    else:
        print('Error: no existing model')
    model.to(device)
    print(model)
    p = sum(map(lambda p: p.numel(), model.parameters()))
    print("number of parameters:", p)
    test_mape, test_mse = eval(model, test_loader, output = output)
    print("test_mape(%):", np.array(test_mape.cpu()) * 100)
    print("test_mse:", np.array(test_mse.cpu()))


def main(mode, output = False):
    """
    :param mode: "train" for training, "test" for testing the existing model or predicting
    :return:
    """
    if mode == "train":
        train()
    elif mode == "test":
        test(output = output)


if __name__ == '__main__':
    main("test")
    #main("test", output = True)
    #main("train")

# 602 parameters
# test_length_path = [1,2,5,10,20,50,100,200,500]
# mape =  [878.4875869750977,104.24556732177734,35.02033352851868,20.90749442577362,14.545997977256775,10.099445283412933,7.709670811891556,6.324310600757599,5.235186591744423]

'''
batch sz 256
window sz 3
header 1
test_mape(%): 143.4951663017273
test_mse: 0.020990536
test_mape(%): 81.2801718711853
test_mse: 0.03261754
test_mape(%): 25.552162528038025
test_mse: 0.086509205
test_mape(%): 16.768477857112885
test_mse: 0.19350451
test_mape(%): 12.350236624479294
test_mse: 0.48980802
test_mape(%): 9.486592561006546
test_mse: 1.889815
test_mape(%): 8.374053239822388
test_mse: 4.772601
test_mape(%): 8.564424514770508
test_mse: 14.494676
'''