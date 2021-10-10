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

# Before running the code, run 'python -m visdom.server' in the terminal to open visdom panel.

# divide a segment equally into n parts according to the length
lengthOfVelocityProfile = 10

# m
meanOfSegmentLength = 607.8692133
stdOfSegmentLength = 899.7448693

omega_time = 0.5
omega_fuel = 0.5

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
label_dimension = 2

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

# local
ckpt_path = "best13dMultiTask.mdl"
data_root = "model_data_newSep"
output_root = "prediction_result.csv"

# load data
train_db = ObdData(root=data_root, mode = "train", percentage=10, window_size=window_sz, \
                   path_length=train_path_length, label_dimension=label_dimension, pace=pace_train, withoutElevation=False)
val_db = ObdData(root=data_root, mode="val", percentage=10, window_size=window_sz, \
                 path_length=train_path_length, label_dimension=label_dimension, pace=pace_test, withoutElevation=False)
test_db = ObdData(root=data_root, mode="test", percentage=10, window_size=window_sz, \
                  path_length=test_path_length, label_dimension=label_dimension, pace=pace_test, withoutElevation=False)
train_loader = DataLoader(train_db, batch_size=batchsz, num_workers=0)
val_loader = DataLoader(val_db, batch_size=batchsz, num_workers=0)
test_loader = DataLoader(test_db, batch_size=batchsz, num_workers=0)


def denormalize(normalized, mean, std):
    return normalized*std + mean

def mape_loss(label, pred):
    """
    Calculate Mean Absolute Percentage Error
    labels with 0 value are masked
    :param pred: [batchsz]
    :param label: [batchsz]
    :return: MAPE
    """
    mask = label >= 1e-4

    mape = torch.mean(torch.abs((pred[mask] - label[mask]) / label[mask]))
    # print(p, l, loss_energy)
    return mape


def timeEstimation(velocityProfile, length):
    '''

    :param velocityProfile: a list of velocity profile (m/s) # [batch size, lengthOfVelocityProfile]
    :param length: m # [batch size]
    :return: time: s
    '''
    lengthOfOneSubSegment = length/lengthOfVelocityProfile
    # need to be revised
    estTime = torch.sum(lengthOfOneSubSegment/velocityProfile)
    return estTime


def fuelEstimation(velocityProfile, length):
    '''

    :param velocityProfile: a list of velocity profile (m/s)
    :param length: m
    :return: fuel: liter
    '''
    return 0


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
    loss_mape_fuel = 0
    loss_mape_time = 0
    loss_mse = 0
    cnt = 0
    id = 0
    model.eval()
    for x, y, c in loader:
        #x, y, c = x.to(device), y.to(device), c.to(device)
        with torch.no_grad():
            label_segment = torch.zeros(y.shape[0], label_dimension).to(device)
            pred_segment = torch.zeros(y.shape[0], label_dimension).to(device)
            #label_segment_denormalized = torch.zeros(y.shape[0], output_dimension).to(device)
            #pred_segment_denormalized = torch.zeros(y.shape[0], output_dimension).to(device)

            # For each batch, predict fuel consumption/time for each segment in a path and sum them
            for i in range(x.shape[1]):
                # [batch size, window length, feature dimension]
                x_segment = x[:, i, :, :]
                # [batch, categorical_dim, window size]
                c_segment = c[:, :, i, :]
                # [batch size, output dimension]
                #print(x_segment.shape,c_segment.shape)
                velocityProfile = model(x_segment, c_segment)
                # extract the length of this segment
                # [batch size]
                length = denormalize(x_segment[:, window_sz // 2, 4], meanOfSegmentLength, stdOfSegmentLength)
                pred_Time = timeEstimation(velocityProfile, length)
                pred_Fuel = fuelEstimation(velocityProfile, length)

                # [batch size, output dimension]
                pred_segment[:, 0] += pred_Fuel
                pred_segment[:, 1] += pred_Time


                #pred_segment_denormalized += denormalize(pred)
                # [batch size, output dimension]
                t = torch.tensor([1, 0.01]).unsqueeze(0).to(device)
                label = y[:, i, window_sz // 2] * t

                # [batch size, output dimension]
                label_segment += label
                #label_segment_denormalized += denormalize(label)

                if output:
                    for j in range(y.shape[0]):
                        writer.writerow([id, np.array(label[j, 0].cpu()), np.array(label[j, 1].cpu()), np.array(pred[j, 0].cpu()), np.array(pred[j, 1].cpu())])
                        id += 1
            mse_fuel = F.mse_loss(label_segment[:0], pred_segment[:0])*y.shape[0]
            mse_time = F.mse_loss(label_segment[:1], pred_segment[:1]) * y.shape[0]
            loss_mse += omega_fuel * mse_fuel + omega_time * mse_time
            #print(label_segment, pred_segment, mape_loss(label_segment, pred_segment))
            loss_mape_fuel += mape_loss(label_segment[:,0], pred_segment[:,0])*y.shape[0]
            loss_mape_time += mape_loss(label_segment[:, 1], pred_segment[:, 1]) * y.shape[0]
            cnt += y.shape[0]
    if output:
        csvFile.close()
    return loss_mse/cnt, loss_mape_fuel/cnt, loss_mape_time/cnt


def train():
    viz = visdom.Visdom()
    # Create a new model or load an existing one.
    model = AttentionBlk(feature_dim=feature_dimension,embedding_dim=[4,2,2,2,2,4,4],num_heads=head_number,output_dimension=lengthOfVelocityProfile)
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
    viz.line([0], [-1], win='learning rate', opts=dict(title='learning rate'))
    viz.line([0], [-1], win='train_time_mse', opts=dict(title='train_time_mse'))
    viz.line([0], [-1], win='train_fuel_mse', opts=dict(title='train_fuel_mse'))
    viz.line([0], [-1], win='train_time_mape', opts=dict(title='train_time_mape'))
    viz.line([0], [-1], win='train_fuel_mape', opts=dict(title='train_fuel_mape'))
    viz.line([0], [-1], win='val_time_mape', opts=dict(title='val_time_mape'))
    viz.line([0], [-1], win='val_fuel_mape', opts=dict(title='val_fuel_mape'))

    for epoch in range(epochs):
        model.train()
        for step, (x, y, c) in tqdm(enumerate(train_loader)):
            # x: numerical features [batch, path length, window size, feature dimension]
            # y: label [batch, path length, window size, (label dimension)]
            # c: categorical features [batch, number of categorical features, path length, window size]
            #x, y, c = x.to(device), y.to(device), c.to(device)

            # For each batch, predict fuel consumption/time for each segment in a path and sum them
            # [batch size, output dimension]

            label_path = torch.zeros(y.shape[0], label_dimension).to(device)
            pred_path = torch.zeros(y.shape[0], label_dimension).to(device)
            #label_segment_denormalized = torch.zeros(y.shape[0], output_dimension).to(device)
            #pred_segment_denormalized = torch.zeros(y.shape[0], output_dimension).to(device)

            # For each batch, predict fuel consumption/time for each segment in a path and sum them
            for i in range(x.shape[1]):
                # [batch size, window length, feature dimension]
                x_segment = x[:, i, :, :]
                # [batch, categorical_dim, window size]
                c_segment = c[:, :, i, :]
                # [batch size, lengthOfVelocityProfile]
                velocityProfile = model(x_segment, c_segment)
                # extract the length of this segment
                # [batch size]
                length = denormalize(x_segment[:, window_sz // 2, 4], meanOfSegmentLength, stdOfSegmentLength)
                pred_Time = timeEstimation(velocityProfile, length)
                pred_Fuel = fuelEstimation(velocityProfile, length)

                # [batch size, output dimension]
                pred_path[:, 0] += pred_Fuel
                pred_path[:, 1] += pred_Time
                #pred_segment_denormalized += denormalize(pred)

                # [batch size, output dimension]
                # time*0.01 so that time and fuel are in the similar unit
                t = torch.tensor([1, 0.01]).unsqueeze(0).to(device)
                label = y[:, i, window_sz // 2]*t


                # [batch size, output dimension]
                label_path += label
                #label_segment_denormalized += denormalize(label)
            loss_fuel = criterion(label_path[:0], pred_path[:0])
            loss_time = criterion(label_path[:1], pred_path[:1])
            #loss = mape_loss(label_path, pred_path)
            #print(loss)
            loss = omega_fuel*loss_fuel + omega_time*loss_time
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        viz.line([loss.item()], [global_step], win='train_mse', update='append')
        viz.line([loss_fuel.item()], [global_step], win='train_fuel_mse', update='append')
        viz.line([loss_time.item()], [global_step], win='train_time_mse', update='append')

        loss_mape_fuel = mape_loss(label_path[:, 0], pred_path[:, 0])
        loss_mape_time = mape_loss(label_path[:, 1], pred_path[:, 1])

        viz.line([loss_mape_fuel.item()], [global_step], win='train_fuel_mape', update='append')
        viz.line([loss_mape_time.item()], [global_step], win='loss_mape_time', update='append')

        schedule.step(loss)
        learning_rate = optimizer.state_dict()['param_groups'][0]['lr']
        viz.line([learning_rate], [global_step], win='learning rate', update='append')

        # print("epoch:", epoch, "test_mse:", loss)
        if epoch % 1 == 0:
            val_mse, val_fuel_mape, val_time_mape = eval(model, val_loader)
            # schedule.step(val_mse)
            print("epoch:", epoch, "val_mape_fuel(%):", np.array(val_fuel_mape.cpu())*100,"val_mape_time(%):", np.array(val_time_mape.cpu())*100, "val_mse:", np.array(val_mse.cpu()))
            print("epoch:", epoch,  "train_mse:", loss)
            if val_mse < best_mse:
                best_epoch = epoch
                best_mse = val_mse
                torch.save(model.state_dict(), ckpt_path)
            viz.line([val_mse.item()], [global_step], win='val_mse', update='append')
            viz.line([val_fuel_mape.item()], [global_step], win='val_fuel_mape', update='append')
            viz.line([val_time_mape.item()], [global_step], win='val_time_mape', update='append')

        global_step += 1
    print("best_epoch:", best_epoch, "best_mse:", np.array(best_mse.cpu()))


def test(output = False):
    """

    :param output: "True" -> output the estimation results to output_root
    :return:
    """

    # load an existing model.
    model = AttentionBlk(feature_dim=feature_dimension, embedding_dim=[4,2,2,2,2,4,4], num_heads=head_number, output_dimension=lengthOfVelocityProfile)
    if os.path.exists(ckpt_path):
        print('Reloading model parameters..')
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
    else:
        print('Error: no existing model')
    model.to(device)
    print(model)
    p = sum(map(lambda p: p.numel(), model.parameters()))
    print("number of parameters:", p)
    test_mse, test_fuel_mape, test_time_mape = eval(model, test_loader, output = output)
    print("test_mape_fuel(%):", np.array(test_fuel_mape.cpu()) * 100)
    print("test_mape_fuel(%):", np.array(test_time_mape.cpu()) * 100)
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
    #main("test")
    #main("test", output = True)
    main("train")

# 602 parameters
# test_length_path = [1,2,5,10,20,50,100,200,500]
# mape =  [878.4875869750977,104.24556732177734,35.02033352851868,20.90749442577362,14.545997977256775,10.099445283412933,7.709670811891556,6.324310600757599,5.235186591744423]

