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

print(torch.__version__)
# batch size
batchsz = 32

# learning rate
lr = 1e-3

# number of training epochs
epochs = 3000
# epochs = 0

# random seed
torch.manual_seed(1234)

# dimension of the output: [fuel consumption, time]
output_dimension = 2

# dimension of the input features:
# [position, road type, speed limit, mass, elevation change, previous orientation, length, direction angle]
feature_dimension = 8

# multi-head attention
head_number = 1

# length of path used for training/validation
train_path_length = 10
# length of path used for test
test_path_length = 10

use_cuda = torch.cuda.is_available()
if use_cuda:
    print('Using GPU..')
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
ckpt_path = "best.mdl"
data_root = "data_normalized"
output_root = "prediction_result.csv"

# load data
train_db = ObdData(root=data_root,mode = "train",percentage=20, path_length=train_path_length)
val_db = ObdData(root=data_root,mode="val",percentage=20, path_length=train_path_length)
test_db = ObdData(root=data_root,mode="test",percentage=20, path_length=test_path_length)
train_loader = DataLoader(train_db, batch_size=batchsz, num_workers=2)
val_loader = DataLoader(val_db, batch_size=batchsz, num_workers=2)
test_loader = DataLoader(test_db, batch_size=batchsz, num_workers=2)


def mape_loss(pred, label):
    """
    Calculate Mean Absolute Percentage Error of the energy consumption estimation
    labels with 0 value are masked
    :param pred: [batchsz, output dimension]
    :param label: [batchsz, output dimension]
    :return: MAPE
    """
    p = pred[:, 0]
    l = label[:, 0]
    mask = l != 0
    loss_energy = torch.mean(torch.abs((p[mask] - l[mask]) / l[mask]))
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
    for x, y in loader:
        x,y = x.to(device), y.to(device)
        with torch.no_grad():
            label_segment = torch.zeros(y.shape[0], output_dimension).to(device)
            pred_segment = torch.zeros(y.shape[0], output_dimension).to(device)
            for i in range(x.shape[1]):
                # [batch size, window length, feature dimension]
                x_segment = x[:, i, :, :]
                # [window size, batch size,feature dimension]
                x_segment = x_segment.transpose(0, 1).contiguous()
                # [batch size, output dimension]
                pred = model(x_segment)
                pred_segment += pred
                # [window size, batch size, output dimension]
                y_segment = y[:, i, :, :].transpose(0, 1).contiguous()
                label = y_segment[y_segment.shape[0] // 2, :, :]
                label_segment += label
                if output:
                    for j in range(y.shape[0]):
                        writer.writerow([id, np.array(label[j,0]), np.array(label[j,1]), np.array(pred[j,0]),\
                                         np.array(pred[j,1])])
                        id += 1
            loss_mse += F.mse_loss(label_segment, pred_segment)*y.shape[0]
            loss_mape += mape_loss(label_segment, pred_segment)*y.shape[0]
            cnt += y.shape[0]
    if output:
        csvFile.close()
    return loss_mape/cnt, loss_mse/cnt


def train():

    # Create a new model or load an existing one.
    model = AttentionBlk(feature_dimension, head_number)
    if os.path.exists(ckpt_path):
        print('Reloading model parameters..')
        model.load_state_dict(torch.load(ckpt_path))
    else:
        print('Creating new model parameters..')
        # this code is very important! It initialises the parameters with a
        # range of values that stops the signal fading or getting too big.
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    model.to(device)
    print(model)
    p = sum(map(lambda p: p.numel(), model.parameters()))
    print("number of parameters:", p)

    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9)

    criterion = nn.MSELoss()

    best_mape, best_mse, best_epoch = float("inf"), float("inf"), 0

    for epoch in range(epochs):

        for step, (x, y) in enumerate(train_loader):
            # [batch size, path length, window length, feature dimension/output dimension]
            x, y = x.to(device), y.to(device)

            # For each batch, predict fuel consumption/time for each segment in a path and sum them
            # [batch size, output dimension]
            label_segment = torch.zeros(y.shape[0], output_dimension).to(device)
            pred_segment = torch.zeros(y.shape[0], output_dimension).to(device)

            # For each batch, predict fuel consumption/time for each segment in a path and sum them
            for i in range(x.shape[1]):
                # [batch size, window length, feature dimension]
                x_segment = x[:, i, :, :]

                # [window size, batch size,feature dimension]
                x_segment = x_segment.transpose(0, 1).contiguous()

                # [batch size, output dimension]
                pred = model(x_segment)

                # [batch size, output dimension]
                pred_segment += pred

                # [window size, batch size, output dimension]
                y_segment = y[:, i, :, :].transpose(0, 1).contiguous()

                # [batch size, output dimension]
                label = y_segment[y_segment.shape[0] // 2, :, :]

                # [batch size, output dimension]
                label_segment += label

            loss = criterion(label_segment, pred_segment)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch % 10 == 0:
            val_mape, val_mse = eval(model, val_loader)
            print("epoch:", epoch, "val_mape(%):", np.array(val_mape)*100, "val_mse:", np.array(val_mse))
            if val_mse < best_mse:
                best_epoch = epoch
                best_mse = val_mse
                torch.save(model.state_dict(), ckpt_path)

    print("best_epoch:", best_epoch, "best_mape(%):", np.array(best_mape)*100, "best_mse:", best_mse)


def test(output = False):
    """

    :param output: "True" -> output the estimation results to output_root
    :return:
    """

    # load an existing model.
    model = AttentionBlk(feature_dimension, head_number)
    if os.path.exists(ckpt_path):
        print('Reloading model parameters..')
        model.load_state_dict(torch.load(ckpt_path))
    else:
        print('Error: no existing model')
    model.to(device)
    print(model)
    p = sum(map(lambda p: p.numel(), model.parameters()))
    print("number of parameters:", p)
    test_mape, test_mse = eval(model, test_loader, output = output)
    print("test_mape(%):", np.array(test_mape)*100)
    print("test_mse:", np.array(test_mse))


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
    # main("test", output = True)
    # main("train")

