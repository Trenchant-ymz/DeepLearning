import torch
from torch import nn, optim
import os
from torch.utils.data import DataLoader
from obddata import ObdData
from pigat import Pigat
import torch.nn.functional as F
import numpy as np
import csv
import time
import visdom
from tqdm import tqdm
from typing import List
import torch.profiler
import torch.utils.data
import config
import pandas as pd
from obddataPreprocessing import loadData
#from torchinterp1d import Interp1d
import math



# Before running the code, run 'python -m visdom.server' in the terminal to open visdom panel.

# pytorch profiler on tensorboard 'tensorboard --logdir=./log'

# Profiling: python -m cProfile -o profile.pstats multiTaskPINN.py
# Visualize profile: snakeviz profile.pstats



use_cuda = torch.cuda.is_available()
if use_cuda:
    print('Using GPU..')
    device = torch.device("cuda")
else:
    device = torch.device("cpu")



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

# #version 1 vd=>vt=>Same time interpolation
# def vd2vtWithInterpolation(velocityProfile, length):
#     lengthOfEachPart = length / (velocityProfile.shape[1] - 1)
#     averageV = velocityProfile[:,0:-1] + velocityProfile[:,1:]
#     tOnSubPath = (2 * lengthOfEachPart).unsqueeze(-1) / averageV
#     tAxis = torch.matmul(tOnSubPath, torch.triu(torch.ones(tOnSubPath.shape[1],tOnSubPath.shape[1])).to(device))
#     tAxis = torch.cat([torch.zeros([tOnSubPath.shape[0],1]).to(device),tAxis],dim=-1)
#     timeSum = tAxis[:, -1]
#     tNew = torch.arange(tParts).unsqueeze(0).to(device) * timeSum.unsqueeze(-1) / (tParts - 1)
#     #print('tAxis',tAxis,'velocityProfile',velocityProfile, 'tNew',tNew)
#     intetpResult = None
#     intetpResult = Interp1d()(tAxis, velocityProfile, tNew, intetpResult).to(device)
#     ##plot interpolation
#     # x = tAxis.cpu().numpy()
#     # y = velocityProfile.cpu().numpy()
#     # xnew = tNew.cpu().numpy()
#     # yq_cpu = intetpResult.cpu().numpy()
#     # print(x.T, y.T, xnew.T, yq_cpu.T)
#     # plt.plot(x.T, y.T, '-', xnew.T, yq_cpu.T, 'x')
#     # plt.grid(True)
#     # plt.show()
#     return intetpResult,tNew

# #version 1 vd=>vt
# def vd2vt(velocityProfile, length):
#     '''
#
#     :param velocityProfile: velocityProfile: velocity profiel (uniform length sampling)
#     :param length: length: total length of the segment
#     :return: tAxis: the axis of time of the velocity profile
#     '''
#     lengthOfEachPart = length / (velocityProfile.shape[1] - 1)
#     averageV = velocityProfile[:,0:-1] + velocityProfile[:,1:]
#     tOnSubPath = (2 * lengthOfEachPart).unsqueeze(-1) / averageV
#     tAxis = torch.matmul(tOnSubPath, torch.triu(torch.ones(tOnSubPath.shape[1],tOnSubPath.shape[1])).to(device))
#     tAxis = torch.cat([torch.zeros([tOnSubPath.shape[0],1]).to(device),tAxis],dim=-1)
#     return velocityProfile,tAxis


#version 1 vt=>calculate t
def vt2t(velocityProfile, length):
    '''

    :param velocityProfile: velocity profiel (uniform time sampling)
    :param length: total length of the segment
    :return: tAxis: the axis of time of the velocity profile
    '''
    mul = torch.cat([torch.ones([1, 1]), 2 * torch.ones([velocityProfile.shape[1] - 2, 1]),\
                       torch.ones([1, 1])], dim=0).to(device)
    # mul =  2 * torch.ones([velocityProfile.shape[1], 1]).to(device)
    # mul[0,0] = mul[-1,0]= 1
    vAverage = torch.matmul(velocityProfile, mul) / 2
    # vAverage_tensor = (velocityProfile[:, :-1] + velocityProfile[:, 1:]) / 2
    # vAverage = torch.sum(vAverage_tensor, dim=1).unsqueeze(-1)
    #print(vAverage_1, vAverage)
    #assert torch.equal(vAverage_1, vAverage)
    tForOnePart = length.unsqueeze(-1) / vAverage
    tAxis = torch.arange(velocityProfile.shape[1]).unsqueeze(0).to(device) * tForOnePart
    return velocityProfile,tAxis


def timeEstimation(tNew):
    return tNew[:, -1]


def fuelEstimation(v, tNew, acc, m, height, length):
    sin_theta = height / length
    fuel = vt2fuel(v, acc, tNew, m, sin_theta)
    return fuel

def vt2a(v,t):
    # calculate a using 1-3, 2-4
    dv = v[:, 2:] - v[:, 0:-2]
    dt = t[:, 2].unsqueeze(-1)
    aMiddle = dv / dt
    dvHead = v[:, 1] - v[:, 0]
    dtSingle = t[:, 1]
    aHead = (dvHead / dtSingle).unsqueeze(-1)
    dvTail = v[:, -1] - v[:, -2]
    aTail = (dvTail / dtSingle).unsqueeze(-1)
    a = torch.cat([aHead, aMiddle, aTail], dim=-1)
    return a


def power(v,a,m,sin_theta,rho):
    R = 0.5003  # wheel radius (m)
    g = 9.81  # gravitational accel (m/s^2)
    A = 10.5  # frontal area (m^2)
    Cd = 0.5  # drag coefficient
    Crr = 0.0067  # rolling resistance
    Iw = 10  # wheel inertia (kg m^2)
    Nw = 10  # number of wheels

    Paccel = (m * a * v).clamp(0)
    #Pascent =(m * g * torch.sin(torch.tensor([theta * (math.pi / 180)]).to(device)) * v).clamp(0)
    Pascent = (m * g * sin_theta.unsqueeze(-1) * v).clamp(0)
    Pdrag = (0.5 * rho * Cd * A * v ** 3).clamp(0)
    Prr = (m * g * Crr * v).clamp(0)
    #Pinert = (Iw * Nw * (a / R) * v).clamp(0)
    # pauxA = torch.zeros(Pinert.shape).to(device)
    Paux = 1000
    #Paux = torch.where(v>0.1, pauxA, pauxB).to(device)
    #Paux = 0
    P = (Paccel + Pascent + Pdrag + Prr +Paux) / 1000
    return P


def vt2fuel(v,a,t,m,sin_theta ):

    #m = 10000  # mass (kg)
    rho = 1.225  # density of air (kg/m^3)
    #theta = 0  # road grade (degrees)
    fc = 40.3  # fuel consumption (kWh/gal) # Diesel ~40.3 kWh/gal
    eff = 0.56  # efficiency of engine

    P = power(v, a, m, sin_theta, rho)
    P_avg = (P[:, :-1] + P[:, 1:]) / 2
    f = P_avg / (fc * eff) * t[:, 1].unsqueeze(-1) / 3600
    #from galon => 10ml
    return torch.sum(f, dim=1)*3.7854*100

class FastTensorDataLoader:
    """
    A DataLoader-like object for a set of tensors that can be much faster than
    TensorDataset + DataLoader because dataloader grabs individual indices of
    the dataset and calls cat (slow).
    Source: https://discuss.pytorch.org/t/dataloader-much-slower-than-manual-batching/27014/6
    """
    def __init__(self, *tensors, batch_size=32, shuffle=False):
        """
        Initialize a FastTensorDataLoader.
        :param *tensors: tensors to store. Must have the same length @ dim 0.
        :param batch_size: batch size to load.
        :param shuffle: if True, shuffle the data *in-place* whenever an
            iterator is created out of this object.
        :returns: A FastTensorDataLoader.
        """
        #print(tensors)
        assert all(t.shape[0] == tensors[0].shape[0] for t in tensors)
        self.tensors = tensors
        self.dataset_len = self.tensors[0].shape[0]
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Calculate # batches
        n_batches, remainder = divmod(self.dataset_len, self.batch_size)
        if remainder > 0:
            n_batches += 1
        self.n_batches = n_batches
    def __iter__(self):
        if self.shuffle:
            r = torch.randperm(self.dataset_len)
            self.tensors = [t[r] for t in self.tensors]
        self.i = 0
        return self

    def __next__(self):
        if self.i >= self.dataset_len:
            raise StopIteration
        batch = tuple(t[self.i:self.i+self.batch_size] for t in self.tensors)
        self.i += self.batch_size
        return batch

    def __len__(self):
        return self.n_batches


def randomsampler(*tensors, num_samples):
    dataset_len = tensors[0].shape[0]
    r = torch.randint(0, dataset_len, (num_samples,))
    tensors = [t[r] for t in tensors]
    return tensors


def calLossOfPath(model,x,y,c,id , mode = 'time',output = False):
    segCriterion = nn.HuberLoss()
    #segCriterion = nn.L1Loss()
    pathCriterion = nn.MSELoss()
    maeCriterion = nn.L1Loss()
    #accCriterion = nn.L1Loss()
    velCriterion = nn.L1Loss()
    jerkCriterion = nn.MSELoss()
    label_segment = torch.zeros(y.shape[0]).to(device)
    pred_segment = torch.zeros(y.shape[0]).to(device)
    if output:
        csvFile = open(config.params.output_root, "a")
        writer = csv.writer(csvFile)
        writer.writerow(["id", "ground truth fuel(l)", "estimated fuel (l)", "ground truth time (s)", "estimated time (s)"])
    for i in range(x.shape[1]):
        # [batch size, window length, feature dimension]
        x_segment = x[:, i, :, :]
        # [batch, categorical_dim, window size]
        c_segment = c[:, :, i, :]
        # [batch, window size]
        id_segment = id[:, i, :]

        # [batch size, output dimension]
        label = y[:, i, config.params.window_sz // 2]

        # [batch size, output dimension]
        label_segment += label

        # [batch size, lengthOfVelocityProfile]
        # offset to make sure the average velocity is higher than 0
        velocityProfile = model(x_segment, c_segment, id_segment)

        # extract the length of this segment
        # [batch size]
        length = denormalize(x_segment[:, config.params.window_sz // 2, 4], config.params.meanOfSegmentLength, config.params.stdOfSegmentLength)
        height = denormalize(x_segment[:, config.params.window_sz // 2, 2], config.params.meanOfSegmentHeightChange, config.params.stdOfSegmentHeightChange)
        speedLimit = denormalize(x_segment[:, config.params.window_sz // 2, 0], config.params.meanOfSpeedLimit, config.params.stdOfSpeedLimit)/3.6

        m = denormalize(x_segment[:, config.params.window_sz // 2, 1], config.params.meanOfMass, config.params.stdOfMass).unsqueeze(-1)
        v, t = vt2t(velocityProfile, length)
        acc = vt2a(v, t)
        jerk = vt2a(acc, t)
        zeros = torch.zeros(acc.shape).to(device)
        speedLimit = torch.ones(v.shape).to(device) * speedLimit.unsqueeze(-1)
        #print('v/acc/jerk',v,acc,jerk)


        if mode == 'time':
            pred = timeEstimation(t)
            if output:
                for j in range(y.shape[0]):
                    writer.writerow(
                        [id_segment[j, config.params.window_sz // 2].item(), "-","-", np.array(label[j].cpu()), np.array(pred[j].cpu())])
        else:
            pred = fuelEstimation(v, t, acc, m, height, length)
            if output:
                for j in range(y.shape[0]):
                    writer.writerow(
                        [id_segment[j, config.params.window_sz // 2].item(), np.array(label[j].cpu()), np.array(pred[j].cpu()), "-","-"])
        # [batch size, output dimension]

        if i == 0 :
            seg_loss = segCriterion(label,pred)
            #acc_loss = accCriterion(acc, zeros).to(device)
            jerk_loss = jerkCriterion(jerk, zeros).to(device)
            vel_loss = velCriterion(F.relu(v-speedLimit),zeros).to(device)
        else:
            seg_loss += segCriterion(label,pred)
            #acc_loss += accCriterion(acc, zeros).to(device)
            jerk_loss += jerkCriterion(jerk, zeros).to(device)
            vel_loss += velCriterion(F.relu(v - speedLimit), zeros).to(device)
        pred_segment += pred
        # label_segment_denormalized += denormalize(label)

    if output:
        csvFile.close()
    mape = mape_loss(label_segment, pred_segment)
    mae = maeCriterion(label_segment, pred_segment)
    mse = pathCriterion(label_segment, pred_segment)
    coefficient = config.params.omega_time if mode == 'time' else config.params.omega_fuel
    if coefficient != 0:
        #totalLoss = mape + (seg_loss  / coefficient) /  x.shape[1]
        totalLoss = mape + (seg_loss + config.params.omega_jerk * jerk_loss  / coefficient)/x.shape[1]
    else:
        totalLoss = mape + (seg_loss + config.params.omega_jerk * jerk_loss  / 1)/x.shape[1]
    #print('mse',mse,seg_loss,jerk_loss)
    return totalLoss, mape,  mse, mae, y.shape[0]
    #return mse , mape, y.shape[0]


def eval(model, loader_time, loader_fuel, output = False):
    """
    Evaluate the model accuracy by MAPE of the energy consumption & average MSE of energy and time
    :param model: trained model
    :param loader: data loader
    :param output: output the estimation result or not
    :return: MAPE of energy, average MSE of energy and time
    """

    if output:
        csvFile = open(config.params.output_root, "w")
        writer = csv.writer(csvFile)
        writer.writerow(["id", "ground truth fuel(l)", "ground truth time (s)", "estimated fuel (l)", "estimated time (s)"])
    loss_mape_fuel_total = 0
    loss_mape_time_total = 0
    loss_total = 0
    loss_fuel_total = 0
    loss_time_total = 0
    mse_time_total = 0
    mae_time_total = 0
    mse_fuel_total = 0
    mae_fuel_total = 0
    cnt = 0
    identity = 0
    model.eval()
    for (xt, yt, ct, idt),(xf,yf,cf,idf) in zip(loader_time,loader_fuel):
        #x, y, c = x.to(device), y.to(device), c.to(device)
        with torch.no_grad():
            # loss_time is not the true mse
            loss_time, loss_mape_time, mse_time, mae_time,  cnt_add = calLossOfPath(model, xt, yt, ct, idt, mode ='time', output=output)
            loss_fuel, loss_mape_fuel, mse_fuel, mae_fuel,  cnt_add = calLossOfPath(model, xf, yf, cf, idf, mode='fuel', output=output)
            loss_fuel_total += loss_fuel * cnt_add
            loss_time_total += loss_time * cnt_add
            loss_total += config.params.omega_fuel * loss_fuel * cnt_add + config.params.omega_time * loss_time * cnt_add
            loss_mape_time_total += loss_mape_time * cnt_add
            loss_mape_fuel_total += loss_mape_fuel * cnt_add
            mse_time_total += mse_time * cnt_add
            mae_time_total += mae_time * cnt_add
            mse_fuel_total += mse_fuel * cnt_add
            mae_fuel_total += mae_fuel * cnt_add
            #print(label_segment, pred_segment, mape_loss(label_segment, pred_segment))
            cnt += cnt_add
    return loss_total/cnt, loss_fuel_total/cnt, loss_time_total/cnt, loss_mape_fuel_total/cnt, loss_mape_time_total/cnt , mse_fuel_total/cnt, mse_time_total/cnt, mae_fuel_total/cnt, mae_time_total/cnt




def train():
    # random seed
    seed = 1234
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    x_TimeTrain, y_TimeTrain, c_TimeTrain, id_TimeTrain = loadData(root=config.params.data_root, mode="train", fuel=False, percentage=20,
                                                                   window_size=config.params.window_sz,path_length=config.params.train_path_length,
                                                                   label_dimension=1, pace=config.params.pace_train, withoutElevation=False)
    train_loader_time = FastTensorDataLoader(x_TimeTrain, y_TimeTrain, c_TimeTrain, id_TimeTrain, batch_size=config.params.batchsz)

    x_FuelTrain, y_FuelTrain, c_FuelTrain, id_FuelTrain = loadData(root=config.params.data_root, mode="train", fuel=True, percentage=20,
                                                                   window_size=config.params.window_sz,path_length=config.params.train_path_length,
                                                                   label_dimension=1, pace=config.params.pace_train, withoutElevation=False)
    x_FuelTrain, y_FuelTrain, c_FuelTrain, id_FuelTrain = randomsampler(x_FuelTrain, y_FuelTrain, c_FuelTrain, id_FuelTrain, num_samples=x_TimeTrain.shape[0])
    train_loader_fuel = FastTensorDataLoader(x_FuelTrain, y_FuelTrain, c_FuelTrain, id_FuelTrain, batch_size=config.params.batchsz)

    # train_db_fuel = ObdData(root=config.params.data_root, mode="train", fuel=True, percentage=20, window_size=config.params.window_sz,\
    #                    path_length=config.params.train_path_length, label_dimension=1, pace=config.params.pace_train,
    #                    withoutElevation=False)
    # train_db_time = ObdData(root=config.params.data_root, mode="train", fuel=False, percentage=20, window_size=config.params.window_sz,\
    #                    path_length=config.params.train_path_length, label_dimension=1, pace=config.params.pace_train,
    #                    withoutElevation=False)
    # train_sampler_fuel = torch.utils.data.RandomSampler(train_db_fuel, replacement=True, num_samples=len(train_db_time),
    #                                                generator=None)
    # train_loader_fuel = DataLoader(train_db_fuel, sampler = train_sampler_fuel, batch_size=config.params.batchsz, num_workers=0)
    # train_loader_time = DataLoader(train_db_time, batch_size=config.params.batchsz, num_workers=0)


    # val_db_fuel = ObdData(root=config.params.data_root, mode="val", fuel=True, percentage=20, window_size=config.params.window_sz,\
    #                  path_length=config.params.train_path_length, label_dimension=1, pace=config.params.pace_test,
    #                  withoutElevation=False)
    # val_db_time = ObdData(root=config.params.data_root, mode="val", fuel=False, percentage=20, window_size=config.params.window_sz, \
    #                       path_length=config.params.train_path_length, label_dimension=1, pace=config.params.pace_test,
    #                       withoutElevation=False)
    # val_sampler_fuel = torch.utils.data.RandomSampler(val_db_fuel, replacement=True, num_samples=len(val_db_time),
    #                                          generator=None)
    #
    # val_loader_fuel = DataLoader(val_db_fuel, sampler = val_sampler_fuel, batch_size=config.params.batchsz, num_workers=0)
    # val_loader_time = DataLoader(val_db_time, batch_size=config.params.batchsz, num_workers=0)

    x_TimeVal, y_TimeVal, c_TimeVal, id_TimeVal = loadData(root=config.params.data_root, mode="val", fuel=False, percentage=20,
                                                                   window_size=config.params.window_sz,path_length=config.params.train_path_length,
                                                                   label_dimension=1, pace=config.params.pace_test, withoutElevation=False)
    val_loader_time = FastTensorDataLoader(x_TimeVal, y_TimeVal, c_TimeVal, id_TimeVal, batch_size=config.params.batchsz)

    x_FuelVal, y_FuelVal, c_FuelVal, id_FuelVal = loadData(root=config.params.data_root, mode="val", fuel=True, percentage=20,
                                                                   window_size=config.params.window_sz,path_length=config.params.train_path_length,
                                                                   label_dimension=1, pace=config.params.pace_test, withoutElevation=False)
    x_FuelVal, y_FuelVal, c_FuelVal, id_FuelVal = randomsampler(x_FuelVal, y_FuelVal, c_FuelVal, id_FuelVal, num_samples=x_TimeVal.shape[0])
    val_loader_fuel = FastTensorDataLoader(x_FuelVal, y_FuelVal, c_FuelVal, id_FuelVal, batch_size=config.params.batchsz)

    viz = visdom.Visdom()
    # Create a new model or load an existing one.
    model = Pigat(feature_dim=config.params.feature_dimension,embedding_dim=[4,2,2,2,2,4,4],num_heads=config.params.head_number,
                  output_dimension=config.params.lengthOfVelocityProfile, n2v_dim=config.params.n2v_dim,window_size=config.params.window_sz)
    print('Creating new model parameters..')
    # this code is very important! It initialises the parameters with a
    # range of values that stops the signal fading or getting too big.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    model.to(device)
    model.n2v.load_state_dict(torch.load('node2vec.mdl'))
    model.n2v.embedding.weight.requires_grad = False
    print(model)
    print(next(model.parameters()).device)
    #print(dict(model.named_parameters()))
    p = sum(map(lambda p: p.numel(), model.parameters()))
    print("number of parameters:", p)

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config.params.lr,
                           betas=(config.params.beta_1, config.params.beta_2), eps=config.params.eps)
    schedule = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5,
                                                    patience=config.params.patienceOfTrainingEpochs)
    train_loss = []
    #for p in filter(lambda p: p.requires_grad, model.parameters()): print(p)
    #criterion = nn.MSELoss()
    global_step = 0
    best_mape, best_mse, best_epoch = torch.tensor(float("inf")), torch.tensor(float("inf")), 0
    viz.line([0],[-1], win='train_mse', opts=dict(title='train_mse'))
    viz.line([0], [-1], win='val_mse', opts=dict(title='val_mse'))
    viz.line([0], [-1], win='learning rate', opts=dict(title='learning rate'))
    viz.line([0], [-1], win='train_time_mse', opts=dict(title='train_time_mse'))
    viz.line([0], [-1], win='train_fuel_mse', opts=dict(title='train_fuel_mse'))
    viz.line([0], [-1], win='val_time_mse', opts=dict(title='val_time_mse'))
    viz.line([0], [-1], win='val_fuel_mse', opts=dict(title='val_fuel_mse'))
    viz.line([0], [-1], win='train_time_mape', opts=dict(title='train_time_mape'))
    viz.line([0], [-1], win='train_fuel_mape', opts=dict(title='train_fuel_mape'))
    viz.line([0], [-1], win='val_time_mape', opts=dict(title='val_time_mape'))
    viz.line([0], [-1], win='val_fuel_mape', opts=dict(title='val_fuel_mape'))
    # prof = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA],
    #                               schedule=torch.profiler.schedule(wait=1, warmup=1, active=2),
    #                               on_trace_ready=torch.profiler.tensorboard_trace_handler('./result', worker_name='worker2'),
    #                               record_shapes=True, profile_memory=False, with_stack=True)
    #with torch.autograd.profiler.profile(with_stack=True, profile_memory=True, use_cuda=True) as prof:
    for epoch in range(config.params.max_epochs):

        model.train()
        #prof.start()
        for step, ((xt, yt, ct, idt),(xf,yf,cf,idf)) in tqdm(enumerate(zip(train_loader_time,train_loader_fuel))):
            # x: numerical features [batch, path length, window size, feature dimension]
            # y: label [batch, path length, window size, (label dimension)]
            # c: categorical features [batch, number of categorical features, path length, window size]
            #x, y, c = x.to(device), y.to(device), c.to(device)

            # For each batch, predict fuel consumption/time for each segment in a path and sum them
            # [batch size, output dimension]

            mse_time, loss_mape_time, _,_,_ = calLossOfPath(model, xt, yt, ct, idt, mode='time')
            mse_fuel, loss_mape_fuel, _,_,_ = calLossOfPath(model, xf, yf, cf, idf, mode='fuel')

            #loss = mape_loss(label_path, pred_path)
            #print('loss',loss_fuel,loss_time)
            loss = config.params.omega_fuel*mse_fuel + config.params.omega_time*mse_time
            #print(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        #     prof.step()
        # prof.stop()
            #print(dict(model.named_parameters()))

        viz.line([loss.item()], [global_step], win='train_mse', update='append')
        viz.line([mse_time.item()], [global_step], win='train_fuel_mse', update='append')
        viz.line([mse_fuel.item()], [global_step], win='train_time_mse', update='append')


        viz.line([loss_mape_fuel.item()], [global_step], win='train_fuel_mape', update='append')
        viz.line([loss_mape_time.item()], [global_step], win='train_time_mape', update='append')
        train_loss.append(loss.item())
        schedule.step(loss)
        learning_rate = optimizer.state_dict()['param_groups'][0]['lr']

        viz.line([learning_rate], [global_step], win='learning rate', update='append')

        # print("epoch:", epoch, "test_mse:", loss)
        if epoch % 1 == 0:
            val_mse, val_loss_fuel,val_loss_time, val_fuel_mape, val_time_mape, mse_fuel, mse_time, mae_fuel, mae_time = eval(model,val_loader_time, val_loader_fuel)
            # schedule.step(val_mse)
            print("epoch:", epoch, "val_mape_fuel(%):", np.array(val_fuel_mape.cpu())*100,"val_mape_time(%):", np.array(val_time_mape.cpu())*100, "val_mse:", np.array(val_mse.cpu()))
            print("epoch:", epoch,  "train_mse:", loss.item())
            if val_mse < best_mse:
                best_epoch = epoch
                best_mse = val_mse
                torch.save(model.state_dict(), config.params.ckpt_path)
            viz.line([val_mse.item()], [global_step], win='val_mse', update='append')
            viz.line([mse_fuel.item()], [global_step], win='val_fuel_mse', update='append')
            viz.line([mse_time.item()], [global_step], win='val_time_mse', update='append')
            viz.line([val_fuel_mape.item()], [global_step], win='val_fuel_mape', update='append')
            viz.line([val_time_mape.item()], [global_step], win='val_time_mape', update='append')

        global_step += 1
        if learning_rate < 1e-3:
            break
    print("best_epoch:", best_epoch, "best_mse:", np.array(best_mse.cpu()))
    np.savetxt('trainLossPiNNwithoutJerk.csv', train_loss, delimiter=',')
    print("training epochs: {}".format(len(train_loss)))
    return len(train_loss)
    #print(prof.key_averages(group_by_stack_n=5).table(sort_by='self_cpu_time_total', row_limit=5))


def test(model, test_path_length, test_pace, output = False):
    """

    :param output: "True" -> output the estimation results to output_root
    :return:
    """

    # load an existing model.
    test_db_time = ObdData(root=config.params.data_root, mode="test",fuel=False, percentage=20, window_size=config.params.window_sz,\
                      path_length=test_path_length, label_dimension=1, pace=test_pace,
                      withoutElevation=False)
    test_db_fuel = ObdData(root=config.params.data_root, mode="test",fuel=True, percentage=20, window_size=config.params.window_sz, \
                        path_length=test_path_length, label_dimension=1, pace=test_pace,
                      withoutElevation=False)

    test_loader_time = DataLoader(test_db_time, batch_size=config.params.batchsz, num_workers=0)
    test_loader_fuel = DataLoader(test_db_fuel, batch_size=config.params.batchsz, num_workers=0)
    #print(model)
    #p = sum(map(lambda p: p.numel(), model.parameters()))
    #print("number of parameters:", p)
    test_mse,_,_, test_fuel_mape, test_time_mape, mse_fuel, mse_time, mae_fuel, mae_time = eval(model, test_loader_time, test_loader_fuel, output=output)
    print("test_mape_fuel(%):", np.array(test_fuel_mape.cpu()) * 100)
    #print("test_mape_time(%):", np.array(test_time_mape.cpu()) * 100)
    #print("test_mse_fuel:", np.array(mse_fuel.cpu())**0.5/100)
    #print("test_mse_time:", np.array(mse_time.cpu()))
    #print("test_mae_fuel:", np.array(mae_fuel.cpu())/100)
    return np.array(test_fuel_mape.cpu()) * 100
    #print("test_mae_time:", np.array(mae_time.cpu()))


def trainTest(mode, output = False):
    """
    :param mode: "train" for training, "test" for testing the existing model or predicting
    :return:
    """

    if mode == "train":
        train()
        return
    elif mode == "test":
        model = Pigat(feature_dim=config.params.feature_dimension, embedding_dim=[4, 2, 2, 2, 2, 4, 4], num_heads=config.params.head_number,
                             output_dimension=config.params.lengthOfVelocityProfile, n2v_dim=32, window_size=config.params.window_sz)
        if os.path.exists(config.params.ckpt_path):
            print('Reloading model parameters..')
            model.load_state_dict(torch.load(config.params.ckpt_path, map_location=device))
        else:
            print('Error: no existing model')
        model.to(device)
        fuelMapeList = []
        # length of path used for test
        test_path_length_list = [1,2,5,10,20,50,100,200]
        for length in test_path_length_list:
            pace_test = config.params.pace_test
            if  pace_test > length:
                pace_test = length
                print("pace test has been changed to:", pace_test)
            print("test path length:",length)
            test_fuel_mape = test(model, length,pace_test,output = output)
            fuelMapeList.append(test_fuel_mape)
        return fuelMapeList
    else:
        return

def gridSearch(windowSizeList: List[int], jerkPenaltyList: List[float], ecotollWeightList: List[float]):
    for i in range(11,21):
        outputFileName = 'mapeOfAllParams/mapeForDataSet{}.csv'.format(i-10)
        config.params.data_root = "ExpDataset/recursion{}".format(i)
        dfOfMape = pd.DataFrame(columns=['Epochs', 'path length 1', 'path length 2',
                                                            'path length 5', 'path length 10', 'path length 20',
                                                            'path length 50', 'path length 100', 'path length 200'])
        indexList= []
        for windowSize in windowSizeList:
            for jerkPenalty in jerkPenaltyList:
                for ecotollWeight in ecotollWeightList:
                    config.params.window_sz = windowSize
                    config.params.omega_jerk = jerkPenalty
                    config.params.omega_fuel = ecotollWeight
                    config.params.omega_time = 1 - ecotollWeight
                    print('windowsize:{wsz},jerkWeight:{jerk},ecotollWeight:{ecotollWeight},Dataset:{numberOfDataset}'\
                          .format(wsz=windowSize, jerk=jerkPenalty, ecotollWeight=ecotollWeight, numberOfDataset=i-10))
                    trainEpochs = train()
                    mapeList = trainTest(mode='test')
                    dfOfMape = pd.concat([dfOfMape, pd.DataFrame([[trainEpochs] + mapeList], columns=dfOfMape.columns)], ignore_index=True)
                    indexList.append('windowsz{wsz}jerk{jerk}ecotoll{ecotollWeight}'.format(wsz=windowSize,jerk=jerkPenalty, ecotollWeight =ecotollWeight ))
        dfOfMape.index = indexList
        dfOfMape.to_csv(outputFileName)



if __name__ == '__main__':
    #trainTest(input("mode="), output = False)
    gridSearch(windowSizeList=[1,3,5], jerkPenaltyList=[0, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7], ecotollWeightList=[0, 0.4, 0.6, 0.8, 1])
    #main("test")
    #main("train")
    # main("test", output = True)


