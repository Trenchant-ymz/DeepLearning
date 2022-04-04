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

import torch.profiler
import torch.utils.data

#from torchinterp1d import Interp1d
import math



# Before running the code, run 'python -m visdom.server' in the terminal to open visdom panel.

# pytorch profiler on tensorboard 'tensorboard --logdir=./log'

# Profiling: python -m cProfile -o profile.pstats multiTaskPINN.py
# Visualize profile: snakeviz profile.pstats

# divide a segment equally into n parts according to the length
lengthOfVelocityProfile = 60
tParts = lengthOfVelocityProfile # divide time into several parts

# mean std
meanOfSegmentLength = 608.2156661
stdOfSegmentLength = 900.4150229

# mean std
meanOfSegmentHeightChange = -0.063065664
stdOfSegmentHeightChange = 8.62278608

# mean std
meanOfSpeedLimit = 80.73990991
stdOfSpeedLimit = 21.5979505



meanOfMass = 23204.9788
stdOfMass = 8224.139199

# weight of acceleration loss (L1)
omega_acc = 0
# weight of jerk loss (MSE)
omega_jerk = 1e-3
#omega_jerk = 0
#omega_seg_fuel = 0.1
#omega_seg_time = omega_seg_fuel

omega_speedLimit = 0

omega_time = 0.4
omega_fuel = 0.6

# batch size 512 BEST
batchsz = 512

# learning rate
lr = 1e-3

# number of training epochs
epochs = 10
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
train_path_length = 20


# window size 3 best
window_sz = 3

# pace between paths
pace_train = 5
pace_test = 5



if pace_train>train_path_length:
    pace_train = train_path_length
    print("pace train has been changed to:", pace_train)



use_cuda = torch.cuda.is_available()
if use_cuda:
    print('Using GPU..')
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# local
ckpt_path = os.path.join(os.getcwd(),r"multitaskModels/pinnMultihead.mdl")
data_root = "model_data_newNov1perc"
#data_root = "ExpDataset/recursion20"
output_root = "prediction_result.csv"

# load data

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


def calLossOfPath(model,x,y,c,id , mode = 'time',output = False):
    segCriterion = nn.HuberLoss()
    #segCriterion = nn.L1Loss()
    pathCriterion = nn.MSELoss()
    maeCriterion = nn.L1Loss()
    accCriterion = nn.L1Loss()
    velCriterion = nn.L1Loss()
    jerkCriterion = nn.L1Loss()
    label_segment = torch.zeros(y.shape[0]).to(device)
    pred_segment = torch.zeros(y.shape[0]).to(device)
    if output:
        csvFile = open(output_root, "a")
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
        label = y[:, i, window_sz // 2]

        # [batch size, output dimension]
        label_segment += label

        # [batch size, lengthOfVelocityProfile]
        # offset to make sure the average velocity is higher than 0
        velocityProfile = model(x_segment, c_segment, id_segment)

        # extract the length of this segment
        # [batch size]
        length = denormalize(x_segment[:, window_sz // 2, 4], meanOfSegmentLength, stdOfSegmentLength)
        height = denormalize(x_segment[:, window_sz // 2, 2], meanOfSegmentHeightChange, stdOfSegmentHeightChange)
        speedLimit = denormalize(x_segment[:, window_sz // 2, 0], meanOfSpeedLimit, stdOfSpeedLimit)/3.6

        m = denormalize(x_segment[:, window_sz // 2, 1], meanOfMass, stdOfMass).unsqueeze(-1)
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
                        [id_segment[j, window_sz // 2].item(), "-","-", np.array(label[j].cpu()), np.array(pred[j].cpu())])
        else:
            pred = fuelEstimation(v, t, acc, m, height, length)
            if output:
                for j in range(y.shape[0]):
                    writer.writerow(
                        [id_segment[j, window_sz // 2].item(), np.array(label[j].cpu()), np.array(pred[j].cpu()), "-","-"])
        # [batch size, output dimension]

        if i == 0 :
            seg_loss = segCriterion(label,pred)
            acc_loss = accCriterion(acc, zeros).to(device)
            jerk_loss = jerkCriterion(jerk, zeros).to(device)
            vel_loss = velCriterion(F.relu(v-speedLimit),zeros).to(device)
        else:
            seg_loss += segCriterion(label,pred)
            acc_loss += accCriterion(acc, zeros).to(device)
            jerk_loss += jerkCriterion(jerk, zeros).to(device)
            vel_loss += velCriterion(F.relu(v - speedLimit), zeros).to(device)
        pred_segment += pred
        # label_segment_denormalized += denormalize(label)

    if output:
        csvFile.close()
    mape = mape_loss(label_segment, pred_segment)
    mae = maeCriterion(label_segment, pred_segment)
    mse = pathCriterion(label_segment, pred_segment)
    coefficient = omega_time if mode == 'time' else omega_fuel
    if coefficient != 0:
        totalLoss = mape + (seg_loss + (omega_acc * acc_loss + omega_jerk * jerk_loss + omega_speedLimit * vel_loss) / coefficient)/x.shape[1]
    else:
        totalLoss = mape + (seg_loss + (omega_acc * acc_loss + omega_jerk * jerk_loss + omega_speedLimit * vel_loss) / 1)/x.shape[1]
    #totalLoss = mse + (seg_loss + (omega_acc * acc_loss + omega_jerk * jerk_loss + omega_speedLimit * vel_loss) / coefficient) / x.shape[1]
    #print('mse',mse,seg_loss,jerk_loss)
    return totalLoss, mape,  mse, mae, y.shape[0]
    #return mse , mape, y.shape[0]




def train():
    train_db_fuel = ObdData(root=data_root, mode="train", fuel=True, percentage=20, window_size=window_sz,
                            path_length=train_path_length, label_dimension=1, pace=pace_train,
                            withoutElevation=False)
    train_db_time = ObdData(root=data_root, mode="train", fuel=False, percentage=20, window_size=window_sz,
                            path_length=train_path_length, label_dimension=1, pace=pace_train,
                            withoutElevation=False)
    train_sampler_fuel = torch.utils.data.RandomSampler(train_db_fuel, replacement=True, num_samples=len(train_db_time),
                                                        generator=None)
    train_loader_fuel = DataLoader(train_db_fuel, sampler=train_sampler_fuel, batch_size=batchsz, num_workers=0)
    train_loader_time = DataLoader(train_db_time, batch_size=batchsz, num_workers=0)

    # Create a new model or load an existing one.
    model = Pigat(feature_dim=feature_dimension, embedding_dim=[4, 2, 2, 2, 2, 4, 4], num_heads=head_number,
                  output_dimension=lengthOfVelocityProfile, n2v_dim=32)
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
    # print(dict(model.named_parameters()))
    p = sum(map(lambda p: p.numel(), model.parameters()))
    print("number of parameters:", p)
    start = time.perf_counter()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, betas=(0.9, 0.98), eps=1e-9)
    schedule = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)
    train_loss = []
    # prof = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA],
    #                               schedule=torch.profiler.schedule(wait=1, warmup=1, active=2),
    #                               on_trace_ready=torch.profiler.tensorboard_trace_handler('./result', worker_name='worker2'),
    #                               record_shapes=True, profile_memory=False, with_stack=True)
    for epoch in range(epochs):
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
            loss = omega_fuel*mse_fuel + omega_time*mse_time
            #print(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_loss.append(loss_mape_fuel.item())
        schedule.step(loss)

    fast_elapsed_seconds = time.perf_counter() - start
    print(f'Custom dataloader: {fast_elapsed_seconds / epochs:.2f}s/epoch.')


train()