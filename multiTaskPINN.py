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
head_number = 16
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
    #print('Using GPU..')
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# local
ckpt_path = os.path.join(os.getcwd(),"multitaskModels/pinn20MAE.mdl")
#data_root = "model_data_newNov1perc"
data_root = "ExpDataset/recursion20"
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

#version 1 vd=>vt
def vd2vt(velocityProfile, length):
    '''

    :param velocityProfile: velocityProfile: velocity profiel (uniform length sampling)
    :param length: length: total length of the segment
    :return: tAxis: the axis of time of the velocity profile
    '''
    lengthOfEachPart = length / (velocityProfile.shape[1] - 1)
    averageV = velocityProfile[:,0:-1] + velocityProfile[:,1:]
    tOnSubPath = (2 * lengthOfEachPart).unsqueeze(-1) / averageV
    tAxis = torch.matmul(tOnSubPath, torch.triu(torch.ones(tOnSubPath.shape[1],tOnSubPath.shape[1])).to(device))
    tAxis = torch.cat([torch.zeros([tOnSubPath.shape[0],1]).to(device),tAxis],dim=-1)
    return velocityProfile,tAxis


#version 1 vt=>calculate t
def vt2t(velocityProfile, length):
    '''

    :param velocityProfile: velocity profiel (uniform time sampling)
    :param length: total length of the segment
    :return: tAxis: the axis of time of the velocity profile
    '''
    mul = torch.cat([torch.ones([1, 1]), 2 * torch.ones([velocityProfile.shape[1] - 2, 1]),\
                       torch.ones([1, 1])], dim=0).to(device)
    vAverage = torch.matmul(velocityProfile, mul) / 2
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


def eval(model, loader_time, loader_fuel, output = False):
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
            loss_total += omega_fuel * loss_fuel * cnt_add + omega_time * loss_time * cnt_add
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
    train_db_fuel = ObdData(root=data_root, mode="train", fuel=True, percentage=20, window_size=window_sz,\
                       path_length=train_path_length, label_dimension=1, pace=pace_train,
                       withoutElevation=False)
    train_db_time = ObdData(root=data_root, mode="train", fuel=False, percentage=20, window_size=window_sz,\
                       path_length=train_path_length, label_dimension=1, pace=pace_train,
                       withoutElevation=False)
    train_sampler_fuel = torch.utils.data.RandomSampler(train_db_fuel, replacement=True, num_samples=len(train_db_time),
                                                   generator=None)
    train_loader_fuel = DataLoader(train_db_fuel, sampler = train_sampler_fuel, batch_size=batchsz, num_workers=0)
    train_loader_time  = DataLoader(train_db_time, batch_size=batchsz, num_workers=0)

    val_db_fuel = ObdData(root=data_root, mode="val", fuel=True, percentage=20, window_size=window_sz,\
                     path_length=train_path_length, label_dimension=1, pace=pace_test,
                     withoutElevation=False)
    val_db_time = ObdData(root=data_root, mode="val", fuel=False, percentage=20, window_size=window_sz, \
                          path_length=train_path_length, label_dimension=1, pace=pace_test,
                          withoutElevation=False)
    val_sampler_fuel = torch.utils.data.RandomSampler(val_db_fuel, replacement=True, num_samples=len(val_db_time),
                                             generator=None)

    val_loader_fuel = DataLoader(val_db_fuel, sampler = val_sampler_fuel, batch_size=batchsz, num_workers=0)
    val_loader_time = DataLoader(val_db_time, batch_size=batchsz, num_workers=0)

    viz = visdom.Visdom()
    # Create a new model or load an existing one.
    model = Pigat(feature_dim=feature_dimension,embedding_dim=[4,2,2,2,2,4,4],num_heads=head_number,output_dimension=lengthOfVelocityProfile, n2v_dim=32)
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

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, betas=(0.9, 0.98), eps=1e-9)
    schedule = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)

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
    prof = torch.profiler.profile(
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/resnet18'),
        record_shapes=True,
        with_stack=True)

    for epoch in range(epochs):
        model.train()
        prof.start()
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
            loss = omega_fuel*mse_fuel + omega_time*mse_time
            #print(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            prof.step()
        prof.stop()
            #print(dict(model.named_parameters()))

        viz.line([loss.item()], [global_step], win='train_mse', update='append')
        viz.line([mse_time.item()], [global_step], win='train_fuel_mse', update='append')
        viz.line([mse_fuel.item()], [global_step], win='train_time_mse', update='append')


        viz.line([loss_mape_fuel.item()], [global_step], win='train_fuel_mape', update='append')
        viz.line([loss_mape_time.item()], [global_step], win='train_time_mape', update='append')

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
                torch.save(model.state_dict(), ckpt_path)
            viz.line([val_mse.item()], [global_step], win='val_mse', update='append')
            viz.line([mse_fuel.item()], [global_step], win='val_fuel_mse', update='append')
            viz.line([mse_time.item()], [global_step], win='val_time_mse', update='append')
            viz.line([val_fuel_mape.item()], [global_step], win='val_fuel_mape', update='append')
            viz.line([val_time_mape.item()], [global_step], win='val_time_mape', update='append')

        global_step += 1
    print("best_epoch:", best_epoch, "best_mse:", np.array(best_mse.cpu()))



def test(model, test_path_length, test_pace, output = False):
    """

    :param output: "True" -> output the estimation results to output_root
    :return:
    """

    # load an existing model.
    test_db_time = ObdData(root=data_root, mode="test",fuel=False, percentage=20, window_size=window_sz,\
                      path_length=test_path_length, label_dimension=1, pace=test_pace,
                      withoutElevation=False)
    test_db_fuel = ObdData(root=data_root, mode="test",fuel=True, percentage=20, window_size=window_sz, \
                      path_length=test_path_length, label_dimension=1, pace=test_pace,
                      withoutElevation=False)

    test_loader_time = DataLoader(test_db_time, batch_size=batchsz, num_workers=0)
    test_loader_fuel = DataLoader(test_db_fuel, batch_size=batchsz, num_workers=0)

    #print(model)
    #p = sum(map(lambda p: p.numel(), model.parameters()))
    #print("number of parameters:", p)
    test_mse,_,_, test_fuel_mape, test_time_mape, mse_fuel, mse_time, mae_fuel, mae_time = eval(model, test_loader_time, test_loader_fuel, output=output)
    print("test_mape_fuel(%):", np.array(test_fuel_mape.cpu()) * 100)
    print("test_mape_time(%):", np.array(test_time_mape.cpu()) * 100)
    print("test_mse_fuel:", np.array(mse_fuel.cpu())**0.5/100)
    #print("test_mse_time:", np.array(mse_time.cpu()))
    print("test_mae_fuel:", np.array(mae_fuel.cpu())/100)
    #print("test_mae_time:", np.array(mae_time.cpu()))


def main(mode, output = False):
    """
    :param mode: "train" for training, "test" for testing the existing model or predicting
    :return:
    """
    if mode == "train":
        train()
    elif mode == "test":
        model = Pigat(feature_dim=feature_dimension, embedding_dim=[4, 2, 2, 2, 2, 4, 4], num_heads=head_number,
                             output_dimension=lengthOfVelocityProfile, n2v_dim=32)
        if os.path.exists(ckpt_path):
            print('Reloading model parameters..')
            model.load_state_dict(torch.load(ckpt_path, map_location=device))
        else:
            print('Error: no existing model')
        model.to(device)
        # length of path used for test
        test_path_length_list = [1,2,5,10,20,50,100,200]
        for length in test_path_length_list:
            pace_test = pace_train
            if  pace_test > length:
                pace_test = length
                print("pace test has been changed to:", pace_test)
            print("test path length:",length)
            test(model, length,pace_test, output = output)
    return

if __name__ == '__main__':
    main(input("mode="), output = False)
    #main("test")
    #main("train")
    # main("test", output = True)


