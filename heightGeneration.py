import networkx as nx
import osmnx as ox
import os
from edgeGdfPreprocessing import edgePreprocessing
from estimationModel import EstimationModel
from window import Window, WindowFromList
from windowNode import NodeInPathGraph
import time
from collections import defaultdict
import routingAlgorithms
import plotly.graph_objects as go
import numpy as np
import plotly
import copy
import torch
import torch.nn.functional as F
import torch
import matplotlib.pyplot as plt
import time
import numpy as np
from torchinterp1d import Interp1d
import math




def vd2vt(velocityProfile, length):
    lengthOfEachPart = length / (velocityprofilelength - 1)
    print(lengthOfEachPart, lengthOfEachPart.shape)
    tAxis = torch.zeros([2, velocityprofilelength])
    print(tAxis, tAxis.shape)
    averageV = velocityProfile[:,0:-1] + velocityProfile[:,1:]
    print(averageV, averageV.shape)
    tOnSubPath = (2 * lengthOfEachPart).unsqueeze(-1) / averageV
    print(tOnSubPath, tOnSubPath.shape)
    tAxis = torch.matmul(tOnSubPath, torch.triu(torch.ones(tOnSubPath.shape[1],tOnSubPath.shape[1])))
    tAxis = torch.cat([torch.zeros([tOnSubPath.shape[0],1]),tAxis],dim=-1)
    print(tAxis, tAxis.shape)
    timeSum = tAxis[:, -1]
    tNew = torch.arange(tParts).unsqueeze(0) * timeSum.unsqueeze(-1) / (tParts - 1)
    intetpResult = None
    intetpResult = Interp1d()(tAxis, velocityProfile, tNew, intetpResult)
    ##plot interpolation
    x = tAxis.cpu().numpy()
    y = velocityProfile.cpu().numpy()
    xnew = tNew.cpu().numpy()
    yq_cpu = intetpResult.cpu().numpy()
    print(x.T, y.T, xnew.T, yq_cpu.T)
    plt.plot(x.T, y.T, '-', xnew.T, yq_cpu.T, 'x')
    plt.grid(True)
    plt.show()
    return tNew, intetpResult



def timeEstimation(tNew):
    return tNew[:, -1]


def fuelEstimation(tNew, v):
    acc = vt2a(v, tNew)
    fuel = vt2fuel(v, acc, tNew)
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


def power(v,a,m,theta,rho):
    R = 0.5003  # wheel radius (m)
    g = 9.81  # gravitational accel (m/s^2)
    A = 10.5  # frontal area (m^2)
    Cd = 0.5  # drag coefficient
    Crr = 0.0067  # rolling resistance
    Iw = 10  # wheel inertia (kg m^2)
    Nw = 10  # number of wheels

    Paccel = (m * a * v).clamp(0)
    Pascent =(m * g * torch.sin(torch.tensor([theta * (math.pi / 180)])) * v).clamp(0)
    Pdrag = (0.5 * rho * Cd * A * v ** 3).clamp(0)
    Prr = (m * g * Crr * v).clamp(0)
    Pinert = (Iw * Nw * (a / R) * v).clamp(0)
    pauxA = torch.zeros(Pinert.shape)
    pauxB = torch.ones(Pinert.shape)*1000
    Paux = torch.where(v>0.1, pauxA, pauxB)
    P = (Paccel + Pascent + Pdrag + Prr + Pinert + Paux) / 1000
    return P


def vt2fuel(v,a,t):

    m = 10000  # mass (kg)
    rho = 1.225  # density of air (kg/m^3)
    theta = 0  # road grade (degrees)
    fc = 40.3  # fuel consumption (kWh/gal) # Diesel ~40.3 kWh/gal
    eff = 0.56  # efficiency of engine

    P = power(v, a, m, theta, rho)
    P_avg = (P[:, :-1] + P[:, 1:]) / 2
    f = P_avg / (fc * eff) * t[:, 1].unsqueeze(-1) / 3600
    return torch.sum(f, dim=1)


batchsize = 2
velocityprofilelength = 4
tParts = 6 # divide time into several parts
# [batcsize]
length = torch.tensor([12,36])
print(length,length.shape)
# [batchsize, velocityprofilelength]
velocityProfile = [[3.0,4.0,6.0,12.0],[4.0,6.0,12.0,18.0]]
velocityProfile = torch.tensor(velocityProfile)
print(velocityProfile,velocityProfile.shape)
tAxis = torch.cat([torch.ones([1, 1]),2*torch.ones([velocityProfile.shape[1]-2,1]),\
                   torch.ones([1, 1])],dim=0)
vAverage = torch.matmul(velocityProfile,tAxis)/2
print(vAverage,vAverage.shape)
t = length.unsqueeze(-1)/vAverage
print(t,t.shape)
tNew = torch.arange(velocityProfile.shape[1]).unsqueeze(0) * t
print(tNew,tNew.shape)






