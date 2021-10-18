import torch
from nets import AttentionBlk
from torch.utils.data import DataLoader
import numpy as np
import gc
from tqdm import tqdm
from torchinterp1d import Interp1d
import math

class EstimationModel:
    featureDim = 6
    embeddingDim = [4, 2, 2, 2, 2, 4, 4]
    numOfHeads = 1
    outputDimension = 1
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    device = torch.device("cpu")
    def __init__(self, outputOfModel):
        '''

        :param outputOfModel: "time" for time estimation, or "fuel" for fuel estimation
        '''
        self.outputOfModel = outputOfModel
        self.model = AttentionBlk(feature_dim=self.featureDim, embedding_dim=self.embeddingDim,
                                  num_heads=self.numOfHeads, output_dimension=self.outputDimension)
        self.modelAddress = "pretrained models/gat" + self.outputOfModel + "OctDropAddRelu.mdl"
        #self.modelAddress = "pretrained models/best_13d_" + self.outputOfModel + "SimulateData.mdl"

        self.model.load_state_dict(torch.load(self.modelAddress, map_location=self.device))
        self.model.to(self.device)

    def predictFromData(self, numericalInputData, categoricalInputData):
        numericalInputData = torch.Tensor(numericalInputData).unsqueeze(0)
        categoricalInputData = torch.LongTensor(categoricalInputData).transpose(0, 1).contiguous().unsqueeze(0)
        #print("numericalInputData", numericalInputData)
        #print("categoricalInputData", categoricalInputData)
        return self.model(numericalInputData.to(self.device), categoricalInputData.to(self.device)).item()

    def predictList(self, dloader):
        self.model.eval()
        energyDictionary = dict()
        for step, (idx, numericalInputData, categoricalInputData) in tqdm(enumerate(dloader)):
            #print(idx.shape, numericalInputData.shape, categoricalInputData.shape)
            with torch.no_grad():
                pred = self.model(numericalInputData, categoricalInputData)
                for i in range(len(idx)):
                    energyDictionary[idx[i].item()] = pred[i].item()
            del numericalInputData
            del categoricalInputData
            del idx
            gc.collect()
        return energyDictionary


class MultiTaskEstimationModel:
    tParts = 20
    lengthOfVelocityProfile = 20
    featureDim = 6
    embeddingDim = [4, 2, 2, 2, 2, 4, 4]
    numOfHeads = 1
    outputDimension = 1
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    device = torch.device("cpu")
    def __init__(self, outputOfModel):
        '''

        :param outputOfModel: "time" for time estimation, or "fuel" for fuel estimation
        '''
        self.outputOfModel = outputOfModel
        self.model = AttentionBlk(feature_dim=self.featureDim, embedding_dim=self.embeddingDim,
                                  num_heads=self.numOfHeads, output_dimension=self.lengthOfVelocityProfile)
        self.modelAddress = "multitaskModels/multiTaskVT.mdl"
        #self.modelAddress = "pretrained models/best_13d_" + self.outputOfModel + "SimulateData.mdl"

        self.model.load_state_dict(torch.load(self.modelAddress, map_location=self.device))
        self.model.to(self.device)

    def predictFromData(self, numericalInputData, categoricalInputData):
        numericalInputTensor = torch.Tensor(numericalInputData).unsqueeze(0)
        categoricalInputTensor = torch.LongTensor(categoricalInputData).transpose(0, 1).contiguous().unsqueeze(0)
        #print("numericalInputData", numericalInputData)
        #print("categoricalInputData", categoricalInputData)
        return self.predictFromTensor(numericalInputTensor, categoricalInputTensor).tolist()[0]

    def predictFromTensor(self, numericalInputTensor, categoricalInputTensor):

        #print("numericalInputData", numericalInputData)
        #print("categoricalInputData", categoricalInputData)
        # mean std
        meanOfSegmentLength = 608.2156661
        stdOfSegmentLength = 900.4150229

        meanOfMass = 23204.9788
        stdOfMass = 8224.139199693
        velocityProfile = self.model(numericalInputTensor, categoricalInputTensor) + 0.1

        length = self.denormalize(numericalInputTensor[:, 3 // 2, 4], meanOfSegmentLength, stdOfSegmentLength)
        m = self.denormalize(numericalInputTensor[:, 3 // 2, 1], meanOfMass, stdOfMass).unsqueeze(-1)
        v, t = self.vt2t(velocityProfile, length)
        pred_Time = self.timeEstimation(t)
        pred_Fuel = self.fuelEstimation(v, t, m)
        if self.outputOfModel == 'fuel':
            #print(pred_Fuel, pred_Fuel.shape)
            return pred_Fuel
        else:
            return pred_Time

    def predictList(self, dloader):
        self.model.eval()
        energyDictionary = dict()
        for step, (idx, numericalInputData, categoricalInputData) in tqdm(enumerate(dloader)):
            #print(idx.shape, numericalInputData.shape, categoricalInputData.shape)
            with torch.no_grad():
                pred = self.model(numericalInputData, categoricalInputData)
                for i in range(len(idx)):
                    energyDictionary[idx[i].item()] = pred[i].item()
            del numericalInputData
            del categoricalInputData
            del idx
            gc.collect()
        return energyDictionary

    # version 1 vd=>vt=>Same time interpolation
    def vd2vtWithInterpolation(self,velocityProfile, length):
        lengthOfEachPart = length / (velocityProfile.shape[1] - 1)
        averageV = velocityProfile[:, 0:-1] + velocityProfile[:, 1:]
        tOnSubPath = (2 * lengthOfEachPart).unsqueeze(-1) / averageV
        tAxis = torch.matmul(tOnSubPath, torch.triu(torch.ones(tOnSubPath.shape[1], tOnSubPath.shape[1])).to(self.device))
        tAxis = torch.cat([torch.zeros([tOnSubPath.shape[0], 1]).to(self.device), tAxis], dim=-1)
        timeSum = tAxis[:, -1]
        tNew = torch.arange(self.tParts).unsqueeze(0).to(self.device) * timeSum.unsqueeze(-1) / (self.tParts - 1)
        # print('tAxis',tAxis,'velocityProfile',velocityProfile, 'tNew',tNew)
        intetpResult = None
        intetpResult = Interp1d()(tAxis, velocityProfile, tNew, intetpResult).to(self.device)
        ##plot interpolation
        # x = tAxis.cpu().numpy()
        # y = velocityProfile.cpu().numpy()
        # xnew = tNew.cpu().numpy()
        # yq_cpu = intetpResult.cpu().numpy()
        # print(x.T, y.T, xnew.T, yq_cpu.T)
        # plt.plot(x.T, y.T, '-', xnew.T, yq_cpu.T, 'x')
        # plt.grid(True)
        # plt.show()
        return intetpResult, tNew

    # version 1 vd=>vt
    def vd2vt(self,velocityProfile, length):
        '''

        :param velocityProfile: velocityProfile: velocity profiel (uniform length sampling)
        :param length: length: total length of the segment
        :return: tAxis: the axis of time of the velocity profile
        '''
        lengthOfEachPart = length / (velocityProfile.shape[1] - 1)
        averageV = velocityProfile[:, 0:-1] + velocityProfile[:, 1:]
        tOnSubPath = (2 * lengthOfEachPart).unsqueeze(-1) / averageV
        tAxis = torch.matmul(tOnSubPath, torch.triu(torch.ones(tOnSubPath.shape[1], tOnSubPath.shape[1])).to(self.device))
        tAxis = torch.cat([torch.zeros([tOnSubPath.shape[0], 1]).to(self.device), tAxis], dim=-1)
        return velocityProfile, tAxis

    # version 1 vt=>calculate t
    def vt2t(self,velocityProfile, length):
        '''

        :param velocityProfile: velocity profiel (uniform time sampling)
        :param length: total length of the segment
        :return: tAxis: the axis of time of the velocity profile
        '''
        mul = torch.cat([torch.ones([1, 1]), 2 * torch.ones([velocityProfile.shape[1] - 2, 1]), \
                         torch.ones([1, 1])], dim=0).to(self.device)
        vAverage = torch.matmul(velocityProfile, mul) / 2
        tForOnePart = length.unsqueeze(-1) / vAverage
        tAxis = torch.arange(velocityProfile.shape[1]).unsqueeze(0).to(self.device) * tForOnePart
        return velocityProfile, tAxis

    def timeEstimation(self,tNew):
        return tNew[:, -1]

    def fuelEstimation(self, v, tNew, m):
        acc = self.vt2a(v, tNew)
        fuel = self.vt2fuel(v, acc, tNew, m)
        return fuel

    def vt2a(self, v, t):
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

    def power(self, v, a, m, theta, rho):
        R = 0.5003  # wheel radius (m)
        g = 9.81  # gravitational accel (m/s^2)
        A = 10.5  # frontal area (m^2)
        Cd = 0.5  # drag coefficient
        Crr = 0.0067  # rolling resistance
        Iw = 10  # wheel inertia (kg m^2)
        Nw = 10  # number of wheels

        Paccel = (m * a * v).clamp(0)
        Pascent = (m * g * torch.sin(torch.tensor([theta * (math.pi / 180)]).to(self.device)) * v).clamp(0)
        Pdrag = (0.5 * rho * Cd * A * v ** 3).clamp(0)
        Prr = (m * g * Crr * v).clamp(0)
        Pinert = (Iw * Nw * (a / R) * v).clamp(0)
        pauxA = torch.zeros(Pinert.shape).to(self.device)
        pauxB = torch.ones(Pinert.shape).to(self.device) * 1000
        Paux = torch.where(v > 0.1, pauxA, pauxB).to(self.device)
        P = (Paccel + Pascent + Pdrag + Prr + Pinert + Paux) / 1000
        return P

    def vt2fuel(self, v, a, t, m):

        # m = 10000  # mass (kg)
        rho = 1.225  # density of air (kg/m^3)
        theta = 0  # road grade (degrees)
        fc = 40.3  # fuel consumption (kWh/gal) # Diesel ~40.3 kWh/gal
        eff = 0.56  # efficiency of engine

        P = self.power(v, a, m, theta, rho)
        P_avg = (P[:, :-1] + P[:, 1:]) / 2
        f = P_avg / (fc * eff) * t[:, 1].unsqueeze(-1) / 3600
        # from galon => ml
        return torch.sum(f, dim=1) * 3.7854 * 100

    def denormalize(self, normalized, mean, std):
        return normalized * std + mean