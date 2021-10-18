import gc
import pickle
from edgeGdfPreprocessing import edgePreprocessing
#from multiprocessing import Pool
from pathos.multiprocessing import ProcessingPool as Pool
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

class LookUpTable:
    def __init__(self, locationRequest, filename, generateNewTable=False, parameterForTableIni = None):
        self.temperature = locationRequest.temperature
        self.mass = locationRequest.mass
        self.day = locationRequest.dayOfTheWeek
        self.time = locationRequest.timeOfTheDay
        self.request = self.__getRequest()
        self.windowIdDict = dict()
        if generateNewTable:
            holeTable = self.generate(parameterForTableIni)
            self.saveTable(filename, holeTable)
            self.lookUpTable = holeTable[self.request]
        else:
            self.lookUpTable = self.readTable(filename)

    def __len__(self):
        return len(self.lookUpTable)

    def generate(self,parameterForTableIni):
        # MultiNum = 2
        # print("MultiProcess:" + MultiNum.__str__())
        table = dict()
        nodes, edges = parameterForTableIni.osmGraph.graphToGdfs()
        flg = 0
        for temp in parameterForTableIni.temperatureList:
            for m in parameterForTableIni.massList:
                for d in parameterForTableIni.dayList:
                    for t in parameterForTableIni.timeList:
                        tableRequest = tuple([temp, m, d, t])
                        table[tableRequest] = dict()
                        self.edgesDict = edgePreprocessing(nodes, edges, temp, m, d, t).to_dict('index')
                        windowFeatureList = []

                        # for i, w in tqdm(enumerate(parameterForTableIni.windowList)):
                        #     if flg == 0:
                        #         self.windowIdDict[w.getTup()] = i
                        #     numericalFeatures, categoricalFeatures = w.extractFeatures(self.edgesDict)
                        #     table[tableRequest][i] = parameterForTableIni.estimationModel.predict(numericalFeatures,categoricalFeatures)

                        windowFeatureList = []
                        for i, w in enumerate(parameterForTableIni.windowList):
                            if flg == 0:
                                self.windowIdDict[w.getTup()] = i
                            numericalFeatures, categoricalFeatures = w.extractFeatures(self.edgesDict)
                            windowFeatureList.append([numericalFeatures, categoricalFeatures])
                        print("preprocess finished")
                        if torch.cuda.is_available():
                            device = torch.device("cuda")
                        else:
                            device = torch.device("cpu")
                        device = torch.device("cpu")
                        numericalFeatures = torch.Tensor([x[0] for x in windowFeatureList]).to(device)
                        categoricalFeatures = torch.LongTensor([x[1] for x in windowFeatureList]).transpose(1,2).contiguous().to(device)
                        print(numericalFeatures.shape, categoricalFeatures.shape)
                        #db = WindowFeatureDataLoader(windowFeatureList)
                        del windowFeatureList
                        gc.collect()
                        #dloader = DataLoader(db, batch_size=256, num_workers=0)
                        energyOfWindows = parameterForTableIni.estimationModel.model(numericalFeatures, categoricalFeatures).squeeze(1).tolist()
                        # mutitask version
                        #energyOfWindows = parameterForTableIni.estimationModel.predictFromTensor(numericalFeatures,categoricalFeatures).tolist()
                        #print(energyOfWindows.shape)
                        for i in range(len(energyOfWindows)):
                            table[tableRequest][i] = energyOfWindows[i]

                        #windowList = self.parameterForTableIni.windowList
                        # multiprocess
                        # pool = Pool(MultiNum)
                        # energyOfWindows = pool.map(self.__calculateOneWindow, windowList)
                        # pool.close()
                        # pool.join()

                    flg = 1
        with open("windowIdDict.pkl", "wb") as tf:
            pickle.dump(self.windowIdDict, tf)
        return table

    def __calculateOneWindow(self, window, estimationModel):
        numericalFeatures, categoricalFeatures = window.extractFeatures(self.edgesDict)
        return estimationModel.predict(numericalFeatures, categoricalFeatures)

    def saveTable(self, filename, holeTable):
        with open(filename+".pkl", "wb") as tf:
            pickle.dump(holeTable, tf)
        return

    def readTable(self, filename):
        with open(filename+".pkl", "rb") as tf:
            loadDict = pickle.load(tf)
        with open("windowIdDict.pkl", "rb") as tf:
            self.windowIdDict = pickle.load(tf)
        return loadDict[self.request]

    def extractValue(self, window):
        return self.lookUpTable[self.windowIdDict[window.getTup()]]

    def __getRequest(self):
        return tuple([self.temperature, self.mass, self.day, self.time])


class WindowFeatureDataLoader:
    def __init__(self, windowFeatureList):

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.numericalFeatures = torch.Tensor([x[0] for x in windowFeatureList]).to(self.device)
        self.categoricalFeatures = torch.LongTensor([x[1] for x in windowFeatureList]).transpose(1,2).contiguous().to(self.device)
        #self.device = torch.device("cpu")
        print(self.__len__())
        print(self.numericalFeatures.shape, self.categoricalFeatures.shape)
        print(self.__getitem__(0))
        #print(self.__getitem__(1))

    def __len__(self):
        return self.numericalFeatures.shape[0]

    def __getitem__(self, idx):
        return torch.Tensor([idx]).to(self.device), self.numericalFeatures[idx,...], self.categoricalFeatures[idx,...]