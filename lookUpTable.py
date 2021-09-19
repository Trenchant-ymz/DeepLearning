import pickle
from edgeGdfPreprocessing import edgePreprocessing

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

    def generate(self, parameterForTableIni):
        table = dict()
        nodes, edges = parameterForTableIni.osmGraph.graphToGdfs()
        flg = 0
        for temp in parameterForTableIni.temperatureList:
            for m in parameterForTableIni.massList:
                for d in parameterForTableIni.dayList:
                    for t in parameterForTableIni.timeList:
                        tableRequest = tuple([temp, m, d, t])
                        table[tableRequest] = dict()
                        edges = edgePreprocessing(nodes, edges, temp, m, d, t)
                        edgesDict = edges.to_dict('index')
                        for i, w in enumerate(parameterForTableIni.windowList):
                            if flg == 0:
                                self.windowIdDict[w.getTup()] = i
                            numericalFeatures, categoricalFeatures = w.extractFeatures(edgesDict)
                            table[tableRequest][i] = parameterForTableIni.estimationModel.predict(numericalFeatures, categoricalFeatures)
                    flg = 1
        with open("windowIdDict.pkl", "wb") as tf:
            pickle.dump(self.windowIdDict, tf)
        return table

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


