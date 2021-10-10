import torch
from nets import AttentionBlk
from torch.utils.data import DataLoader
import numpy as np
import gc
from tqdm import tqdm

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
        self.modelAddress = "pretrained models/best_13d_" + self.outputOfModel + "SimulateDatamlDrop180.mdl"
        #self.modelAddress = "pretrained models/best_13d_" + self.outputOfModel + "SimulateData.mdl"

        self.model.load_state_dict(torch.load(self.modelAddress, map_location=self.device))
        self.model.to(self.device)

    def predict(self, numericalInputData, categoricalInputData):
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


