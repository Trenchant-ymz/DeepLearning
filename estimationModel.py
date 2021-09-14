import torch
from nets import AttentionBlk


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
        self.modelAddress = "pretrained models/best_13d_" + self.outputOfModel + "SimulateData.mdl"
        self.model.load_state_dict(torch.load(self.modelAddress, map_location=self.device))
        self.model.to(self.device)

    def predict(self, numericalInputData, categoricalInputData):
        numericalInputData = torch.Tensor(numericalInputData).unsqueeze(0)
        categoricalInputData = torch.LongTensor(categoricalInputData).transpose(0, 1).contiguous().unsqueeze(0)
        #print("numericalInputData", numericalInputData)
        #print("categoricalInputData", categoricalInputData)
        return self.model(numericalInputData.to(self.device), categoricalInputData.to(self.device)).item()