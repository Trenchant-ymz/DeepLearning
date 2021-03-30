import torch
from torch import nn
import os
import glob
import random
import csv
import torch.nn.functional as F
import math, copy
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

class FFNet4(nn.Module):

    def __init__(self):
        super(FFNet4, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(8,16),
            nn.ReLU(),
            nn.Linear(16,32),
            nn.ReLU(),
            nn.Linear(32,64),
            nn.ReLU(),
            nn.Linear(64,2),
            nn.ReLU()
        )


    def forward(self, x):
        """

        :param x: [b 8]
        :return: [b 2] -> [fuel time]
        """
        return self.fc(x)


class AttentionNet(nn.Module):

    def __init__(self):
        super(AttentionNet, self).__init__()
        pass

    def forward(self, x):
        pass

def main():

    net = FFNet4()
    tmp = torch.randn(64, 8)
    out = net(tmp)
    print("fc out:", out.shape)

if __name__ == "__main__":
    main()