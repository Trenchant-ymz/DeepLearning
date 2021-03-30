# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import torch
from torch import nn
import os, glob
import random, csv
from torch.utils.data import Dataset, DataLoader
from obddata import ObdData
from obddata import ObdData
from nets import FFNet4


def main():
    # device = torch.device("cuda")
    model = FFNet4()
    print(model)
    p = sum(map(lambda p: p.numel(), model.parameters()))
    print('parameters size:', p)
    for epoch in range(1000):
        break


if __name__ == '__main__':
    main()
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
