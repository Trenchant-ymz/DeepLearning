# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import torch
from torch import nn
import os, glob
import random, csv

from torch.utils.data import Dataset, DataLoader
from obddata import ObdData

if __name__ == '__main__':
    print(torch.__version__)
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
