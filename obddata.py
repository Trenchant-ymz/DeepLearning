import torch
from torch import nn
import os
import glob
import random
import csv

from torch.utils.data import Dataset, DataLoader


class ObdData(Dataset):

    def __init__(self, root, mode):
        """
        :param root:
        :param mode: str "train","val","test"
        """
        super(ObdData, self).__init__()
        self.root = root
        self.mode = mode

    def load_csv(self, filename):
        """
        :param filename:
        :return:
        """


    def __len__(self):
        pass


def main():
    db = ObdData("20", "train")
    print("test git")


if __name__ == "__main__":
    main()
