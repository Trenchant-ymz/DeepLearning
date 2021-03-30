import torch
from torch import nn
import os
import glob
import random
import csv

from torch.utils.data import Dataset, DataLoader


class ObdData(Dataset):

    def __init__(self, root, mode, percentage):
        """
        :param root:
        :param mode: str "train","val","test"
        """
        super(ObdData, self).__init__()
        self.percentage = str(percentage)
        self.root = os.path.join(root, self.percentage)
        self.mode = mode
        print(os.path.join(self.root, mode+"_data.csv"))
        #self.data, self.label = self.load_csv(os.path.join(self.percentage, mode+".csv"))

    def load_csv(self, filename):
        """
        :param filename:
        :return:
        """
        if not os.path.exists(os.path.join(self.root, filename)):
            pass




    def __len__(self):
        pass


def main():
    db = ObdData("data", "train", 20)


if __name__ == "__main__":
    main()
