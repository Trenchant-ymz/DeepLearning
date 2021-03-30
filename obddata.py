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
        :param root -> str: root file directory for data "model_data"
        :param mode -> str: str "train","val","test"
        :param percentage -> int or str: percentage of the data used for validation 20 or 10
        """
        super(ObdData, self).__init__()
        self.percentage = str(percentage)
        self.root = os.path.join(root, self.percentage)
        self.mode = mode
        # print(os.path.join(self.root, mode+"_data.csv"))
        self.data_list, self.label_list = self.load_csv(self.mode+"_data.csv")

    def load_csv(self, filename):
        """
        :param filename: -> str: filename to be read
        :return: data_list -> list(list(float)) -> list of data
                 label_list -> list(list(float)) -> list of labels
        """
        data_list, label_list = [], []
        print(os.path.join(self.root, filename))
        if not os.path.exists(os.path.join(self.root, filename)):
            print("Warning: Wrong File Directory")
        else:
            with open(os.path.join(self.root, filename)) as f:
                reader = csv.reader(f)
                for row in reader:
                    data, label = row
                    data_list.append(data)
                    label_list.append(label)
            assert len(data_list) == len(label_list)
        return data_list, label_list

    def __len__(self):
        """

        :return: the length of the db
        """
        return len(self.data_list)

    def __getitem__(self, idx):
        data = self.data_list[idx]
        data = list(map(float, data[1:-1].split(", ")))
        data = torch.tensor(data)
        label = self.label_list[idx]
        label = torch.tensor(list(map(float, label[1:-1].split(", "))))
        return data, label


def main():
    db = ObdData("model_data", "train", 20)
    x, y = next(iter(db))
    print("data:", x , "label:", y)
    loader = DataLoader(db, batch_size= 32, shuffle= False)
    for x,y in loader:
        print(x.shape,y.shape)
        break


if __name__ == "__main__":
    main()
