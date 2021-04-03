import torch
from torch import nn
import os
import glob
import random
import csv

from torch.utils.data import Dataset, DataLoader


class ObdData(Dataset):

    def __init__(self, root="model_data", mode="train", percentage=20, window_size=5, path_length=10):
        """
        :param root -> str: root file directory for data "model_data"
        :param mode -> str: str "train","val","test"
        :param percentage -> int or str: percentage of the data used for validation 20 or 10
        """
        super(ObdData, self).__init__()
        self.percentage = str(percentage)
        self.root = os.path.join(root, self.percentage)
        self.mode = mode
        self.windowsz = window_size
        self.path_length = path_length
        # print(os.path.join(self.root, mode+"_data.csv"))
        self.data_list_w, self.label_list_w = self.load_csv(self.mode + "_data.csv")
        print(self.__len__())
        #print(self.__getitem__(-1))
        # print(self.data_list[-1])
        # print(self.label_list[-1])

        # tensor_data = torch.tensor(self.data_list)
        # tensor_label = torch.tensor(self.label_list)
        # print(tensor_data.shape)
        # print(tensor_label.shape)

    def load_csv(self, filename):
        """
        :param filename: -> str: filename to be read
        :return: data_list -> list(list(float)) -> list of data in the window
                 label_list -> list(list(float)) -> list of labels in the window
        """
        data_list_with_window, label_list_with_window = [], []
        data_list, label_list, id_list, position_list = [], [], [], []
        # print(os.path.join(self.root, filename))

        if not os.path.exists(os.path.join(self.root, filename)):
            print("Warning: Wrong File Directory")
        else:
            with open(os.path.join(self.root, filename)) as f:
                reader = csv.reader(f)

                for row in reader:
                    data, label, segment_id, length, position = row
                    # length -> the number of segments in the trip
                    data = list(map(float, data[1:-1].split(", ")))
                    label = list(map(float, label[1:-1].split(", ")))

                    data_list.append(data)
                    label_list.append(label)
                    id_list.append(int(int(float(segment_id))))
                    length = int(float(length))
                    # print(int(float(position)))
                    position_list.append(int(float(position)))

                    # for an entire trip
                    if int(float(position)) == length:

                        # construct a feature matrix for each window (windowsz * feature_dimension),
                        # each row of the matrix is a feature(or label) of a segment in the window
                        for i in range(len(position_list)):
                            left = i - self.windowsz // 2  # [left, right) bound of the features in the window
                            right = i + self.windowsz // 2 + 1
                            # print(left,right)

                            if left >= 0 and right <= length:
                                data_list_with_window.append(data_list[left:right])
                                label_list_with_window.append(label_list[left:right])
                            elif left < 0 and right <= length:
                                data_list_with_window.append(
                                    [[0 for _ in range(len(data))] for _ in range(0 - left)] + data_list[0:right])
                                label_list_with_window.append(
                                    [[0 for _ in range(len(label))] for _ in range(0 - left)] + label_list[0:right])
                            elif left >= 0 and right > length:
                                data_list_with_window.append(
                                    data_list[left: length] + [[0 for _ in range(len(data))] for _ in
                                                               range(right - length)])
                                label_list_with_window.append(
                                    label_list[left: length] + [[0 for _ in range(len(label))] for _ in
                                                                range(right - length)])
                            else:
                                data_list_with_window.append(
                                    [[0 for _ in range(len(data))] for _ in range(0 - left)] + data_list[0: length] \
                                    + [[0 for _ in range(len(data))] for _ in range(right - length)])
                                label_list_with_window.append(
                                    [[0 for _ in range(len(label))] for _ in range(0 - left)] + \
                                    label_list[0: length] + [[0 for _ in range(len(label))] for _ in
                                                             range(right - length)])

                        data_list, label_list, id_list, position_list = [], [], [], []

            assert len(data_list_with_window) == len(label_list_with_window)
        return data_list_with_window, label_list_with_window

    def __len__(self):
        """

        :return: the length of the db
        """
        return len(self.data_list_w)//self.path_length

    def __getitem__(self, idx):
        data = torch.tensor(self.data_list_w[idx*self.path_length:(idx+1)*self.path_length])
        label = torch.tensor(self.label_list_w[idx*self.path_length:(idx+1)*self.path_length])
        return data, label


def main():
    db = ObdData("model_data", "train", percentage=20)

    x, y = next(iter(db))
    # print("data:", x, "label:", y)
    loader = DataLoader(db, batch_size= 2, shuffle= False, num_workers=4)
    for x,y in loader:
        #print(x,y)

        # [batch, path length, window size, feature dimension]
        # data shape: torch.Size([32, 10, 5, 8]) label shape: torch.Size([32, 10, 5, 2])
        print("data shape:", x.shape, "label shape:", y.shape)
        # [batch, window size, feature dimension]
        x_one_path = x[:,9,:,:]
        # [window size, batch ,feature dimension]
        x_one_path = x_one_path.transpose(0,1).contiguous()
        print(x_one_path.shape)
        print(x_one_path)
        break



if __name__ == "__main__":
    main()
