import torch
import os
import csv
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

class ObdData(Dataset):

    def __init__(self, root="data_normalized", mode="train", percentage=20, window_size=5, path_length=10\
                 , label_dimension=1, pace=5, withoutElevation = False):
        """

        :param root: root of the data(normalized)
        :param mode: "train" / "val" / "test"
        :param percentage: the percentage of data used for validation/test
            "20" -> 60/20/20 train/val/test; "10" -> 80/10/10 train/val/test
        :param window_size: length of window for attention
        :param path_length: length of path
        :param label_dimension: if == 2 -> label_list = [energy, time]; if == 1 -> label_list = [energy]
        """
        super(ObdData, self).__init__()
        self.percentage = str(percentage)
        self.root = os.path.join(root, self.percentage)
        self.mode = mode
        self.windowsz = window_size
        self.path_length = path_length
        self.label_dimension = label_dimension
        self.pace = pace
        self.len_of_index = 0
        self.withoutElevation = withoutElevation
        if self.withoutElevation == True:
            self.numerical_dimension = 5
        else:
            self.numerical_dimension = 6
        #self.data_list_w, self.label_list_w = self.load_csv(self.mode + "_data.csv")
        self.data = self.load_csv(self.mode + "_data.csv")
        #print(self.root)
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            # print('Using GPU..')
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        self.data_x = torch.Tensor([x[0] for x in self.data]).to(device)
        #print(self.data_x[0,...].shape)
        self.data_y = torch.Tensor([x[1] for x in self.data]).to(device)
        self.data_c = torch.LongTensor([x[5:12] for x in self.data]).to(device)



    def load_csv(self, filename):
        """
        :param filename: -> str: filename to be read
        :return: data_list -> list(list(float)) -> list of data in the window
                 label_list -> list(list(float)) -> list of labels in the window
        """
        # [index, 13 dimensions, path length, windows size, feature dimension]
        data_list_path= []
        # {trip-id: [13 dimensions, windowsz, feature]}
        data_dict_trip_id = dict()
        data_list = []

        if not os.path.exists(os.path.join(self.root, filename)):
            print("Warning: Wrong File Directory")
        else:
            with open(os.path.join(self.root, filename)) as f:
                reader = csv.reader(f)

                for row in reader:
                    data_row = row
                    # [data, label, segment_id, length, position, road_type, time_stage,
                    # week_day, lanes, bridge, endpoint_u, endpoint_v, trip_id]
                    if int(data_row[3]) >= self.path_length:

                        # length -> the number of segments in the trip
                        # 6 numerical attributes
                        if self.withoutElevation == True:
                            data_row[0] = list(map(float, data_row[0][1:-1].split(", ")))[:2]+list(map(float, data_row[0][1:-1].split(", ")))[3:]
                        else:
                            data_row[0] = list(map(float, data_row[0][1:-1].split(", ")))
                        # label
                        if self.label_dimension == 2:
                            data_row[1] = list(map(float, data_row[1][1:-1].split(", ")))
                        elif self.label_dimension == 1:
                            # [0] for fuel; [1] for time
                            data_row[1] = list(map(float, data_row[1][1:-1].split(", ")))[1]

                        data_row[2:] = map(int, map(float, data_row[2:]))
                        data_list.append(data_row)
            if not len(data_list):
                print("No qualified data")
                return None
            for i in range(len(data_list)):
                position = data_list[i][4]
                length = data_list[i][3]
                trip_id = data_list[i][-1]
                # construct a feature matrix for each window (windowsz * feature_dimension),
                # each row of the matrix is a feature(or label) of a segment in the window
                left = position - self.windowsz // 2  # [left, right) in the trip
                right = position + self.windowsz // 2 + 1
                left_idx = i - self.windowsz // 2  # [left_idx, right_idx) in the data_list
                right_idx = i + self.windowsz // 2 + 1
                if self.label_dimension ==1:
                    data_zero_line = [[0]*self.numerical_dimension]+[0]*12
                else:
                    data_zero_line = [[0] * self.numerical_dimension] + [[0,0]] + [0] * 11
                if left > 0 and right <= length:
                    #print(left,right,left_idx,right_idx)
                    data_sub = data_list[left_idx:right_idx]
                elif left <= 0 and right <= length:
                    #print(left, right, left_idx, right_idx)
                    data_sub = [data_zero_line]*(1 - left)+data_list[left_idx+1 - left:right_idx]
                elif left > 0 and right > length+1:
                    # print(left, right, left_idx, right_idx)
                    data_sub = data_list[left_idx:(i+(length-position)+1)] + [data_zero_line] * (right - length-1)
                else:
                    data_sub = [data_zero_line] * self.windowsz
                # [dimension, windowsz, feature]
                data_w = [[x] for x in data_sub[0]]
                for j in range(1, len(data_sub)):
                    data_j = [[x] for x in data_sub[j]]
                    data_w = [x + y for x, y in zip(data_w, data_j)]
                data_dict_trip_id[trip_id] = data_dict_trip_id.get(trip_id,[])+[data_w]
            # print(len(data_dict_trip_id))
            for i in sorted(data_dict_trip_id.keys()):
                data_trip = data_dict_trip_id[i]
                for j in range(0,len(data_trip)-self.path_length+1,self.pace):
                    data_trip_j = [[x] for x in data_trip[j]]
                    for k in range(1, self.path_length):
                        data_trip_k = [[x] for x in data_trip[j+k]]
                        data_trip_j = [x + y for x, y in zip(data_trip_j, data_trip_k)]
                    data_list_path.append(data_trip_j)
        return data_list_path

    def __len__(self):
        """

        :return: the length of the db
        """
        return len(self.data)

    def __getitem__(self, idx):

        return self.data_x[idx,...],self.data_y[idx,...],self.data_c[idx,...]



def testDataloader():
    # test dataloader
    db = ObdData("normalized data", "test", percentage=20, label_dimension= 1,withoutElevation=True)
    x,y,c = next(iter(db))
    print("data:", x, y, c)



    def denormalize(x_hat):
        print(x_hat.shape)
        fuel_mean = [0.205986075]
        fuel_std = [0.32661580545285]
        mean = torch.tensor(fuel_mean).unsqueeze(1)
        std = torch.tensor(fuel_std).unsqueeze(1)
        return x_hat * std + mean

    print(y)
    print(c[0,...])
    print(c[-1, ...])

    # print(x,y,d)
    loader = DataLoader(db, batch_size= 2, shuffle= False, num_workers=0)
    for x,y,c in loader:
        # x: numerical features [batch, path length, window size, feature dimension]
        # y: label [batch, path length, window size, (label dimension)]
        # c: categorical features [batch, number of categorical features, path length, window size]
        print(x.shape, y.shape,c.shape)
        t = torch.tensor([1, 0.01]).unsqueeze(0).to("cuda")
        print(t.shape)
        label = y[:, 0, 5 // 2]
        print(label.shape)
        print(label)
        print(label * t)
        # c[:,0,:,:] the first categorical features
        # "road_type", "time_stage", "week_day", "lanes", "bridge", "endpoint_u", "endpoint_v"
        print(c[:,:,0,:])
        print(c[:, :, 0, :].shape)
        # [batch, path length, window size, feature dimension]
        # data shape: torch.Size([32, 10, 5, 8]) label shape: torch.Size([32, 10, 5, 2])
        print("data shape:", x.shape, "label shape:", y.shape)
        # [batch, window size, feature dimension]
        x_one_path = x[:,0,:,:]
        # [window size, batch ,feature dimension]
        x_one_path = x_one_path.transpose(0,1).contiguous()
        print(x_one_path.shape)
        # print(x_one_path)
        break
    
    




if __name__ == "__main__":
    testDataloader()
