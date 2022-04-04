import torch
import os
import csv
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import networkx as nx
import osmnx as ox
import geopandas as gpd
import torch
import pickle
from torch.utils.data.dataset import ConcatDataset
import math
import torch
from torch.utils.data.sampler import RandomSampler


def readGraph():
    '''
    :param
        'dual': segment -> node; intersection -> edge
    :return:
    '''
    percentile = '005'
    filefold = r'D:/cygwin64/home/26075/workspace/'
    network_gdf = gpd.read_file(filefold + 'network_' + percentile + '/edges.shp')
    nodes_gdf = gpd.read_file(filefold + 'network_' + percentile + '/nodes.shp')
    graph = ox.utils_graph.graph_from_gdfs(nodes_gdf, network_gdf)
    # convert the graph to the dual graph
    lineGraph = nx.line_graph(graph)
    # transfer the multiDilineGraph to 2 dimension
    eNew = [(x[0], x[1]) for x in lineGraph.edges]

    graph = nx.Graph()
    graph.update(edges=eNew, nodes=lineGraph.nodes)
    return graph


def loadData(root="data_normalized", mode = "train", fuel=False, percentage=20, window_size=5, path_length=20, label_dimension=1, pace=5, withoutElevation=False):
    '''

    :param root: root of the data(normalized)
    :param mode: "train" / "val" / "test"
    :param percentage: the percentage of data used for validation/test
        "20" -> 60/20/20 train/val/test; "10" -> 80/10/10 train/val/test
    :param window_size: length of window for attention
    :param path_length: length of path
    :param label_dimension: if == 2 -> label_list = [energy, time]; if == 1 -> label_list = [energy]
    :return:
    '''
    percentage = str(percentage)
    root = os.path.join(root, percentage)
    len_of_index = 0
    dualGraphNode = readDualGraphNode("dualGraphNodes.pkl")

    numerical_dimension = 5 if withoutElevation else 6

    inputFileName =  mode+"_data_fuel.csv" if fuel else mode+"_data.csv"
    data_x, data_y, data_c, id = processAndSave(root, inputFileName, fuel, window_size, path_length, label_dimension, numerical_dimension, pace, withoutElevation)
    return data_x, data_y, data_c, id


def readDualGraphNode(filename):
    open_file = open(filename, "rb")
    dualGraphNode = pickle.load(open_file)
    open_file.close()
    return dualGraphNode

def processAndSave(root, filename, fuel, windowsz, path_length, label_dimension, numerical_dimension, pace, withoutElevation):
    """
    :param filename: -> str: filename to be read
    :return: data_list -> list(list(float)) -> list of data in the window
             label_list -> list(list(float)) -> list of labels in the window
    """
    # [index, 13 dimensions, path length, windows size, feature dimension]
    data_list_path = []
    # {trip-id: [13 dimensions, windowsz, feature]}
    data_dict_trip_id = dict()
    data_list = []
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    #print(os.path.join(root, filename))
    with open(os.path.join(root, filename)) as f:
        reader = csv.reader(f)

        for row in reader:
            data_row = row
            # [data, label, segment_id, length, position, road_type, time_stage,
            # week_day, lanes, bridge, endpoint_u, endpoint_v, trip_id]
            if int(data_row[3]) >= path_length:

                # length -> the number of segments in the trip
                # 6 numerical attributes
                # speed_limit,mass,elevation_change,previous_orientation,length,direction_angle
                if withoutElevation:
                    data_row[0] = list(map(float, data_row[0][1:-1].split(", ")))[:2] + list(
                        map(float, data_row[0][1:-1].split(", ")))[3:]
                else:
                    data_row[0] = list(map(float, data_row[0][1:-1].split(", ")))
                    # data_row[0][1] = 0
                # label
                if label_dimension == 2:
                    data_row[1] = list(map(float, data_row[1][1:-1].split(", ")))
                elif label_dimension == 1:
                    # [0] for fuel *100; [1] for time
                    if fuel:
                        data_row[1] = list(map(float, data_row[1][1:-1].split(", ")))[0] * 100
                    else:
                        data_row[1] = list(map(float, data_row[1][1:-1].split(", ")))[1]

                data_row[2:] = map(int, map(float, data_row[2:]))
                # data_row[-1] = map(int, data_row[-1].split(", "))
                # data_row[-1] =self.dualGraphNode.index((data_row[-1][0], data_row[-1][1], 0))
                # print('data_row',data_row)
                data_list.append(data_row)
    if not len(data_list):
        print("No qualified data")
        return None
    for i in range(len(data_list)):
        position = data_list[i][4]
        length = data_list[i][3]
        trip_id = data_list[i][-2]
        # construct a feature matrix for each window (windowsz * feature_dimension),
        # each row of the matrix is a feature(or label) of a segment in the window
        left = position - windowsz // 2  # [left, right) in the trip
        right = position + windowsz // 2 + 1
        left_idx = i - windowsz // 2  # [left_idx, right_idx) in the data_list
        right_idx = i + windowsz // 2 + 1
        if label_dimension == 1:
            data_zero_line = [[0] * numerical_dimension] + [0] * 13
        else:
            data_zero_line = [[0] * numerical_dimension] + [[0, 0]] + [0] * 12
        if left > 0 and right <= length:
            # print(left,right,left_idx,right_idx)
            data_sub = data_list[left_idx:right_idx]
        elif left <= 0 and right <= length:
            # print(left, right, left_idx, right_idx)
            data_sub = [data_zero_line] * (1 - left) + data_list[left_idx + 1 - left:right_idx]
        elif left > 0 and right > length + 1:
            # print(left, right, left_idx, right_idx)
            data_sub = data_list[left_idx:(i + (length - position) + 1)] + [data_zero_line] * (
                        right - length - 1)
        else:
            data_sub = [data_zero_line] * windowsz
        # [dimension, windowsz, feature]
        data_w = [[x] for x in data_sub[0]]
        for j in range(1, len(data_sub)):
            data_j = [[x] for x in data_sub[j]]
            data_w = [x + y for x, y in zip(data_w, data_j)]
        data_dict_trip_id[trip_id] = data_dict_trip_id.get(trip_id, []) + [data_w]
    # print(len(data_dict_trip_id))
    for i in sorted(data_dict_trip_id.keys()):
        data_trip = data_dict_trip_id[i]
        for j in range(0, len(data_trip) - path_length + 1, pace):
            data_trip_j = [[x] for x in data_trip[j]]
            for k in range(1, path_length):
                data_trip_k = [[x] for x in data_trip[j + k]]
                data_trip_j = [x + y for x, y in zip(data_trip_j, data_trip_k)]
            data_list_path.append(data_trip_j)
    data_x = torch.Tensor([x[0] for x in data_list_path]).to(device)
    # print(self.data_x[0,...].shape)
    data_y = torch.Tensor([x[1] for x in data_list_path]).to(device)
    data_c = torch.LongTensor([x[5:12] for x in data_list_path]).to(device)
    id = torch.LongTensor([x[-1] for x in data_list_path]).to(device)
    return data_x, data_y, data_c, id



if __name__ == "__main__":
    x_train, y_train, c_train, id_train = loadData(root="model_data_newNov1perc")
    print(x_train.shape, y_train.shape, c_train.shape, id_train.shape)
    print(x_train.shape[0])

