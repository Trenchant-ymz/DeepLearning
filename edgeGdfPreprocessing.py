import torch
from torch import nn, optim
import os
from torch.utils.data import DataLoader
from obddata import ObdData
from nets import AttentionBlk
import torch.nn.functional as F
import numpy as np
import csv
import time
import visdom
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import osmnx as ox
import networkx as nx
import plotly.graph_objects as go
import psycopg2
import datetime
import plotly.io as pio
import osmnx as ox
from shapely.geometry import Polygon
import os
import gc
from os import walk
import geopandas as gpd
import plotly
from random import shuffle
import math
import copy

def edgePreprocessing(edgesGdf, nodesGdf):
    # mass
    edgesGdf['mass'] = 23000
    edgesGdf['mass'] = (edgesGdf.mass - 23185.02515) / 8227.65140266416
    # speed limit
    edgesGdf['speedLimit'] = edgesGdf.apply(lambda x: calSpeedlimit(x), axis=1)
    edgesGdf['speedLimit'] = (edgesGdf.speedLimit - 80.5318397987996) / 21.7071763681126

    # elevation change
    segmentElevationChange = np.load('segmentElevationChange.npy', allow_pickle=True).item()
    edgesGdf['elevationChange'] = edgesGdf.apply(lambda x: segmentElevationChange[(x.u, x.v)], axis=1)
    edgesGdf['elevationChange'] = (edgesGdf.elevationChange + 0.00450470150885644) / 8.62149031019689

    # previous orientation
    edgesGdf['points'] = edgesGdf.geometry.apply(lambda x: pointsInSegment(x))

    #edgesGdf['previousOrientation'] = (edgesGdf.previousOrientation + 1.46016587027665) / 33.3524612794841

    # length no changes
    edgesGdf['length'] = (edgesGdf.length + 611.410287539911) / 903.292309592642

    # direction angle
    edgesGdf['directionAngle'] = edgesGdf.points.apply(lambda x: directionAngle(x))
    edgesGdf['directionAngle'] = (edgesGdf.directionAngle - 1.67006008669261) / 102.77763894989

    # road type
    edgesGdf['roadtype'] = edgesGdf.apply(lambda x: highway_cal(x), axis=1)
    roadtypeDict = np.load('road_type_dictionary.npy', allow_pickle=True).item()
    edgesGdf['roadtype'] = edgesGdf.roadtype.apply(lambda x: roadtypeDict[x] if x in roadtypeDict else 0)

    # time
    edgesGdf['timeOfTheDay'] = 9
    edgesGdf['timeOfTheDay'] = edgesGdf.timeOfTheDay.apply(lambda x: calTimeStage(x))
    edgesGdf['dayOfTheWeek'] = 3

    # lanes
    edgesGdf['lanes'] = edgesGdf.apply(lambda x: cal_lanes(x), axis=1)
    edgesGdf['lanes'] = edgesGdf['lanes'].apply(lambda x: x if x <= 8 else 8)

    #bridge
    edgesGdf['bridgeOrNot'] = edgesGdf.bridge.apply(lambda x: bridgeOrNot(x))


    # endpoints
    edgesGdf['oriSignal'] = edgesGdf.u.apply(lambda x: nodesGdf.loc[x, 'highway']).fillna("None")
    edgesGdf['destSignal'] = edgesGdf.v.apply(lambda x: nodesGdf.loc[x, 'highway']).fillna("None")
    endpoints_dictionary = np.load('endpoints_dictionary.npy', allow_pickle=True).item()
    edgesGdf['oriSignalCategoried'] = edgesGdf.oriSignal.apply(lambda x: endpoints_dictionary[x] if x in endpoints_dictionary else 0)
    edgesGdf['destSignalCategoried'] = edgesGdf.destSignal.apply(lambda x: endpoints_dictionary[x] if x in endpoints_dictionary else 0)

    edgesGdf['categoricalFeature'] = edgesGdf.apply(lambda x: categoricalFeature(x), axis=1)

    return edgesGdf


def categoricalFeature(arraylike):
    return [arraylike.roadtype, arraylike.timeOfTheDay, arraylike.dayOfTheWeek, arraylike.lanes, arraylike.bridgeOrNot, arraylike.oriSignalCategoried, arraylike.destSignalCategoried]


def bridgeOrNot(bridge):
    if isinstance(bridge,float):
        return 0
    else:
        return 1


def directionAngle(pointsList):
    longitude_o, latitude_o = pointsList[0]
    longitude_d, latitude_d = pointsList[-1]
    direction = [latitude_d - latitude_o, longitude_d - longitude_o]
    direction_array = np.array(direction)
    cosangle = direction_array.dot(np.array([1, 0])) / (np.linalg.norm(direction_array))
    if np.cross(direction_array, np.array([1, 0])) < 0:
        direction_angle = math.acos(cosangle) * 180 / np.pi
    else:
        direction_angle = -math.acos(cosangle) * 180 / np.pi
    return direction_angle


def pointsInSegment(geometry):
    pointsStringList = list(str(geometry)[12: -1].split(", "))
    for i,val in enumerate(pointsStringList):
        pointsStringList[i] = tuple(map(float, val.split(" ")))
    return pointsStringList


def highway_cal(network_seg):
    if 'highway' in network_seg and network_seg['highway']:
        if isinstance(network_seg['highway'], str):
            return network_seg['highway']
        elif isinstance(network_seg['highway'], list):
            return network_seg['highway'][0]
    else:
        return 'unclassified'

def calSpeedlimit(array_like):
    if isinstance(array_like['maxspeed'],float):
        if math.isnan(array_like['maxspeed']):
            if array_like['highway'] == "motorway":
                return 55 * 1.609
            elif array_like['highway'] == "motorway_link":
                return 50 * 1.609
            return 30 * 1.609
        else:
            return array_like['maxspeed'] * 1.609
    else:
        res = ''
        flag = 0
        for i in list(array_like['maxspeed']):
            if i.isdigit():
                flag = 1
                res += i
            else:
                if flag == 1:
                    speed = int(res) * 1.609
                    return speed

def calTimeStage(timeOfTheDay):
    return timeOfTheDay // 4 + 1

def cal_lanes(array_like):
    if isinstance(array_like['lanes'],list):
        for i in array_like['lanes']:
            if i.isdecimal():
                return int(i)
    if isinstance(array_like['lanes'],int):
        return array_like['lanes']
    if pd.isna(array_like['lanes']):
        return 0
    if array_like['lanes'].isalpha():
        return 0
    if array_like['lanes'].isalnum():
        return int(array_like['lanes']) if int(array_like['lanes']) > 0 else 0
    else:
        return 0