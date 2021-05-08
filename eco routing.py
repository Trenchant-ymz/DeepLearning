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


class Point:
    def __init__(self, x, y):
        Point.x = x
        Point.y = y


class OdPair:
    def __init__(self, origin, destination):
        OdPair.origin = origin
        OdPair.destination = destination


class Box:
    def __init__(self, lonMin, lonMax, latMin, latMax):
        Box.lonMin = lonMin
        Box.lonMax = lonMax
        Box.latMin = latMin
        Box.latMax = latMax

    def polygon(self):
        x1, x2, y1, y2 = Box.lonMin, Box.lonMax, Box.latMin, Box.latMax
        return Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])



class LocationRequest:
    def __init__(self):
        origin = Point(-93.2219, 44.979)
        destination = Point(-93.4696, 44.7854)
        LocationRequest.odPair = OdPair(origin, destination)
        LocationRequest.boundingBox = Box(-93.4696,-93.2219, 44.7854, 44.979)

class Graph:
    def __init__(self,network_gdf, nodes_gdf, G):
        Graph.network = network_gdf
        Graph.nodes = nodes_gdf
        Graph.g = G

def main():
    locationRequest = LocationRequest()
    segmentsInBox = extractSegmentsFromBbox(locationRequest.boundingBox)
    ecoRoute, energyOnEcoRoute = findEcoRouteAndCalEnergy(locationRequest.odPair, segmentsInBox)


def extractSegmentsFromBbox(boundingBox):
    pathOfFileFolderConsistNetwork = r'networkInBbox'
    if os.path.exists(pathOfFileFolderConsistNetwork):
        print("reloading network..")
        graphOfNetworkInBox = reloadingNetworkFrom(pathOfFileFolderConsistNetwork)
    else:
        print("downloading network..")
        graphOfNetworkInBox = downloadingNetworkTo(pathOfFileFolderConsistNetwork, boundingBox)



def reloadingNetworkFrom(pathOfFileFolderConsistNetwork):
    network_gdf = gpd.read_file(pathOfFileFolderConsistNetwork + '/edges.shp')
    nodes_gdf = gpd.read_file(pathOfFileFolderConsistNetwork + '/nodes.shp')
    g = ox.utils_graph.graph_from_gdfs(nodes_gdf, network_gdf)
    ox.plot_graph(g)
    return Graph(network_gdf, nodes_gdf, g)


def downloadingNetworkTo(pathOfFileFolderConsistNetwork, boundingBox):
    g = ox.graph_from_polygon(boundingBox.polygon(), network_type='drive')
    edges, nodes = save_graph_shapefile_directional(g, filepath=pathOfFileFolderConsistNetwork)
    return Graph(edges, nodes, g)


def save_graph_shapefile_directional(G, filepath=None, encoding="utf-8"):
    # default filepath if none was provided
    if filepath is None:
        filepath = os.path.join(ox.settings.data_folder, "graph_shapefile")

    # if save folder does not already exist, create it (shapefiles
    # get saved as set of files)
    if not filepath == "" and not os.path.exists(filepath):
        os.makedirs(filepath)
    filepath_nodes = os.path.join(filepath, "nodes.shp")
    filepath_edges = os.path.join(filepath, "edges.shp")

    # convert undirected graph to gdfs and stringify non-numeric columns
    gdf_nodes, gdf_edges = ox.utils_graph.graph_to_gdfs(G)
    gdf_nodes = ox.io._stringify_nonnumeric_cols(gdf_nodes)
    gdf_edges = ox.io._stringify_nonnumeric_cols(gdf_edges)
    # We need an unique ID for each edge
    gdf_edges["fid"] = gdf_edges.index
    # save the nodes and edges as separate ESRI shapefiles
    gdf_nodes.to_file(filepath_nodes, encoding=encoding)
    gdf_edges.to_file(filepath_edges, encoding=encoding)

    return gdf_edges, gdf_nodes

def findEcoRouteAndCalEnergy(odPair, segmentsInBox):
    pass


if __name__ == '__main__':
    main()


