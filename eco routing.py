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
        self.x = x
        self.y = y


class OdPair:
    def __init__(self, origin, destination):
        self.origin = origin
        self.destination = destination


class Box:
    def __init__(self, lonMin, lonMax, latMin, latMax):
        self.lonMin = lonMin
        self.lonMax = lonMax
        self.latMin = latMin
        self.latMax = latMax

    def polygon(self):
        x1, x2, y1, y2 = self.lonMin, self.lonMax, self.latMin, self.latMax
        return Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])



class LocationRequest:
    def __init__(self):
        # Murphy Depot
        #origin = Point(-93.2219, 44.979)
        origin = Point(-93.2466, 44.8959)
        # Shakopee East (Depot):
        #destination = Point(-93.4620, 44.7903)
        #destination = Point(-93.4301, 44.8640)
        destination = Point(-93.4495, 44.8611)

        self.odPair = OdPair(origin, destination)
        self.boundingBox = Box(-93.4975, -93.1850, 44.7458, 45.0045)


class Graph:
    def __init__(self, network_gdf, nodes_gdf, G):
        self.network = network_gdf
        self.nodes = nodes_gdf
        self.g = G

def main():
    locationRequest = LocationRequest()
    osmGraphInBbox = extractGraphFromBbox(locationRequest.boundingBox)
    #shortestPath = findShorestPath(osmGraphInBbox, locationRequest.odPair)
    ecoRoute, energyOnEcoRoute = findEcoRouteAndCalEnergy(locationRequest.odPair, osmGraphInBbox)



def extractGraphFromBbox(boundingBox):
    folderOfGraph = r'GraphDataInBbox'
    if os.path.exists(folderOfGraph):
        print("reloading graph..")
        osmGraph = reloadGraphFrom(folderOfGraph)
    else:
        print("downloading graph..")
        osmGraph = downloadAndSaveGraphTo(folderOfGraph, boundingBox)
    #fig, ax = ox.plot_graph(osmGraph)
    return osmGraph


def reloadGraphFrom(folderOfGraph):
    return ox.load_graphml(os.path.join(folderOfGraph, 'osmGraph.graphml'))


def downloadAndSaveGraphTo(folderOfGraph, boundingBox):
    os.makedirs(folderOfGraph)
    osmGraph = ox.graph_from_polygon(boundingBox.polygon(), network_type='drive')
    ox.save_graphml(osmGraph, filepath=os.path.join(folderOfGraph, 'osmGraph.graphml'))
    return osmGraph


def findShorestPath(osmGraph, odPair):
    orig_yx = (odPair.origin.y, odPair.origin.x)
    target_yx = (odPair.destination.y, odPair.destination.x)
    nodes, edges = ox.graph_to_gdfs(osmGraph, nodes=True, edges=True)
    edges['uvPair'] = edges.apply(lambda x: (x.u, x.v), axis=1)
    segmentElevationChange = np.load('segmentElevationChange.npy', allow_pickle=True).item()
    edges['isInMurphy'] = edges.uvPair.apply(lambda x: x in segmentElevationChange)
    edgesInMurphy = edges[edges.isInMurphy]
    graphInMurphy = ox.utils_graph.graph_from_gdfs(nodes, edgesInMurphy)
    graphInMurphy.remove_nodes_from(list(nx.isolates(graphInMurphy)))
    orig_node = ox.get_nearest_node(graphInMurphy, orig_yx, method='euclidean')
    target_node = ox.get_nearest_node(graphInMurphy, target_yx, method='euclidean')
    shortestPath = nx.shortest_path(G=graphInMurphy, source=orig_node, target=target_node, weight='length')
    fig, ax = ox.plot_graph_route(graphInMurphy, shortestPath)
    return shortestPath


def findEcoRouteAndCalEnergy(odPair, osmGraphInBbox):
    graph = graphPreprocess(osmGraphInBbox)
    ecoRoute, energyConsumption = dijkstra(odPair, graph)

def graphPreprocess(osmGraphInBbox):
    pass


def dijkstra(odPair, graph):
    pass


if __name__ == '__main__':
    main()


