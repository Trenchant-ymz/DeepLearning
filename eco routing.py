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
from edgeGdfPreprocessing import edgePreprocessing

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def xy(self):
        return (self.x, self.y)
    def yx(self):
        return (self.y, self.x)


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
        origin = Point(-93.2219, 44.979)
        #origin = Point(-93.2466, 44.8959)
        # Shakopee East (Depot):
        destination = Point(-93.4620, 44.7903)
        #destination = Point(-93.4495, 44.8611)

        self.odPair = OdPair(origin, destination)
        self.boundingBox = Box(-93.4975, -93.1850, 44.7458, 45.0045)


class EstimationModel:
    featureDim = 6
    embeddingDim = [4, 2, 2, 2, 2, 4, 4]
    numOfHeads = 1
    outputDimension = 1
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    def __init__(self, outputOfModel):
        '''

        :param outputOfModel: "time" for time estimation, or "fuel" for fuel estimation
        '''
        self.outputOfModel = outputOfModel
        self.model = AttentionBlk(feature_dim=self.featureDim, embedding_dim=self.embeddingDim,
                                  num_heads=self.numOfHeads, output_dimension=self.outputDimension)
        self.modelAddress = "best_13d_" + self.outputOfModel + ".mdl"
        self.model.load_state_dict(torch.load(self.modelAddress, map_location=self.device))
        self.model.to(self.device)

    def predict(self, numericalInputData, categoricalInputData):
        numericalInputData = torch.Tensor(numericalInputData).unsqueeze(0)
        categoricalInputData = torch.LongTensor(categoricalInputData).transpose(0, 1).contiguous().unsqueeze(0)
        return self.model(numericalInputData.to(self.device), categoricalInputData.to(self.device)).item()


class Window:
    def __init__(self, prevSeg, midSeg, sucSeg):
        self.prevSeg = prevSeg
        self.midSeg = midSeg
        self.sucSeg = sucSeg

    def extractFeatures(self, edgesGdf, prevEdge):
        prevSegNumFeature, prevSegCatFeature = edgeFeature(self.prevSeg, edgesGdf, prevEdge)
        midSegNumFeature, midSegCatFeature = edgeFeature(self.midSeg, edgesGdf, prevEdge)
        sucSegNumFeature, sucSegCatFeature = edgeFeature(self.sucSeg, edgesGdf, prevEdge)
        numericalFeatures = [prevSegNumFeature, midSegNumFeature, sucSegNumFeature]
        categoricalFeatures = [prevSegCatFeature, midSegCatFeature, sucSegCatFeature]
        return numericalFeatures, categoricalFeatures


def edgeFeature(segmentIDInGdf, edgesGdf, prevEdge):
    if segmentIDInGdf == -1:
        return [0]*6, [0]*7
    edge = edgesGdf.loc[segmentIDInGdf]
    catFeature = edge.categoricalFeature
    previousOrientation = calPrevOrientation(edge, prevEdge)
    numFeature = [edge.speedLimit, edge.mass, edge.elevationChange, previousOrientation, edge.length, edge.directionAngle]
    return numFeature, catFeature


def calPrevOrientation(edge, prevEdge):
    if prevEdge is None:
        orientation = 0
    else:
        a = prevEdge.points[-2]
        b = edge.points[0]
        c = edge.points[1]
        orientation = ori_cal(a,b,c)
    orientation = (orientation + 1.46016587027665) / 33.3524612794841
    return orientation


def ori_cal(coor_a, coor_b, coor_c):
    '''Calcuate the orientation change from vector ab to bc (0 in default, right turn > 0, left turn < 0)

    10 degree is the threshold of a turn.

    Args:
        coor_a: coordinate of point a
        coor_b: coordinate of point b
        coor_c: coordinate of point c

    Returns:
        's': straight
        'l': left-hand turn
        'r': right-hand turn
    '''
    a = np.array(coor_a)
    b = np.array(coor_b)
    c = np.array(coor_c)
    v_ab = b - a
    v_bc = c - b
    cosangle = v_ab.dot(v_bc) / (np.linalg.norm(v_bc) * np.linalg.norm(v_ab) + 1e-16)
    res =  math.acos(cosangle) * 180 / np.pi if np.cross(v_ab, v_bc) < 0 else -math.acos(cosangle) * 180 / np.pi
    return res if not math.isnan(res) else 0




class NodeInPathGraph:
    def __init__(self, window, node):
        self.window = window
        self.node = node

    def __eq__(self, other):
        return self.window == other.window and self.node == other.node

    def calVal(self, estimationModel, edgesGdf, prevEdge):
        if self.window.midSeg == -1:
            return 0
        else:
            numericalFeatures, categoricalFeatures = self.window.extractFeatures(edgesGdf, prevEdge)
            return estimationModel.predict(numericalFeatures, categoricalFeatures)

    def generateNextNode(self, edgesGdf, destNode):
        if self.node == -1 or self.node == destNode:
            nextNodeId = -1
            nextWindow = Window(self.window.midSeg, self.window.sucSeg, -1)
            nextNodes = NodeInPathGraph(nextWindow, nextNodeId)
            return [nextNodes]
        else:
            edgesGdfFromNode = edgesGdf[edgesGdf['u'] == self.node]
            nextNodesList = []
            for edgeIdInGdf in list(edgesGdfFromNode.index):
                nextNodeId = edgesGdfFromNode.loc[edgeIdInGdf, 'v']
                nextWindow = Window(self.window.midSeg, self.window.sucSeg, edgeIdInGdf)
                nextNodesList.append(NodeInPathGraph(nextWindow,nextNodeId))
            return nextNodesList



class OsmGraph:

    def __init__(self, g):
        self.graph = g

    def saveHmlTo(self, folderAddress):
        os.makedirs(folderAddress)
        ox.save_graphml(self.graph, filepath=os.path.join(folderAddress, 'osmGraph.graphml'))

    def graphToGdfs(self):
        nodes, edges = ox.graph_to_gdfs(self.graph, nodes=True, edges=True)
        return nodes, edges

    def getEdges(self):
        _, edges = self.graphToGdfs()
        return edges

    def getNodes(self):
        nodes, _ = self.graphToGdfs()
        return nodes

    def removeIsolateNodes(self):
        self.graph.remove_nodes_from(list(nx.isolates(self.graph)))

    def getNearestNode(self, yx):
        return ox.get_nearest_node(self.graph, yx, method='euclidean')

    def getODNodesFromODPair(self, odPair):
        origNode = self.getNearestNode(odPair.origin.yx())
        targetNode = self.getNearestNode(odPair.destination.yx())
        return origNode, targetNode

    def plotPath(self, path):
        fig, ax = ox.plot_graph_route(self.graph, path)

    def shortestPath(self, origNode, destNode):
        return nx.shortest_path(G=self.graph, source=origNode, target=destNode, weight='length')

    def ecoPath(self, origNode, destNode):
        self.origNode = origNode
        self.destNode = destNode
        self.estimationModel = EstimationModel("fuel")
        ecoPath, ecoEnergy = self.dijkstra()

    def fastestPath(self, origNode, destNode):
        self.origNode = origNode
        self.destNode = destNode
        self.estimationModel = EstimationModel("time")
        fastestPath, shortestTime = self.dijkstra()

    def findMinValNotPassedNode(self):
        return min(self.notPassedNodeDict, key=self.notPassedNodeDict.get)



    def updateWith(self, curNode):
        self.passedNodes.add(curNode)
        self.notPassedNodeSet.remove(curNode)
        edgesGdfFromNode = self.edgesGdf[self.edgesGdf['u'] == curNode]
        for edgeIdInGdf in list(edgesGdfFromNode.index):
            nextNode = edgesGdfFromNode.loc[edgeIdInGdf, 'v']
            if edgeIdInGdf in self.edgeFeatureDictionary:
                curEdgeFeature = self.edgeFeatureDictionary[edgeIdInGdf]
            else:
                curEdgeFeature = self.extractEdgeFeature(edgesGdfFromNode.loc[edgeIdInGdf])
                self.edgeFeatureDictionary[edgeIdInGdf] = curEdgeFeature
            value = self.calval(curNode, nextNode)
            if nextNode not in self.valueOfNodes or self.valueOfNodes[curNode] + value < self.valueOfNodes[nextNode]:
                self.valueOfNodes[nextNode] = self.valueOfNodes[curNode] + value
                self.prevEdgeOfNode[nextNode] = edgeIdInGdf


    def dijkstra(self):
        self.passedNodesSet = set()
        self.notPassedNodeDict = dict()
        #self.nodesGdf = self.getNodes()
        #nodesSet = set(self.nodesGdf.index)
        #self.notPassedNodeSet = copy.deepcopy(nodesSet)
        #self.notPassedNodeSet.remove(self.origNode)
        self.edgesGdf = edgePreprocessing(self.getEdges(), self.getNodes())
        dummyWindow = Window(-1,-1,-1)
        dummyOriNodeInPathGraph = NodeInPathGraph(dummyWindow, self.origNode)
        dummyDestNodeInPathGraph = NodeInPathGraph(dummyWindow, -1)
        edgesGdfFromOrigNode = self.edgesGdf[self.edgesGdf['u'] == self.origNode]
        for origEdgeIdInGdf in list(edgesGdfFromOrigNode.index):
            nextNodeId = edgesGdfFromOrigNode.loc[origEdgeIdInGdf, 'v']
            nextWindow = Window(dummyWindow.midSeg, dummyWindow.sucSeg, nextNodeId)
            self.notPassedNodeDict[NodeInPathGraph(nextWindow, nextNodeId)] = 0



        if not len(edgesGdfFromOrigNode):
            print("not path from node:", self.origNode)
            return
        else:
            while self.destNode not in self.passedNodesSet:
                if len(self.notPassedNodeDict) == 0:
                    print("No route")
                    return
                else:
                    curNodeInPathGraph = self.findMinValNotPassedNode()
                    valOfCurNode = self.notPassedNodeDict.pop(curNodeInPathGraph)
                    self.passedNodesSet.add(curNodeInPathGraph)
                    nextNodeList = curNodeInPathGraph.generateNextNode(self.edgesGdf, self.destNode)
                    for nextNode in nextNodeList:
                        valOfNextNode = nextNode.calVal(self.estimationModel,self.edgesGdf,curNodeInPathGraph.window.prevSeg)
                        if valOfNextNode not in self.notPassedNodeDict or valOfNextNode + valOfCurNode < self.notPassedNodeDict[nextNode]:
                            self.notPassedNodeDict[nextNode] = valOfNextNode + valOfCurNode
            #############
            return self.notPassedNodeDict[self.destNode]


class GraphFromHmlFile(OsmGraph):
    def __init__(self, hmlAddress):
        self.graph = ox.load_graphml(hmlAddress)


class GraphFromBbox(OsmGraph):
    def __init__(self, boundingBox):
        self.graph = ox.graph_from_polygon(boundingBox.polygon(), network_type='drive')


class GraphFromGdfs(OsmGraph):
    def __init__(self, nodes, edges):
        self.graph = ox.utils_graph.graph_from_gdfs(nodes, edges)


def main():
    locationRequest = LocationRequest()
    osmGraphInBbox = extractGraphOf(locationRequest.boundingBox)
    #graphInMurphy = extractGraphInMurphy(osmGraphInBbox)
    graphInMurphy = osmGraphInBbox
    graphInMurphy.removeIsolateNodes()
    shortestPath = findShortestPath(graphInMurphy, locationRequest.odPair)
    #ecoRoute, energyOnEcoRoute = findEcoPathAndCalEnergy(graphInMurphy, locationRequest.odPair)
    #fastestPath, shortestTime = findFastestPathAndCalTime(graphInMurphy, locationRequest.odPair)


def extractGraphOf(boundingBox):
    folderOfGraph = r'GraphDataInBbox'
    if os.path.exists(folderOfGraph):
        print("reloading graph..")
        osmGraph = GraphFromHmlFile(os.path.join(folderOfGraph, 'osmGraph.graphml'))
    else:
        print("downloading graph..")
        osmGraph = GraphFromBbox(boundingBox)
        osmGraph.saveHmlTo(folderOfGraph)
    fig, ax = ox.plot_graph(osmGraph.graph)
    return osmGraph


def extractGraphInMurphy(osmGraph):
    nodes, edges = osmGraph.graphToGdfs()
    edgesInMurphy = extractEdgesInMurphy(edges)
    graphInMurphy = GraphFromGdfs(nodes, edgesInMurphy)
    return graphInMurphy


def extractEdgesInMurphy(edges):
    edges['uvPair'] = edges.apply(lambda x: (x.u, x.v), axis=1)
    segmentElevationChange = np.load('segmentElevationChange.npy', allow_pickle=True).item()
    edges['isInMurphy'] = edges.uvPair.apply(lambda x: x in segmentElevationChange)
    return edges[edges.isInMurphy]


def findShortestPath(osmGraph, odPair):
    origNode, targetNode = osmGraph.getODNodesFromODPair(odPair)
    shortestPath = osmGraph.shortestPath(origNode,targetNode)
    #ox.plot_graph(osmGraph)
    osmGraph.plotPath(shortestPath)
    return shortestPath


def findEcoPathAndCalEnergy(osmGraph, odPair):
    origNode, targetNode = osmGraph.getODNodesFromODPair(odPair)
    ecoPath, ecoEnergy = osmGraph.ecoPath(origNode,targetNode)
    osmGraph.plotPath(ecoPath)
    return ecoPath, ecoEnergy


def findFastestPathAndCalTime(osmGraph, odPair):
    origNode, targetNode = osmGraph.getODNodesFromODPair(odPair)
    fastestPath, shortestTime = osmGraph.fastestPath(origNode,targetNode)
    osmGraph.plotPath(fastestPath)
    return fastestPath, shortestTime




if __name__ == '__main__':
    main()


