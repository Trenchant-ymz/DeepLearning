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
        #origin = Point(-93.2219, 44.979)
        origin = Point(-93.2466, 44.8959)
        # Shakopee East (Depot):
        #destination = Point(-93.4620, 44.7903)
        destination = Point(-93.4495, 44.8611)

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
        numericalInputData = torch.Tensor(numericalInputData)
        categoricalInputData = torch.LongTensor(categoricalInputData)
        return self.model(numericalInputData.to(self.device), categoricalInputData.to(self.device)).item()


class Window:
    def __init__(self, prevSeg, midSeg, sucSeg):
        self.prevSeg = prevSeg
        self.midSeg = midSeg
        self.sucSeg = sucSeg

    def extractFeatures(self, edgesGdf, nodesGdf):
        prevSegNumFeature, prevSegCatFeature = edgeFeature(self.prevSeg, edgesGdf, nodesGdf)
        midSegNumFeature, midSegCatFeature = edgeFeature(self.midSeg, edgesGdf, nodesGdf)
        sucSegNumFeature, sucSegCatFeature = edgeFeature(self.sucSeg, edgesGdf, nodesGdf)
        numericalFeatures = concatFeatures(prevSegNumFeature, midSegNumFeature, sucSegNumFeature)
        categoricalFeatures = concatFeatures(prevSegCatFeature, midSegCatFeature, sucSegCatFeature)
        return numericalFeatures, categoricalFeatures


def edgeFeature(segmentIDInGdf, edgesGdf, nodesGdf):
    edge = edgesGdf.loc[segmentIDInGdf]
    edge['uSignal'] = nodesGdf.loc[edge['u'], 'highway']
    edge['vSignal'] = nodesGdf.loc[edge['v'], 'highway']








class NodeInPathGraph:
    def __init__(self, window, node):
        self.window = window
        self.node =  node

    def calVal(self, estimationModel, edgesGdf, nodesGdf):
        numericalFeatures, categoricalFeatures = self.window.extractFeatures(edgesGdf, nodesGdf)
        return estimationModel.predict(numericalFeatures, categoricalFeatures)

    def generateNextNode(self, edgesGdf):
        edgesGdfFromNode  = edgesGdf[edgesGdf['u'] == self.node]
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

    def findMinValNode(self):
        pass

    def extractEdgeFeature(self, edgesGdfFromNode):
        pass

    def calValue(self, curNode, nextNode):
        pass

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
        self.passedNodes = set([self.origNode])
        self.nodesGdf = self.getNodes()
        nodesSet = set(self.nodesGdf.index)
        self.notPassedNodeSet = copy.deepcopy(nodesSet)
        self.notPassedNodeSet.remove(self.origNode)
        self.edgesGdf = self.getEdges()

        self.edgesGdf = preprocessing(self.edgesGdf, self.nodesGdf)

        self.edgeFeatureDictionary = dict()
        self.valueOfNodes = dict()
        self.prevEdgeOfNode = dict()
        self.prevEdgeOfNode[self.origNode] = -1
        self.valueOfNodes[self.origNode] = 0
        edgesGdfFromOrigNode  = self.edgesGdf[self.edgesGdf['u'] == self.origNode]

        if not len(edgesGdfFromOrigNode):
            print("not path from node:", self.origNode)
            return
        else:
            while self.destNode not in self.passedNodes:
                time = 0
                if time == 0:
                    curNode = self.origNode
                else:
                    curNode = self.findMinValNode()
                self.updateWith(curNode)




                '''
                minValueSegment = find min value in notPassedNodes
                passedNodes.add
                notPassedNodes.remove
                for nextNode in notPassedNodes:
                    if minValueSegment -> nextNode == True:
                        for edge in edge(minValueSegment, nextNode):
                            numericalFeatureOfEdge, categoricalFeatureOfEdge = extractFeature(edge)
                            succeedEdges = calculateNextEdge(nextNode)
                            for edgeSucceed in succeedEdges:
                                windowOfNumericalFeatures, windowOfCategoricalFeatures = windowgeneration(edge, edgeBefore, edgeSucceed)
                                predictResult = self.estimationModel.predict(windowOfNumericalFeatures, windowOfCategoricalFeatures)
                            averagePredict = averge(predictResult)
                            if averagePredict + valueofminValuesegment < valueNextSegment:
                                valueNextSegment = averagePredict + valueofminValuesegment
                '''


def preprocessing(edgesGdf, nodesGdf):
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

    edgesGdf['previousOrientation'] = (edgesGdf.previousOrientation + 1.46016587027665) / 33.3524612794841

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
    edgesGdf['bridge_d'] = edgesGdf.bridge.apply(lambda x: bridgeOrNot(x))


    # endpoints
    edgesGdf['uSignal'] = edgesGdf.u.apply(lambda x: nodesGdf.loc[x, 'highway']).fillna("None")
    edgesGdf['vSignal'] = edgesGdf.v.apply(lambda x: nodesGdf.loc[x, 'highway']).fillna("None")
    endpoints_dictionary = np.load('endpoints_dictionary.npy', allow_pickle=True).item()
    edgesGdf['uSignalCategoried'] = edgesGdf.uSignal.apply(lambda x: endpoints_dictionary[x] if x in endpoints_dictionary else 0)
    edgesGdf['vSignalCategoried'] = edgesGdf.vSignal.apply(lambda x: endpoints_dictionary[x] if x in endpoints_dictionary else 0)








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
    if pd.isna(array_like['lanes']):
        return 0
    if array_like['lanes'].isalpha():
        return 0
    if array_like['lanes'].isalnum():
        return int(array_like['lanes']) if int(array_like['lanes']) > 0 else 0
    else:
        return 0

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
    graphInMurphy = extractGraphInMurphy(osmGraphInBbox)
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
    #fig, ax = ox.plot_graph(osmGraph)
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


