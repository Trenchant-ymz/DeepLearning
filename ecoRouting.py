import numpy as np
import os
import pandas as pd
from osmgraph import GraphFromBbox, GraphFromHmlFile, GraphFromGdfs
from spaitalShape import Point, OdPair, Box
from edgeGdfPreprocessing import edgePreprocessing
import osmnx as ox
import math
import time

# Profiling: python -m cProfile -o profile.pstats ecoRouting.py
# Visualize profile: snakeviz profile.pstats


class LocationRequest:
    def __init__(self):
        # Murphy Depot
        # (longitude, latitude)
        self.origin = Point(-93.2219, 44.979)
        # test
        #self.origin = Point(-93.4254, 44.7888)
        #self.origin = Point(-93.470167, 44.799720)
        #origin = Point(-93.2466, 44.8959)
        # Shakopee East (Depot):
        self.destination = Point(-93.4620, 44.7903)
        #self.destination = Point(-93.230358, 44.973583)

        self.odPair = OdPair(self.origin, self.destination)
        self.boundingBox = Box(-93.4975, -93.1850, 44.7458, 45.0045)
        self.mass = 23000
        # 9 am
        self.timeOfTheDay = 9
        # Monday
        self.dayOfTheWeek = 1



def main():
    locationRequest = LocationRequest()
    osmGraphInBbox = extractGraphOf(locationRequest.boundingBox)
    nodes, edges = osmGraphInBbox.graphToGdfs()
    extractElevation(nodes, edges)
    edges = edgePreprocessing(nodes, edges, locationRequest)
    graphWithElevation = GraphFromGdfs(nodes, edges)
    graphWithElevation.removeIsolateNodes()
    print(graphWithElevation.getEdges().loc[54197])
    print(graphWithElevation.getEdgesDict()[54197])
    testfunction(graphWithElevation.getEdgesDict())
    # shortest route
    shortestNodePath = findShortestPath(graphWithElevation, locationRequest)
    shortestPath = nodePathTOEdgePath(shortestNodePath, edges)
    calAndPrintPathAttributes(graphWithElevation, shortestPath, "shortestPath")
    # eco route
    ecoRoute, energyOnEcoRoute, ecoEdgePath = findEcoPathAndCalEnergy(graphWithElevation, locationRequest)
    calAndPrintPathAttributes(graphWithElevation, ecoEdgePath, "ecoRoute")
    # fastest route
    fastestPath, shortestTime, fastestEdgePath = findFastestPathAndCalTime(graphWithElevation, locationRequest)
    calAndPrintPathAttributes(graphWithElevation, fastestEdgePath, "fastestPath")
    graphWithElevation.plotPathList([shortestNodePath, ecoRoute, fastestPath],'routing result.pdf')


def testfunction(edgesGdf):
    from window import Window
    w = Window(61851, 61877, 61893, 33971)
    print(w.extractFeatures(edgesGdf))
    return

def extractGraphOf(boundingBox):
    folderOfGraph = r'GraphDataInBbox'
    if os.path.exists(folderOfGraph):
        print("reloading graph..")
        osmGraph = GraphFromHmlFile(os.path.join(folderOfGraph, 'osmGraph.graphml'))
    else:
        print("downloading graph..")
        osmGraph = GraphFromBbox(boundingBox)
        osmGraph.saveHmlTo(folderOfGraph)
    # fig, ax = ox.plot_graph(osmGraph.graph)
    return osmGraph


def extractElevation(nodes, edges):
    extractNodesElevation(nodes)
    extractEdgesElevation(nodes, edges)



def extractNodesElevation(nodes):
    nodesElevation = pd.read_csv("nodesWithElevation.csv", index_col=0)
    nodes['indexId'] = nodes.index
    nodes['elevation'] = nodes.apply(lambda x: nodesElevation.loc[x['indexId'], 'MeanElevation'], axis=1)



def extractEdgesElevation(nodesWithElevation, edges):
    edges['uElevation'] = edges['u'].apply(lambda x: nodesWithElevation.loc[x,'elevation'])
    edges['vElevation'] = edges['v'].apply(lambda x: nodesWithElevation.loc[x,'elevation'])


def extractGraphInMurphy(nodes, edges):
    edgesInMurphy = extractEdgesInMurphy(edges)
    graphInMurphy = GraphFromGdfs(nodes, edgesInMurphy)
    return graphInMurphy


def extractEdgesInMurphy(edges):
    edges['uvPair'] = edges.apply(lambda x: (x.u, x.v), axis=1)
    segmentElevationChange = np.load('statistical data/segmentElevationChange.npy', allow_pickle=True).item()
    edges['isInMurphy'] = edges.uvPair.apply(lambda x: x in segmentElevationChange)
    return edges[edges.isInMurphy]


def findShortestPath(osmGraph, localRequest):
    shortestPath = osmGraph.shortestPath(localRequest)
    print("shortestPath:", shortestPath)
    # ox.plot_graph(osmGraph)
    #osmGraph.plotPath(shortestPath, "shortest route.pdf")
    return shortestPath


def findEcoPathAndCalEnergy(osmGraph, localRequest):
    ecoPath, ecoEnergy , ecoEdgePath = osmGraph.ecoPath(localRequest)
    print("ecoPath:", ecoPath, "ecoEnergy:", ecoEnergy, ecoEdgePath)
    #osmGraph.plotPath(ecoPath, "eco route.pdf")
    return ecoPath, ecoEnergy,  ecoEdgePath


def findFastestPathAndCalTime(osmGraph, localRequest):

    fastestPath, shortestTime, fastEdgePath = osmGraph.fastestPath(localRequest)
    print("fastestPath:", fastestPath, "shortestTime:", shortestTime, fastEdgePath)
    #osmGraph.plotPath(fastestPath,"fastest route.pdf")
    return fastestPath, shortestTime, fastEdgePath


def nodePathTOEdgePath(nodePath, edgesGdf):
    edgePath = []
    for i, OdPair in enumerate(zip(nodePath[:-1], nodePath[1:])):
        segmentId = edgesGdf[edgesGdf['odPair'] == OdPair].index[0]
        edgePath.append(segmentId)
    return edgePath



def calAndPrintPathAttributes(osmGraph, edgePath, pathname):
    numberOfSegments = len(edgePath)
    length = osmGraph.totalLength(edgePath)
    energy = osmGraph.totalEnergy(edgePath)
    time = osmGraph.totalTime(edgePath)
    print(pathname+":"+f"{numberOfSegments} segments, {length} meters, {energy} liters, {time} seconds")
    return


'''
def calAndPrintPathAttributes(osmGraph, path, edgePath, pathname):
    numberOfSegments = len(path)
    length = osmGraph.totalLength(path)
    energy = osmGraph.totalEnergy(path)
    time = osmGraph.totalTime(path)
    print(pathname+":"+f"{numberOfSegments} segments, {length} meters, {energy} liters, {time} seconds")
    return
'''


if __name__ == '__main__':
    main()


