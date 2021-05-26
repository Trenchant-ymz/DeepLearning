import numpy as np
import os
import pandas as pd
from osmgraph import GraphFromBbox, GraphFromHmlFile, GraphFromGdfs
from edgeGdfPreprocessing import edgePreprocessing
import osmnx as ox
from datetime import datetime


def extractGraphAndPreprocessing(locationRequest):
    osmGraphInBbox = extractGraphOf(locationRequest.boundingBox)
    nodes, edges = osmGraphInBbox.graphToGdfs()
    extractElevation(nodes, edges)
    edges = edgePreprocessing(nodes, edges)
    graphWithElevation = GraphFromGdfs(nodes, edges)
    graphWithElevation.removeIsolateNodes()
    return graphWithElevation, edges


def extractGraphOf(boundingBox):
    folderOfGraph = r'GraphDataInBbox'
    if os.path.exists(folderOfGraph):
        print("reloading graph..")
        osmGraph = GraphFromHmlFile(os.path.join(folderOfGraph, 'osmGraph.graphml'))
    else:
        print("downloading graph..")
        osmGraph = GraphFromBbox(boundingBox)
        osmGraph.saveHmlTo(folderOfGraph)
    fig, ax = ox.plot_graph(osmGraph.graph, node_size=5)
    fig.savefig('graph.pdf')
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


def findShortestPath(osmGraph, odPair):
    origNode, targetNode = osmGraph.getODNodesFromODPair(odPair)
    shortestPath = osmGraph.shortestPath(origNode, targetNode)
    print("shortestPath:", shortestPath)
    # ox.plot_graph(osmGraph)
    osmGraph.plotPath(shortestPath, "shortest route.pdf")
    time0 = datetime.now()
    edges = osmGraph.getEdges()
    time1 = datetime.now()
    shortestEdgePath = nodePathTOEdgePath(shortestPath, edges)
    time2 = datetime.now()
    # print('getEdges', time1-time0, 'node->edge',time2-time1)
    calAndPrintPathAttributes(osmGraph, shortestEdgePath, "shortestPath")
    return shortestPath


def findEcoPathAndCalEnergy(osmGraph, odPair):
    origNode, targetNode = osmGraph.getODNodesFromODPair(odPair)
    ecoPath, ecoEnergy , ecoEdgePath = osmGraph.ecoPath(origNode,targetNode)
    print("ecoPath:", ecoPath, "ecoEnergy:", ecoEnergy, ecoEdgePath)
    osmGraph.plotPath(ecoPath, "eco route.pdf")
    calAndPrintPathAttributes(osmGraph, ecoEdgePath, "ecoRoute")
    return ecoPath, ecoEnergy,  ecoEdgePath


def findFastestPathAndCalTime(osmGraph, odPair):
    origNode, targetNode = osmGraph.getODNodesFromODPair(odPair)
    fastestPath, shortestTime, fastEdgePath = osmGraph.fastestPath(origNode,targetNode)
    print("fastestPath:", fastestPath, "shortestTime:", shortestTime, fastEdgePath)
    osmGraph.plotPath(fastestPath,"fastest route.pdf")
    calAndPrintPathAttributes(osmGraph, fastEdgePath, "fastestPath")
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