import numpy as np
import os
import pandas as pd
from osmgraph import GraphFromBbox, GraphFromHmlFile, GraphFromGdfs
from spaitalShape import Point, OdPair, Box
from edgeGdfPreprocessing import edgePreprocessing
import plotly.graph_objects as go
import plotly
import osmnx as ox
import math
import time
from estimationModel import EstimationModel
from lookUpTable import LookUpTable

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
        self.temperature = 1
        self.mass = 23000
        # Monday
        self.dayOfTheWeek = 1
        # 9 am
        time = 9
        self.timeOfTheDay = self.calTimeStage(time)

    def calTimeStage(self, t):
        return t // 4 + 1


class ParameterForTableIni:
    def __init__(self, windowList, osmGraph, estMode = 'fuel'):
        self.temperatureList = [1]
        self.massList = [23000]
        self.dayList = [1,2]
        self.timeList = [1,3]
        self.windowList = windowList
        self.osmGraph = osmGraph
        self.estMode = estMode
        self.estimationModel = EstimationModel(estMode)


def main():
    locationRequest = LocationRequest()
    osmGraphInBbox = extractGraphOf(locationRequest.boundingBox)
    nodes, edges = osmGraphInBbox.graphToGdfs()
    extractElevation(nodes, edges)
    # for look-up-table method, the edges donot neet to be preprocessed.
    edges = edgePreprocessing(nodes, edges, locationRequest.temperature, locationRequest.mass ,locationRequest.dayOfTheWeek, locationRequest.timeOfTheDay)
    graphWithElevation = GraphFromGdfs(nodes, edges)
    graphWithElevation.removeIsolateNodes()
    print('Graph loaded!')
    estMode = "fuel"
    filename = "lookUpTableFor" + estMode
    # train new table and save it to filename.pkl
    #lookUpTable = trainNewLUTable(graphWithElevation, locationRequest, filename, mode=estMode)
    # load table from filename.pkl
    lookUpTable = LookUpTable(locationRequest, filename)
    #windowList = graphWithElevation.extractAllWindows(4)
    #energyEst = lookUpTable.extractValue(windowList[0])
    #print(energyEst)
    print(len(lookUpTable))
    #print(windowList[0], energyEst)
    # shortest route
    #shortestNodePath = findShortestPath(graphWithElevation, locationRequest)
    #shortestPath = nodePathTOEdgePath(shortestNodePath, edges)
    #calAndPrintPathAttributes(graphWithElevation, shortestPath, "shortestPath")
    # eco route
    ecoRoute, energyOnEcoRoute, ecoEdgePath = findEcoPathAndCalEnergy(graphWithElevation, locationRequest, lookUpTable)
    calAndPrintPathAttributes(graphWithElevation, ecoEdgePath, "ecoRoute")
    # fastest route
    #fastestPath, shortestTime, fastestEdgePath = findFastestPathAndCalTime(graphWithElevation, locationRequest)
    #calAndPrintPathAttributes(graphWithElevation, fastestEdgePath, "fastestPath")
    #plotRoutes([shortestPath, ecoEdgePath, fastestEdgePath], edges, ['green', 'red', 'blue'], ['shortest route', 'eco route', 'fastest route'])
    #graphWithElevation.plotPathList([shortestNodePath, ecoRoute, fastestPath],'routing result.pdf')


def trainNewLUTable(graphWithElevation, locationRequest, filename, mode='fuel'):
    windowList = graphWithElevation.extractAllWindows(4)
    print(len(windowList))
    paramForTable = ParameterForTableIni(windowList, graphWithElevation, mode)
    lookUpTable = LookUpTable(locationRequest, filename, generateNewTable=True, parameterForTableIni=paramForTable)
    return lookUpTable

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


def findEcoPathAndCalEnergy(osmGraph, localRequest, lookUpTable):
    ecoPath, ecoEnergy , ecoEdgePath = osmGraph.ecoPath(localRequest, lookUpTable)
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

# plot map matching results
def plotRoutes(routeList, network_gdf, colorLists, nameLists):
    directory = './results'
    colorLists = ['green', 'red', 'blue']
    if not os.path.exists(directory):
        os.makedirs(directory)
    edgeLongList = []
    edgeLatList = []
    for i in range(len(routeList)):
        route = routeList[i]
        long_edge = []
        lat_edge = []
        for j in route:
            e = network_gdf.loc[j]
            if 'geometry' in e:
                xs, ys = e['geometry'].xy
                z = list(zip(xs, ys))
                l1 = list(list(zip(*z))[0])
                l2 = list(list(zip(*z))[1])
                for k in range(len(l1)):
                    long_edge.append(l1[k])
                    lat_edge.append(l2[k])
        if i == 0:
            fig = go.Figure(go.Scattermapbox(
                name = nameLists[i],
                mode = "lines",
                lon = long_edge,
                lat = lat_edge,
                marker = {'size': 5, 'color':colorLists[i]},
                line = dict(width = 3, color = colorLists[i])))
        else:
            fig.add_trace(go.Scattermapbox(
                name = nameLists[i],
                mode = "lines",
                lon = long_edge,
                lat = lat_edge,
                marker = {'size': 5, 'color':colorLists[i]},
                line = dict(width = 3, color = colorLists[i])))
    # getting center for plots:
    lat_center = np.mean(lat_edge)
    long_center = np.mean(long_edge)
    zoom = 9.5
    # defining the layout using mapbox_style
    fig.update_layout(mapbox_style="stamen-terrain",
        mapbox_center_lat = 30, mapbox_center_lon=-80)
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0},
                    mapbox = {
                            'center': {'lat': lat_center,
                            'lon': long_center},
                            'zoom': zoom})

    plotly.offline.plot(fig,filename = os.path.join(directory,'resultnewdropped.html'),auto_open=True)

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


