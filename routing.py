import numpy as np
import os
import pandas as pd
from code.osmgraph import GraphFromBbox, GraphFromHmlFile, GraphFromGdfs
from code.spaitalShape import Point, OdPair, Box
from code.edgeGdfPreprocessing import edgePreprocessing
import plotly.graph_objects as go
import plotly
import osmnx as ox
import math
import time
from code.estimationModel import EstimationModel, MultiTaskEstimationModel
from code.lookUpTable import LookUpTable
import gc
import json

# Profiling: python -m cProfile -o profile.pstats routing.py
# Visualize profile: snakeviz profile.pstats

# packages: torch, osmnx = 0.16.1, tqdm, bintrees, plotly

class LocationRequest:
    def __init__(self, origin, destination, temperature, mass, dayOfTheWeek , timeOfTheDay, boundingBox):
        '''
        :param origin: (longitude, latitude)
        :param destination: (longitude, latitude)
        :param temperature: 1  # temperature === 1 since we don't use temperature as a feature right now
        :param mass: kg
        :param dayOfTheWeek: Monday => 1
        :param timeOfTheDay: 9am => 9
        :param boundingBox: bounding box for eco-routing
        '''

        # (longitude, latitude)
        self.origin = origin
        # test
        #self.origin = Point(-93.4254, 44.7888)
        #self.origin = Point(-93.470167, 44.799720)
        #origin = Point(-93.2466, 44.8959)
        self.destination = destination
        #self.destination = Point(-93.230358, 44.973583)

        self.odPair = OdPair(self.origin, self.destination)
        self.temperature = temperature
        self.mass = mass
        # Monday
        self.dayOfTheWeek = dayOfTheWeek
        # 9 am
        time = timeOfTheDay
        self.timeOfTheDay = self.calTimeStage(time)
        self.boundingBox = boundingBox

    def calTimeStage(self, t):
        return t // 4 + 1


class ParameterForTableIni:
    '''
    Used in trainNewLUTable
    Define the bins of lookup table
    estMode = 'fuel'/ 'time'
    '''
    def __init__(self, windowList, osmGraph, estMode = 'fuel'):
        self.temperatureList = [1]
        self.massList = [23000]
        self.dayList = [1]
        self.timeList = [3]
        self.windowList = windowList
        self.osmGraph = osmGraph
        self.estMode = estMode
        self.estimationModel = EstimationModel(estMode)
        #self.estimationModel = EstimationModel(estMode)


def main():
    # Murphy depot => Shakopee East (Depot)
    origin, destination = Point(-93.2219, 44.979), Point(-93.4620, 44.7903)
    temperature = 1
    mass = 23000
    # Monday
    dayOfTheWeek = 1
    # 9am
    timeOfTheDay = 9
    '''
    #big bounding box: from murphy company (-93.22025, 44.9827), travel 70 miles
    distance = 70
    distance = distance*1609.34 # mile->km
    bbox = ox.utils_geo.bbox_from_point((44.9827, -93.22025), dist=self.distance, project_utm = False, return_crs = False)
    boundingBox = Box(bbox[-1], bbox[-2], bbox[-3], bbox[-4])
    '''
    # small bounding box
    boundingBox = Box(-93.4975, -93.1850, 44.7458, 45.0045)
    # Request
    locationRequest = LocationRequest(origin, destination, temperature, mass, dayOfTheWeek , timeOfTheDay, boundingBox)

    osmGraphInBbox = extractGraphOf(locationRequest.boundingBox)
    nodes, edges = osmGraphInBbox.graphToGdfs()

    # extract elevation change of edges
    extractElevation(nodes, edges, boundingBox)

    # define lookUpTable to None if you don't want to use the lookuptable method
    # lookUpTable = None


    edges = edgePreprocessing(nodes, edges, locationRequest.temperature, locationRequest.mass, locationRequest.dayOfTheWeek, locationRequest.timeOfTheDay)

    graphWithElevation = GraphFromGdfs(nodes, edges)
    graphWithElevation.removeIsolateNodes()
    print('Graph loaded!')

    # filename for lookup table
    filenameFuel = "lUTableForFuel"

    # train new table and save it to filename.pkl
    #lookUpTable = trainNewLUTable(graphWithElevation, locationRequest, filenameFuel, mode="fuel")

    # load table from filename.pkl
    lookUpTable = LookUpTable(locationRequest, filenameFuel)

    #windowList = graphWithElevation.extractAllWindows(4)
    #print(len(lookUpTable))


    # eco route
    ecoRoute, energyOnEcoRoute, ecoEdgePath = findEcoPathAndCalEnergy(graphWithElevation, locationRequest, lookUpTable)
    print(len(ecoEdgePath))
    calAndPrintPathAttributes(graphWithElevation, ecoEdgePath, "ecoRoute")
    ecoRouteFileName= 'ecoRouteTest.json'
    saveRoutes(ecoEdgePath, graphWithElevation.getEdges(), ecoRouteFileName)

    # shortest route
    # shortestNodePath = findShortestPath(graphWithElevation, locationRequest)
    # shortestPath = nodePathTOEdgePath(shortestNodePath, edges)
    # calAndPrintPathAttributes(graphWithElevation, shortestPath, "shortestPath")

    # fastest route
    filenameTime = "lookUpTableForTime"
    # lookUpTable = trainNewLUTable(graphWithElevation, locationRequest, filenameTime, mode='time')
    # lookUpTable = LookUpTable(locationRequest, filenameTime)
    # fastestPath, shortestTime, fastestEdgePath = findFastestPathAndCalTime(graphWithElevation, locationRequest, lookUpTable)
    # calAndPrintPathAttributes(graphWithElevation, fastestEdgePath, "fastestPath")

    # save the routing results to the "./results/filename.html"
    # plotRoutes([ecoEdgePath, fastestEdgePath, shortestPath], graphWithElevation.getEdges(), ['green','red','blue'], ['eco route','fastest route','shortest route'], 'routingresults')
    # plotRoutes([ecoEdgePath], graphWithElevation.getEdges(), ['green'],['eco route'], 'testBigBox')
    # graphWithElevation.plotPathList([shortestNodePath, ecoRoute, fastestPath],'routing result.pdf')


def trainNewLUTable(graphWithElevation, locationRequest, filename, mode='fuel'):
    windowList = graphWithElevation.extractAllWindows(4)
    print(len(windowList))
    paramForTable = ParameterForTableIni(windowList, graphWithElevation, mode)
    del windowList
    gc.collect()
    lookUpTable = LookUpTable(locationRequest, filename, generateNewTable=True, parameterForTableIni=paramForTable)
    return lookUpTable

def extractGraphOf(boundingBox):
    folderOfGraph = r'Graphs/GraphDataInBbox'+str(boundingBox)
    print(folderOfGraph)
    if os.path.exists(os.path.join(folderOfGraph,'osmGraph.graphml')):
        print("reloading graph..")
        osmGraph = GraphFromHmlFile(os.path.join(folderOfGraph, 'osmGraph.graphml'))
    else:
        if not os.path.exists(folderOfGraph):
            os.makedirs(folderOfGraph)
        print("downloading graph..")
        osmGraph = GraphFromBbox(boundingBox)
        osmGraph.saveHmlTo(folderOfGraph)
    # fig, ax = ox.plot_graph(osmGraph.graph)
    return osmGraph


def extractElevation(nodes, edges, boundingBox):
    extractNodesElevation(nodes, boundingBox)
    extractEdgesElevation(nodes, edges)


def extractNodesElevation(nodes, boundingBox):
    filename = "nodesWithElevation"+str(boundingBox)+".csv"
    nodesElevation = pd.read_csv(os.path.join("statistical data", filename), index_col=0)
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


def findFastestPathAndCalTime(osmGraph, localRequest,lookUpTable):

    fastestPath, shortestTime, fastEdgePath = osmGraph.fastestPath(localRequest,lookUpTable)
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
def plotRoutes(routeList, network_gdf, colorLists, nameLists, filename):
    directory = './results'
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
            with open("ecoRouteLong.json", 'w') as f:
                # indent=2 is not needed but makes the file human-readable
                json.dump(long_edge, f, indent=2)
            with open("ecoRouteLat.json", 'w') as f:
                # indent=2 is not needed but makes the file human-readable
                json.dump(lat_edge, f, indent=2)
            with open("ecoRoute.json", 'w') as f:
                # indent=2 is not needed but makes the file human-readable
                json.dump(list(zip(lat_edge, long_edge)), f, indent=2)
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

    plotly.offline.plot(fig,filename = os.path.join(directory,filename+'.html'),auto_open=True)

def saveRoutes(route, network_gdf, filename):
    directory = './results'
    if not os.path.exists(directory):
        os.makedirs(directory)

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
    with open(os.path.join(directory,filename), 'w') as f:
        # indent=2 is not needed but makes the file human-readable
        json.dump(list(zip(lat_edge, long_edge)), f, indent=2)

    # with open("ecoRouteLong.json", 'w') as f:
    #     # indent=2 is not needed but makes the file human-readable
    #     json.dump(long_edge, f, indent=2)
    # with open("ecoRouteLat.json", 'w') as f:
    #     # indent=2 is not needed but makes the file human-readable
    #     json.dump(lat_edge, f, indent=2)



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


