import networkx as nx
import osmnx as ox
import os
from edgeGdfPreprocessing import edgePreprocessing
from estimationModel import EstimationModel, MultiTaskEstimationModel
from window import Window, WindowFromList
from windowNode import NodeInPathGraph
import time
from collections import defaultdict
import routingAlgorithms
import plotly.graph_objects as go
import numpy as np
import plotly
import copy

class OsmGraph:

    def __init__(self, g):
        self.graph = g
        self.nodesGdf, self.edgesGdf = self.graphToGdfs()

    def saveHmlTo(self, folderAddress):
        ox.save_graphml(self.graph, filepath=os.path.join(folderAddress, 'osmGraph.graphml'))

    def graphToGdfs(self):
        nodes, edges = ox.graph_to_gdfs(self.graph, nodes=True, edges=True)
        return nodes, edges

    def getEdges(self):
        _, edges = self.graphToGdfs()
        return edges

    def getEdgesDict(self):
        _, edges = self.graphToGdfs()
        return edges.to_dict('index')

    def getUToV(self):
        _, edges = self.graphToGdfs()
        self.uToV = defaultdict(list)
        edges.apply(lambda x: self.__update(x), axis=1)
        return self.uToV

    def __update(self,x):
        self.uToV[x.u].append((x.name, x.v))

    def getNodes(self):
        nodes, _ = self.graphToGdfs()
        return nodes

    def saveNodesLatLong(self, filename):
        nodes = self.getNodes()
        nodeLatLong = nodes[['y', 'x']]
        nodeLatLong.columns = ['latitude', 'longitude']
        nodeLatLong.to_csv(filename)
        return

    def removeIsolateNodes(self):
        self.graph.remove_nodes_from(list(nx.isolates(self.graph)))
        #self.nodesGdf, self.edgesGdf = self.graphToGdfs()

    def getNearestNode(self, yx):
        return ox.get_nearest_node(self.graph, yx, method='euclidean')

    def getODNodesFromODPair(self, odPair):
        origNode = self.getNearestNode(odPair.origin.yx())
        targetNode = self.getNearestNode(odPair.destination.yx())
        return origNode, targetNode

    def plotGraph(self,filename):
        fig, ax = ox.plot_graph(self.graph)
        fig.savefig(filename)

    def plotPath(self, path, filename):
        fig, ax = ox.plot_graph_route(self.graph, path,node_size=5)
        fig.savefig(filename)

    def plotPathList(self, pathList, filename):
        fig, ax = ox.plot_graph_routes(self.graph, pathList, route_colors=['g', 'r', 'b'], node_size=5)
        fig.savefig(filename)


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
                    name=nameLists[i],
                    mode="lines",
                    lon=long_edge,
                    lat=lat_edge,
                    marker={'size': 5, 'color': colorLists[i]},
                    line=dict(width=3, color=colorLists[i])))
            else:
                fig.add_trace(go.Scattermapbox(
                    name=nameLists[i],
                    mode="lines",
                    lon=long_edge,
                    lat=lat_edge,
                    marker={'size': 5, 'color': colorLists[i]},
                    line=dict(width=3, color=colorLists[i])))
                # getting center for plots:
        lat_center = np.mean(lat_edge)
        long_center = np.mean(long_edge)
        zoom = 9.5
        # defining the layout using mapbox_style
        fig.update_layout(mapbox_style="stamen-terrain",
                          mapbox_center_lat=30, mapbox_center_lon=-80)
        fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0},
                          mapbox={
                              'center': {'lat': lat_center,
                                         'lon': long_center},
                              'zoom': zoom})

        plotly.offline.plot(fig, filename=os.path.join(directory, 'resultnewdropped.html'), auto_open=True)

    def plot_traj(data):
        lat = data['gps_Latitude'].tolist()
        long = data['gps_Longitude'].tolist()
        # adding the lines joining the nodes
        fig = go.Figure(go.Scattermapbox(
            name="Path",
            mode="lines",
            lon=long,
            lat=lat,
            marker={'size': 10},
            line=dict(width=4.5, color='blue')))
        # getting center for plots:
        lat_center = np.mean(lat)
        long_center = np.mean(long)
        # defining the layout using mapbox_style
        fig.update_layout(mapbox_style="stamen-terrain",
                          mapbox_center_lat=30, mapbox_center_lon=-80)
        # for different trips, maybe you should modify the zoom value
        fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0},
                          mapbox={
                              'center': {'lat': lat_center,
                                         'lon': long_center},
                              'zoom': 14})
        # you can change the name of the figure saved
        # pio.write_image(fig,'trajectory.png')
        #fig.write_html("results/file.html")
        fig.show()

    def shortestPath(self, localRequest):
        origNode, destNode = self.getODNodesFromODPair(localRequest.odPair)
        shortestPath = nx.shortest_path(G=self.graph, source=origNode, target=destNode, weight='length')
        return shortestPath

    def ecoPath(self, localRequest, lookUpTable):
        self.origNode, self.destNode = self.getODNodesFromODPair(localRequest.odPair)
        self.estimationModel = EstimationModel("fuel")
        #ecoPath, ecoEnergy, ecoEdgePath = self.dijkstra()
        ecoPath, ecoEnergy, ecoEdgePath = self.aStar(localRequest, lookUpTable)
        return ecoPath, ecoEnergy, ecoEdgePath

    def fastestPath(self, localRequest, lookUpTable = None):
        self.origNode, self.destNode = self.getODNodesFromODPair(localRequest.odPair)
        self.estimationModel =EstimationModel("time")
        fastestPath, shortestTime, fastestEdgePath = self.dijkstra(localRequest, lookUpTable)
        return fastestPath, shortestTime, fastestEdgePath

    def dijkstra(self, localRequest, lookUpTable):
        if lookUpTable is None:
            routingModel = routingAlgorithms.Dijkstra(self.getEdgesDict(), self.getUToV(), self.origNode, self.destNode, self.estimationModel)
        else:
            routingModel = routingAlgorithms.DijkstraFromLUTable(self.getEdgesDict(), self.getUToV(), self.origNode, self.destNode,
                                 self.estimationModel, lookUpTable)
        return routingModel.routing()

    def aStar(self, localRequest, lookUpTable):
        if lookUpTable is None:
            routingModel = routingAlgorithms.AStar(self.getEdgesDict(), self.getUToV(), self.origNode, self.destNode, self.estimationModel,
                             localRequest, self.getNodes())
        else:
            routingModel = routingAlgorithms.AStarFromLUTable(self.getEdgesDict(), self.getUToV(), self.origNode, self.destNode,
                                 self.estimationModel, localRequest, self.getNodes(), lookUpTable)
        #print('initialized')
        return routingModel.routing()

    def extractAllWindows(self, lenthOfWindows):
        uToV = self.getUToV()
        windowList, tempWindowStack, tempNodeIdStack = [], [], []
        for i in uToV:
            listOfNodes = uToV[i]
            for edgeIdAndV in listOfNodes:
                edgeIdInGdf = edgeIdAndV[0]
                nextNodeId = edgeIdAndV[1]
                tempWindowStack.append([edgeIdInGdf])
                tempNodeIdStack.append(nextNodeId)
            tempWindowStack.append([-1])
            tempNodeIdStack.append(i)
            tempWindowStack.append([-1, -1])
            tempNodeIdStack.append(i)
        while tempWindowStack:
            tempWindow = list(tempWindowStack.pop())

            tempNode = tempNodeIdStack.pop()
            listOfNodes = uToV[tempNode]
            #print(tempWindow, tempNode,  listOfNodes)
            if len(tempWindow) == lenthOfWindows-1:
                tempWindow.append(-1)
                windowList.append(WindowFromList(tempWindow))
                tempWindow.pop()
                for edgeIdAndV in listOfNodes:
                    edgeIdInGdf = edgeIdAndV[0]
                    tempWindow.append(edgeIdInGdf)
                    #copy.deepcopy(tempWindow)
                    windowList.append(WindowFromList(tempWindow))
                    tempWindow.pop()
            else:
                for edgeIdAndV in listOfNodes:
                    edgeIdInGdf = edgeIdAndV[0]
                    nextNodeId = edgeIdAndV[1]
                    tempWindow.append(edgeIdInGdf)
                    #copy.deepcopy(tempWindow)
                    tempWindowStack.append(tuple(tempWindow))
                    tempNodeIdStack.append(nextNodeId)
                    tempWindow.pop()
        return windowList


    def totalLength(self, path):
        length = 0
        for i in path:
            length += self.edgesGdf.loc[i, 'length']
        return length

    def totalEnergy(self, path):
        print(1)
        return self.__calculateValue(path, "fuel")

    def totalTime(self, path):
        return self.__calculateValue(path, "time")

    def __calculateValue(self, path, estimationType):
        edgeDict = self.getEdgesDict()
        pointList = []
        estimationModel = EstimationModel(estimationType)
        #estimationModel = EstimationModel(estimationType)
        value = 0
        firstSeg = path[0]
        window = Window(-1, -1, -1, firstSeg)
        # prevWindowSeg = -1
        for i in range(len(path)):
            window.minusSeg = window.prevSeg
            window.prevSeg = window.midSeg
            window.midSeg = window.sucSeg
            if i < len(path)-1:
                window.sucSeg = path[i+1]
            else:
                window.sucSeg = -1
            numericalFeatures, categoricalFeatures = window.extractFeatures(edgeDict)
            # print(numericalFeatures, categoricalFeatures)
            addValue = estimationModel.predictFromData(numericalFeatures, categoricalFeatures)
            value += addValue
            if path[i] in [58029, 59122, 62170, 6004, 52169]:
                print(value)
            pointList.append((str(window.midSeg), numericalFeatures[1], categoricalFeatures[1], addValue, value))
        f = estimationType+'.txt'
        filename = open(f, 'w')
        for p in pointList:
            filename.write(str(p) + "\n")
        filename.write("path: ")
        for p in path:
            filename.write(str(p) + ", ")
        filename.close()
        return value

    def __findSegId(self, path, i):
        OdPair = (path[i], path[i+1])
        segId = self.edgesGdf[self.edgesGdf['odPair'] == OdPair].index[0]
        return segId

    def __updateWindow(self, window, path, i):
        window.minusSeg = window.prevSeg
        window.prevSeg = window.midSeg
        window.midSeg = window.sucSeg
        if i < len(path) - 2:
            nextSeg = self.__findSegId(path, i+1)
            window.sucSeg = nextSeg
        else:
            window.sucSeg = -1
        return window


class GraphFromHmlFile(OsmGraph):
    def __init__(self, hmlAddress):
        self.graph = ox.load_graphml(hmlAddress)
        self.nodesGdf, self.edgesGdf = self.graphToGdfs()


class GraphFromBbox(OsmGraph):
    def __init__(self, boundingBox):
        self.graph = ox.graph_from_polygon(boundingBox.polygon(), network_type='drive')
        self.nodesGdf, self.edgesGdf = self.graphToGdfs()


class GraphFromGdfs(OsmGraph):
    def __init__(self, nodes, edges):
        self.graph = ox.utils_graph.graph_from_gdfs(nodes, edges)
        self.nodesGdf, self.edgesGdf = nodes, edges