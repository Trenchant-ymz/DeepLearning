import networkx as nx
import osmnx as ox
import os
from edgeGdfPreprocessing import edgePreprocessing
from estimationModel import EstimationModel
from window import Window
from pathGraph import NodeInPathGraph
import time
from collections import defaultdict
from routingAlgorithms import Dijkstra, AStar
import plotly.graph_objects as go
import numpy as np
import plotly


class OsmGraph:

    def __init__(self, g):
        self.graph = g
        self.nodesGdf, self.edgesGdf = self.graphToGdfs()

    def saveHmlTo(self, folderAddress):
        os.makedirs(folderAddress)
        ox.save_graphml(self.graph, filepath=os.path.join(folderAddress, 'osmGraph.graphml'))

    def graphToGdfs(self):
        nodes, edges = ox.graph_to_gdfs(self.graph, nodes=True, edges=True)
        return nodes, edges

    def getEdges(self):
        _, edges = self.graphToGdfs()
        self.uToV = defaultdict(list)
        edges.apply(lambda x: self.__update(x), axis=1)
        return edges

    def getEdgesDict(self):
        _, edges = self.graphToGdfs()
        self.uToV = defaultdict(list)
        edges.apply(lambda x: self.__update(x), axis=1)
        return edges.to_dict('index')

    def __update(self,x):
        self.uToV[x.u].append((x.name, x.v))

    def getNodes(self):
        nodes, _ = self.graphToGdfs()
        return nodes

    def removeIsolateNodes(self):
        self.graph.remove_nodes_from(list(nx.isolates(self.graph)))
        #self.nodesGdf, self.edgesGdf = self.graphToGdfs()

    def getNearestNode(self, yx):
        return ox.get_nearest_node(self.graph, yx, method='euclidean')

    def getODNodesFromODPair(self, odPair):
        origNode = self.getNearestNode(odPair.origin.yx())
        targetNode = self.getNearestNode(odPair.destination.yx())
        return origNode, targetNode

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

    def ecoPath(self, localRequest):
        self.origNode, self.destNode = self.getODNodesFromODPair(localRequest.odPair)
        self.estimationModel = EstimationModel("fuel")

        #ecoPath, ecoEnergy, ecoEdgePath = self.dijkstra()
        ecoPath, ecoEnergy, ecoEdgePath = self.aStar(localRequest)
        return ecoPath, ecoEnergy, ecoEdgePath

    def fastestPath(self, localRequest):
        self.origNode, self.destNode = self.getODNodesFromODPair(localRequest.odPair)
        self.estimationModel = EstimationModel("time")
        fastestPath, shortestTime, fastestEdgePath = self.aStar(localRequest)
        return fastestPath, shortestTime, fastestEdgePath

    def dijkstra(self):
        routingModel = Dijkstra(self.getEdgesDict(), self.uToV, self.origNode, self.destNode, self.estimationModel)
        return routingModel.routing()

    def aStar(self, localRequest):
        routingModel = AStar(self.getEdgesDict(), self.uToV, self.origNode, self.destNode, self.estimationModel,
                             localRequest, self.getNodes())
        #print('initialized')
        return routingModel.routing()

    def totalLength(self, path):
        length = 0
        for i in path:
            length += self.edgesGdf.loc[i, 'length']
        return length

    def totalEnergy(self, path):
        return self.__calculateValue(path, "fuel")

    def totalTime(self, path):
        return self.__calculateValue(path, "time")

    def __calculateValue(self, path, estimationType):
        edgeDict = self.getEdgesDict()
        pointList = []
        estimationModel = EstimationModel(estimationType)
        value = 0
        firstSeg = path[0]
        window = Window(-1, -1, -1, firstSeg)
        prevWindowSeg = -1
        for i in range(len(path)):
            window.minusSeg = window.prevSeg
            window.prevSeg = window.midSeg
            window.midSeg = window.sucSeg
            if i < len(path)-1:
                window.sucSeg = path[i+1]
            else:
                window.sucSeg = -1
            numericalFeatures, categoricalFeatures = window.extractFeatures(edgeDict)
            #print(numericalFeatures, categoricalFeatures)
            addValue = estimationModel.predict(numericalFeatures, categoricalFeatures)
            value += addValue
            pointList.append((str(window.minusSeg)+',' + str(window),numericalFeatures, categoricalFeatures, addValue, value))
        f = estimationType+'.txt'
        filename = open(f, 'w')
        for p in pointList:
            filename.write(str(p) + "\n")
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