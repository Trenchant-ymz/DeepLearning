import networkx as nx
import osmnx as ox
import os
from edgeGdfPreprocessing import edgePreprocessing
from estimationModel import EstimationModel
from window import Window
from pathGraph import NodeInPathGraph


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
        ecoPath, ecoEnergy = self.__dijkstra()
        return ecoPath, ecoEnergy

    def fastestPath(self, origNode, destNode):
        self.origNode = origNode
        self.destNode = destNode
        self.estimationModel = EstimationModel("time")
        fastestPath, shortestTime = self.__dijkstra()
        return fastestPath, shortestTime


    def __dijkstra(self):
        initialStatus = self.__initDijkstra()
        if initialStatus:
            while not self.__ifFinished():
                if len(self.notPassedNodeDict) == 0:
                    print("No route")
                    return
                else:
                    self.__onePaceUpdateOfDijstra()
            pathWitMinVal = self.__generateMinValPath()
            minVal = self.notPassedNodeDict[self.dummyDestNodeInPathGraph]
            return pathWitMinVal, minVal
        return

    def __initDijkstra(self):
        self.passedNodesSet = set()
        self.notPassedNodeDict = dict()
        self.edgesGdf = edgePreprocessing(self.getEdges(), self.getNodes())
        self.dummyWindow = Window(-1, -1, -1)
        self.dummyOriNodeInPathGraph = NodeInPathGraph(self.dummyWindow, self.origNode, None)
        self.dummyDestNodeInPathGraph = NodeInPathGraph(self.dummyWindow, -1, None)
        edgesGdfFromOrigNode = self.edgesGdf[self.edgesGdf['u'] == self.origNode]
        for origEdgeIdInGdf in list(edgesGdfFromOrigNode.index):
            nextNodeId = edgesGdfFromOrigNode.loc[origEdgeIdInGdf, 'v']
            nextWindow = Window(self.dummyWindow.midSeg, self.dummyWindow.sucSeg, origEdgeIdInGdf)
            self.notPassedNodeDict[NodeInPathGraph(nextWindow, nextNodeId, self.dummyOriNodeInPathGraph)] = 0
        if not len(edgesGdfFromOrigNode):
            print("not path from node:", self.origNode)
            return False
        else:
            return True

    def __ifFinished(self):
        return self.dummyDestNodeInPathGraph in self.passedNodesSet

    def __onePaceUpdateOfDijstra(self):
        curNodeInPathGraph = self.__findMinValNotPassedNode()
        # print(str(curNodeInPathGraph))
        valOfCurNode = self.notPassedNodeDict.pop(curNodeInPathGraph)
        self.passedNodesSet.add(curNodeInPathGraph)
        nextNodeList = curNodeInPathGraph.generateNextNode(self.edgesGdf, self.destNode)
        for nextNodeInPathGraph in nextNodeList:
            self.__updateValOfNextNode(nextNodeInPathGraph, valOfCurNode)

    def __findMinValNotPassedNode(self):
        return min(self.notPassedNodeDict, key=self.notPassedNodeDict.get)

    def __updateValOfNextNode(self, nextNodeInPathGraph, valOfCurNode):
        valOfNextNode = nextNodeInPathGraph.calVal(self.estimationModel, self.edgesGdf)
        if nextNodeInPathGraph not in self.notPassedNodeDict:
            self.notPassedNodeDict[nextNodeInPathGraph] = valOfNextNode + valOfCurNode
        if valOfNextNode + valOfCurNode < self.notPassedNodeDict[nextNodeInPathGraph]:
            _ = self.notPassedNodeDict.pop(nextNodeInPathGraph)
            self.notPassedNodeDict[nextNodeInPathGraph] = valOfNextNode + valOfCurNode

    def __generateMinValPath(self):
        pathWitMinVal = []
        destNodeGenerated = self.__getDestNodeGenerated()
        pathWitMinVal.append(destNodeGenerated.node)
        while destNodeGenerated.prevNode:
            destNodeGenerated = destNodeGenerated.prevNode
            pathWitMinVal.append(destNodeGenerated.node)
        return pathWitMinVal[:2:-1]

    def __getDestNodeGenerated(self):
        for tempNode in self.passedNodesSet:
            if tempNode == self.dummyDestNodeInPathGraph:
                return tempNode


class GraphFromHmlFile(OsmGraph):
    def __init__(self, hmlAddress):
        self.graph = ox.load_graphml(hmlAddress)


class GraphFromBbox(OsmGraph):
    def __init__(self, boundingBox):
        self.graph = ox.graph_from_polygon(boundingBox.polygon(), network_type='drive')


class GraphFromGdfs(OsmGraph):
    def __init__(self, nodes, edges):
        self.graph = ox.utils_graph.graph_from_gdfs(nodes, edges)