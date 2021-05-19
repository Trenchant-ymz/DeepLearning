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
        self.nodesGdf, self.edgesGdf = self.graphToGdfs()

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

    def shortestPath(self, origNode, destNode):
        shortestPath = nx.shortest_path(G=self.graph, source=origNode, target=destNode, weight='length')
        return shortestPath

    def ecoPath(self, origNode, destNode):
        self.origNode = origNode
        self.destNode = destNode
        self.estimationModel = EstimationModel("fuel")
        ecoPath, ecoEnergy, ecoEdgePath = self.__dijkstra()
        return ecoPath, ecoEnergy, ecoEdgePath

    def fastestPath(self, origNode, destNode):
        self.origNode = origNode
        self.destNode = destNode
        self.estimationModel = EstimationModel("time")
        fastestPath, shortestTime, fastestEdgePath = self.__dijkstra()
        return fastestPath, shortestTime, fastestEdgePath


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
            edgePathWithMinVal = self.__generateMinValEdgePath()
            return pathWitMinVal, self.minVal, edgePathWithMinVal[-2:2:-1]
        return

    def __initDijkstra(self):
        self.passedNodesSet = set()
        self.notPassedNodeDict = dict()
        self.edgesGdf = self.getEdges()
        self.dummyWindow = Window(-1, -1, -1)
        self.dummyOriNodeInPathGraph = NodeInPathGraph(self.dummyWindow, self.origNode, None,-1)
        self.dummyDestNodeInPathGraph = NodeInPathGraph(self.dummyWindow, -1, None,-1)
        edgesGdfFromOrigNode = self.edgesGdf[self.edgesGdf['u'] == self.origNode]
        for origEdgeIdInGdf in list(edgesGdfFromOrigNode.index):
            nextNodeId = edgesGdfFromOrigNode.loc[origEdgeIdInGdf, 'v']
            nextWindow = Window(self.dummyWindow.midSeg, self.dummyWindow.sucSeg, origEdgeIdInGdf)
            self.notPassedNodeDict[NodeInPathGraph(nextWindow, nextNodeId, self.dummyOriNodeInPathGraph, origEdgeIdInGdf)] = 0
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
        if self.__isDestNode(curNodeInPathGraph):
            self.minVal = valOfCurNode
            self.destNodeGenerated = curNodeInPathGraph
            return
        nextNodeList = curNodeInPathGraph.generateNextNode(self.edgesGdf, self.destNode)
        for nextNodeInPathGraph in nextNodeList:
            self.__updateValOfNextNode(nextNodeInPathGraph, valOfCurNode)

    def __findMinValNotPassedNode(self):
        return min(self.notPassedNodeDict, key=self.notPassedNodeDict.get)

    def __isDestNode(self, curNodeInPathGraph):
        return curNodeInPathGraph == self.dummyDestNodeInPathGraph

    def __updateValOfNextNode(self, nextNodeInPathGraph, valOfCurNode):
        if nextNodeInPathGraph not in self.passedNodesSet:
            valOfNextNode = nextNodeInPathGraph.calVal(self.estimationModel, self.edgesGdf)
            if nextNodeInPathGraph not in self.notPassedNodeDict:
                self.notPassedNodeDict[nextNodeInPathGraph] = valOfNextNode + valOfCurNode
            if valOfNextNode + valOfCurNode < self.notPassedNodeDict[nextNodeInPathGraph]:
                _ = self.notPassedNodeDict.pop(nextNodeInPathGraph)
                self.notPassedNodeDict[nextNodeInPathGraph] = valOfNextNode + valOfCurNode

    def __generateMinValPath(self):
        dNode = self.destNodeGenerated
        pathWitMinVal = [dNode.node]
        while dNode.prevNode:
            dNode = dNode.prevNode
            pathWitMinVal.append(dNode.node)
        return pathWitMinVal[:2:-1]

    def __generateMinValEdgePath(self):
        estimationModel = EstimationModel('fuel')
        dNode = self.destNodeGenerated
        val = 0
        edgePathWithMinVal = [dNode.edge]
        val += dNode.calVal(estimationModel, self.edgesGdf)
        while dNode.prevNode:
            dNode = dNode.prevNode

            addval = dNode.calVal(estimationModel, self.edgesGdf)
            val += addval
            edgePathWithMinVal.append(dNode.edge)
        #return edgePathWithMinVal[:2:-1]
        return edgePathWithMinVal

    def totalLength(self, path):
        length = 0
        for i in path:
            length += self.edgesGdf.loc[i, 'length']
        return length

    '''
    def totalLength(self, path):
        length = 0
        for i, OdPair in enumerate(zip(path[:-1], path[1:])):
            segmentId = self.edgesGdf[self.edgesGdf['odPair'] == OdPair].index[0]
            length += self.edgesGdf.loc[segmentId, 'length']
        return length
    '''


    def totalEnergy(self, path):
        return self.__calculateValue(path, "fuel")

    def totalTime(self, path):
        return self.__calculateValue(path, "time")


    def __calculateValue(self, path, estimationType):
        estimationModel = EstimationModel(estimationType)
        value = 0
        firstSeg = path[0]
        window = Window(-1, -1, firstSeg)
        prevWindowSeg = -1
        for i in range(len(path)):
            prevWindowSeg = window.prevSeg
            window.prevSeg = window.midSeg
            window.midSeg = window.sucSeg
            if i < len(path)-1:
                window.sucSeg = path[i+1]
            else:
                window.sucSeg = -1

            numericalFeatures, categoricalFeatures = window.extractFeatures(self.edgesGdf, prevWindowSeg)
            addValue = estimationModel.predict(numericalFeatures, categoricalFeatures)
            value += addValue
            #prevWindowSeg = tempPrevSeg
        return value

    '''
    def __calculateValue(self, path, estimationType):
        estimationModel = EstimationModel(estimationType)
        value = 0
        firstSeg = self.__findSegId(path, 0)
        window = Window(-1, -1, firstSeg)
        prevWindowSeg = -1
        for i in range(len(path) - 1):
            window, tempPrevSeg = self.__updateWindow(window, path, i)
            numericalFeatures, categoricalFeatures = window.extractFeatures(self.edgesGdf, prevWindowSeg)
            addValue = estimationModel.predict(numericalFeatures, categoricalFeatures)
            value += addValue
            prevWindowSeg = tempPrevSeg
        return value
    
    '''


    def __findSegId(self, path, i):
        OdPair = (path[i], path[i+1])
        segId = self.edgesGdf[self.edgesGdf['odPair'] == OdPair].index[0]
        return segId

    def __updateWindow(self, window, path, i):
        tempPrevSeg = window.prevSeg
        window.prevSeg = window.midSeg
        window.midSeg = window.sucSeg
        if i < len(path) - 2:
            nextSeg = self.__findSegId(path, i+1)
            window.sucSeg = nextSeg
        else:
            window.sucSeg = -1
        return window, tempPrevSeg


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