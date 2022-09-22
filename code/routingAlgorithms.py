import networkx as nx
import osmnx as ox
import os
from edgeGdfPreprocessing import edgePreprocessing
from estimationModel import EstimationModel
from window import Window
from windowNode import NodeInPathGraph
import time
from collections import defaultdict
import math
import pandas as pd
import heapq
from bintrees import RBTree
from math import inf

class Dijkstra:
    def __init__(self, edgesDict, uToV , origNode, destNode, estimationModel):
        #self.pointList = []
        self.passedNodesSet = set()
        self.notPassedNodeDict = dict()
        #self.notPassedNodeHeapq = []
        #self.notPassedNodeDict = dict()
        self.edgesDict = edgesDict
        self.uToV = uToV
        self.dummyWindow = Window(-1, -1, -1, -1)
        self.origNode = origNode
        self.destNode = destNode
        self.estimationModel = estimationModel
        self.dummyOriNodeInPathGraph = self.generateDummyOirNode()
        self.notPassedNodeDict[self.dummyOriNodeInPathGraph] = 0
        self.dummyDestNodeInPathGraph = self.generateDummyDestNode()
        self.notPassedNodeQ = RBTree()
        self.notPassedNodeQ.insert(self.dummyOriNodeInPathGraph, 0)
        listOfNodesFromOrig = self.uToV[self.origNode]
        self.initializeQ(listOfNodesFromOrig)
        if not len(listOfNodesFromOrig):
            print("not path from node:", self.origNode)
            self.__initializedStatus = False
        else:
            self.__initializedStatus = True

    def generateDummyOirNode(self):
        return NodeInPathGraph(self.dummyWindow, self.origNode, None, -1, 0)

    def generateDummyDestNode(self):
        return NodeInPathGraph(self.dummyWindow, -1, None, -1, inf)

    def initializeQ(self, listOfNodesFromOrig):
        # print(edgesGdfFromNode.iloc[0])
        for edgeIdAndV in listOfNodesFromOrig:
            edgeIdInGdf = edgeIdAndV[0]
            nextNodeId = edgeIdAndV[1]
            nextWindow = Window(self.dummyWindow.prevSeg, self.dummyWindow.midSeg, self.dummyWindow.sucSeg, edgeIdInGdf)
            nextNode = NodeInPathGraph(nextWindow, nextNodeId, self.dummyOriNodeInPathGraph, edgeIdInGdf, 0)
            self.notPassedNodeQ.insert(nextNode, 0)
            self.notPassedNodeDict[nextNode] = 0
        return

    def routing(self):
        steps = 0
        if self.checkIniStatus():
            while not self.ifFinished():
                if self.noNodeToExplore():
                    print("No route")
                    return
                else:
                    steps += 1
                    self.onePaceUpdate()

            pathWitMinVal = self.generateMinValNodePath()
            edgePathWithMinVal = self.generateMinValEdgePath()
            print("num of steps:", steps)
            return pathWitMinVal, self.minVal, edgePathWithMinVal
        return

    def checkIniStatus(self):
        return self.__initializedStatus

    def ifFinished(self):
        return self.dummyDestNodeInPathGraph in self.passedNodesSet

    def noNodeToExplore(self):
        return self.notPassedNodeQ.is_empty()

    def onePaceUpdate(self):
        curNodeInPathGraph, valOfCurNode = self.notPassedNodeQ.min_item()
        self.notPassedNodeQ.remove(curNodeInPathGraph)
        _ = self.notPassedNodeDict.pop(curNodeInPathGraph)
        self.passedNodesSet.add(curNodeInPathGraph)
        if self.isDestNode(curNodeInPathGraph):
            self.minVal = valOfCurNode
            self.destNodeGenerated = curNodeInPathGraph
            return
        self.exploreNextNode(curNodeInPathGraph, valOfCurNode)

    def isDestNode(self, curNodeInPathGraph):
        return curNodeInPathGraph == self.dummyDestNodeInPathGraph

    def exploreNextNode(self, curNodeInPathGraph, valOfCurNode):
        if self.isNoSucNode(curNodeInPathGraph):
            self.exploreDummySucNode(curNodeInPathGraph,valOfCurNode)
        else:
            self.exploreNextNodeFromEdge(curNodeInPathGraph, valOfCurNode)
        return

    def isNoSucNode(self, curNodeInPathGraph):
        return curNodeInPathGraph.node == -1 or curNodeInPathGraph.node == self.destNode

    def exploreDummySucNode(self, curNodeInPathGraph, valOfCurNode):
        nextNodeId = -1
        nextWindow = Window(curNodeInPathGraph.window.prevSeg, curNodeInPathGraph.window.midSeg, curNodeInPathGraph.window.sucSeg, -1)
        nextNodes = NodeInPathGraph(nextWindow, nextNodeId, curNodeInPathGraph, -1)
        valOfNextNode = self.calVal(nextNodes)
        self.updateQ(nextNodes, valOfCurNode+valOfNextNode)
        return

    def exploreNextNodeFromEdge(self, curNodeInPathGraph, valOfCurNode):
        listOfNodes = self.uToV[curNodeInPathGraph.node]
        for edgeIdAndV in listOfNodes:
            edgeIdInGdf = edgeIdAndV[0]
            nextNodeId = edgeIdAndV[1]
            nextWindow = Window(curNodeInPathGraph.window.prevSeg, curNodeInPathGraph.window.midSeg, curNodeInPathGraph.window.sucSeg, edgeIdInGdf)
            nextNodeInPathGraph = NodeInPathGraph(nextWindow, nextNodeId, curNodeInPathGraph, edgeIdInGdf)
            if nextWindow.valid() and nextNodeInPathGraph not in self.passedNodesSet:
                valOfNextNode = self.calVal(nextNodeInPathGraph)
                self.updateQ(nextNodeInPathGraph, valOfNextNode+valOfCurNode)

    def updateQ(self, nextNodeInPathGraph, curShortPathEst):
        if nextNodeInPathGraph not in self.notPassedNodeDict:
            nextNodeInPathGraph.shortPathEst = curShortPathEst
            self.notPassedNodeQ.insert(nextNodeInPathGraph, nextNodeInPathGraph.shortPathEst)
            self.notPassedNodeDict[nextNodeInPathGraph] = curShortPathEst
        else:
            nextNodeInPathGraph.shortPathEst = self.notPassedNodeDict[nextNodeInPathGraph]
            if curShortPathEst < nextNodeInPathGraph.shortPathEst:
                self.notPassedNodeQ.remove(nextNodeInPathGraph)
                nextNodeInPathGraph.shortPathEst = curShortPathEst
                self.notPassedNodeQ.insert(nextNodeInPathGraph, nextNodeInPathGraph.shortPathEst)
                self.notPassedNodeDict[nextNodeInPathGraph] = curShortPathEst

    def calVal(self, curNodeInPathGraph):
        if curNodeInPathGraph.window.midSeg == -1:
            return 0
        else:
            numericalFeatures, categoricalFeatures = curNodeInPathGraph.window.extractFeatures(self.edgesDict)
            return self.estimationModel.predictFromData(numericalFeatures, categoricalFeatures)

    def generateMinValNodePath(self):
        dNode = self.destNodeGenerated
        pathWitMinVal = [dNode.node]
        while dNode.prevNode:
            dNode = dNode.prevNode
            pathWitMinVal.append(dNode.node)
        return pathWitMinVal[:2:-1]

    def generateMinValEdgePath(self):
        dNode = self.destNodeGenerated
        edgePathWithMinVal = [dNode.edge]
        while dNode.prevNode:
            dNode = dNode.prevNode
            edgePathWithMinVal.append(dNode.edge)
        return edgePathWithMinVal[-2:2:-1]


class AStar(Dijkstra):

    def __init__(self, edgesDict, uToV , origNode, destNode, estimationModel, localRequest, nodes):
        self.hValues = defaultdict(float)
        #self.pointList = []
        self.localRequest = localRequest
        self.nodes = nodes.to_dict('index')
        super().__init__(edgesDict,uToV , origNode, destNode, estimationModel)
        self.hValues[self.dummyOriNodeInPathGraph] = 0
        self.notPassedNodeDict[self.dummyOriNodeInPathGraph] = 0
        #print(self.notPassedNodeQ)


    def initializeQ(self, listOfNodesFromOrig):
        # print(edgesDict.iloc[0])
        for edgeIdAndV in listOfNodesFromOrig:
            edgeIdInGdf = edgeIdAndV[0]
            nextNodeId = edgeIdAndV[1]
            nextWindow = Window(self.dummyWindow.prevSeg, self.dummyWindow.midSeg, self.dummyWindow.sucSeg, edgeIdInGdf)
            hValOfNextNode = self.calH(nextNodeId)
            nextNodeInPathGraph = NodeInPathGraph(nextWindow, nextNodeId, self.dummyOriNodeInPathGraph, edgeIdInGdf, hValOfNextNode)
            self.notPassedNodeQ.insert(nextNodeInPathGraph, hValOfNextNode)
            self.hValues[nextNodeInPathGraph] = hValOfNextNode
            #print(edgeIdAndV, self.hValues)
            self.notPassedNodeDict[nextNodeInPathGraph] = hValOfNextNode
        #print(self.notPassedNodeQ)
        #print(self.hValues)
        #print(self.notPassedNodeDict)
        return


    def onePaceUpdate(self):
        curNodeInPathGraph, valOfCurNode = self.notPassedNodeQ.min_item()
        #print(curNodeInPathGraph, valOfCurNode)
        #self.pointList.append((curNodeInPathGraph.getStr(), valOfCurNode))
        hvalue = self.hValues[curNodeInPathGraph]
        self.notPassedNodeQ.remove(curNodeInPathGraph)
        _ = self.notPassedNodeDict.pop(curNodeInPathGraph)
        self.passedNodesSet.add(curNodeInPathGraph)
        curGVal = valOfCurNode - hvalue
        if self.isDestNode(curNodeInPathGraph):
            self.minVal = curGVal
            self.destNodeGenerated = curNodeInPathGraph
            return
        self.exploreNextNode(curNodeInPathGraph,curGVal)


    def exploreNextNodeFromEdge(self, curNodeInPathGraph,curGVal):
        #print("curbest", valOfCurNode, hValOfCurNode)
        listOfNodes = self.uToV[curNodeInPathGraph.node]
        for edgeIdAndV in listOfNodes:
            edgeIdInGdf = edgeIdAndV[0]
            nextNodeId = edgeIdAndV[1]
            nextWindow = Window(curNodeInPathGraph.window.prevSeg, curNodeInPathGraph.window.midSeg, curNodeInPathGraph.window.sucSeg, edgeIdInGdf)
            nextNodeInPathGraph = NodeInPathGraph(nextWindow, nextNodeId, curNodeInPathGraph, edgeIdInGdf)
            if nextWindow.valid() and nextNodeInPathGraph not in self.passedNodesSet:
                if nextNodeInPathGraph in self.hValues:
                    hValOfNextNode = self.hValues[nextNodeInPathGraph]
                else:
                    hValOfNextNode = self.calH(nextNodeId)
                    self.hValues[nextNodeInPathGraph] = hValOfNextNode
                #valOfNextNode = self.calVal(nextNodeInPathGraph)
                valOfNextNode = self.calVal(nextNodeInPathGraph)
                #self.pointList.append((nextNodeInPathGraph.getStr(), valOfNextNode, valOfCurNode, hValOfCurNode))
                self.updateQInAStar(nextNodeInPathGraph, valOfNextNode+curGVal, hValOfNextNode)


    def updateQInAStar(self, nextNodeInPathGraph, curGValEst, hValOfNextNode):
        curFVal = curGValEst + hValOfNextNode
        #print('explored', curFVal)
        if nextNodeInPathGraph not in self.notPassedNodeDict:
            nextNodeInPathGraph.shortPathEst = curFVal
            self.notPassedNodeQ.insert(nextNodeInPathGraph, curFVal)
            self.notPassedNodeDict[nextNodeInPathGraph] = curFVal
        else:
            nextNodeInPathGraph.shortPathEst = self.notPassedNodeDict[nextNodeInPathGraph]
            if curFVal < nextNodeInPathGraph.shortPathEst:
                self.notPassedNodeQ.remove(nextNodeInPathGraph)
                nextNodeInPathGraph.shortPathEst = curFVal
                self.notPassedNodeQ.insert(nextNodeInPathGraph, curFVal)
                self.notPassedNodeDict[nextNodeInPathGraph] = curFVal


    def calH(self, node):
        if node == -1:
            return 0
        crr = 0.0067
        # m/s^2
        g = 9.81
        # m^2
        A = 10.5
        cd = 0.5
        # kg/m^3
        r = 1.225
        # m/s
        v = 20/3.6
        curPoint = (self.nodes[node]['x'], self.nodes[node]['x'])
        #curPoint = (curLine['x'], curLine['x'])
        #print("point", curPoint)
        dis, _ = self.pos2dist(curPoint, self.localRequest.destination.xy())
        H = (self.localRequest.mass*crr*g + 0.5*A*cd*r*v*v) * dis * 3.785/40.3/3.6e6
        return H

    def pos2dist(self, point1, point2):

        """

        :param point1: lat lon of origin point [lat lon]
        :param point2: lat lon of destination point [lat lon]
        :return:
           d1: distance calculated by Haversine formula
           d2: distance calculated based on Pythagorean theorem
        """

        radius = 6371  # Earth radius
        lat1 = point1[1] * math.pi / 180
        lat2 = point2[1] * math.pi / 180
        lon1 = point1[0] * math.pi / 180
        lon2 = point2[0] * math.pi / 180
        deltaLat = lat2 - lat1
        deltaLon = lon2 - lon1
        a = math.sin((deltaLat) / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(deltaLon / 2) ** 2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        x = deltaLon * math.cos((lat1 + lat2) / 2)
        y = deltaLat
        d1 = radius * c * 1000  # Haversine distance
        d2 = radius * math.sqrt(x * x + y * y) * 1000 # Pythagorean distance
        return d1, d2


class DijkstraFromLUTable(Dijkstra):

    def __init__(self, edgesDict, uToV , origNode, destNode, estimationModel, lookUpTable):
        super().__init__(edgesDict, uToV, origNode, destNode, estimationModel)
        self.lookUpTable = lookUpTable

    def calVal(self, curNodeInPathGraph):
        if curNodeInPathGraph.window.midSeg == -1:
            return 0
        else:
            return self.lookUpTable.extractValue(curNodeInPathGraph.window)


class AStarFromLUTable(AStar):

    def __init__(self, edgesDict, uToV , origNode, destNode, estimationModel, localRequest, nodes, lookUpTable):
        super().__init__(edgesDict, uToV , origNode, destNode, estimationModel, localRequest, nodes)
        self.lookUpTable = lookUpTable

    def calVal(self, curNodeInPathGraph):
        if curNodeInPathGraph.window.midSeg == -1:
            return 0
        else:
            return self.lookUpTable.extractValue(curNodeInPathGraph.window)