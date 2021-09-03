import networkx as nx
import osmnx as ox
import os
from edgeGdfPreprocessing import edgePreprocessing
from estimationModel import EstimationModel
from window import Window
from pathGraph import NodeInPathGraph
import time
from collections import defaultdict
import math

class Dijkstra:
    def __init__(self, edgesGdf, uIdInEdges, origNode, destNode, estimationModel):
        self.passedNodesSet = set()
        self.notPassedNodeDict = dict()
        self.edgesGdf = edgesGdf
        self.dummyWindow = Window(-1, -1, -1)
        self.origNode = origNode
        self.destNode = destNode
        self.uIdInEdges = uIdInEdges
        self.estimationModel = estimationModel
        self.dummyOriNodeInPathGraph = NodeInPathGraph(self.dummyWindow, self.origNode, None, -1)
        self.dummyDestNodeInPathGraph = NodeInPathGraph(self.dummyWindow, -1, None, -1)
        edgesGdfFromOrigNode = self.edgesGdf[self.edgesGdf['u'] == self.origNode]
        self.initializeDict(edgesGdfFromOrigNode)
        if not len(edgesGdfFromOrigNode):
            print("not path from node:", self.origNode)
            self.__initializedStatus = False
        else:
            self.__initializedStatus = True

    def initializeDict(self, edgesGdfFromOrigNode):
        for origEdgeIdInGdf in list(edgesGdfFromOrigNode.index):
            nextNodeId = edgesGdfFromOrigNode.loc[origEdgeIdInGdf, 'v']
            nextWindow = Window(self.dummyWindow.midSeg, self.dummyWindow.sucSeg, origEdgeIdInGdf)
            self.notPassedNodeDict[
                NodeInPathGraph(nextWindow, nextNodeId, self.dummyOriNodeInPathGraph, origEdgeIdInGdf)] = 0

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
        return len(self.notPassedNodeDict) == 0

    def onePaceUpdate(self):
        #t0 = time.time()
        curNodeInPathGraph = self.findMinValNotPassedNode()
        #t1 = time.time()
        # print(str(curNodeInPathGraph))
        valOfCurNode = self.notPassedNodeDict.pop(curNodeInPathGraph)
        #t2 = time.time()
        self.passedNodesSet.add(curNodeInPathGraph)
        #t3 = time.time()
        if self.isDestNode(curNodeInPathGraph):
            self.minVal = valOfCurNode
            self.destNodeGenerated = curNodeInPathGraph
            return
        #t6 = time.time()
        nextNodeList = curNodeInPathGraph.generateNextNode(self.edgesGdf, self.uIdInEdges, self.destNode)
        #t4 = time.time()
        for nextNodeInPathGraph in nextNodeList:
            self.updateValOfNextNode(nextNodeInPathGraph, valOfCurNode)
        #t5 = time.time()
        #print(t5-t0, t1-t0, t2-t1, t3-t2, t6-t3, t4-t6, t5-t4)

    def findMinValNotPassedNode(self):
        return min(self.notPassedNodeDict, key=self.notPassedNodeDict.get)

    def isDestNode(self, curNodeInPathGraph):
        return curNodeInPathGraph == self.dummyDestNodeInPathGraph

    def updateValOfNextNode(self, nextNodeInPathGraph, valOfCurNode):
        if nextNodeInPathGraph not in self.passedNodesSet:
            valOfNextNode = nextNodeInPathGraph.calVal(self.estimationModel, self.edgesGdf)
            if nextNodeInPathGraph not in self.notPassedNodeDict:
                self.notPassedNodeDict[nextNodeInPathGraph] = valOfNextNode + valOfCurNode
            if valOfNextNode + valOfCurNode < self.notPassedNodeDict[nextNodeInPathGraph]:
                _ = self.notPassedNodeDict.pop(nextNodeInPathGraph)
                self.notPassedNodeDict[nextNodeInPathGraph] = valOfNextNode + valOfCurNode

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

    def __init__(self, edgesGdf, uIdInEdges, origNode, destNode, estimationModel, localRequest, nodes):
        self.gValues = defaultdict(float)
        self.hValues = defaultdict(float)
        self.localRequest = localRequest
        self.nodes = nodes
        super().__init__(edgesGdf, uIdInEdges, origNode, destNode, estimationModel)


    def initializeDict(self, edgesGdfFromOrigNode):
        for origEdgeIdInGdf in list(edgesGdfFromOrigNode.index):
            nextNodeId = edgesGdfFromOrigNode.loc[origEdgeIdInGdf, 'v']
            nextWindow = Window(self.dummyWindow.midSeg, self.dummyWindow.sucSeg, origEdgeIdInGdf)
            nextNodeInPathGraph = NodeInPathGraph(nextWindow, nextNodeId, self.dummyOriNodeInPathGraph, origEdgeIdInGdf)
            hValOfNextNode = self.calH(nextNodeInPathGraph)
            self.gValues[nextNodeInPathGraph] = 0
            self.hValues[nextNodeInPathGraph] = hValOfNextNode
            self.notPassedNodeDict[nextNodeInPathGraph] = hValOfNextNode

    def onePaceUpdate(self):
        #t0 = time.time()
        curNodeInPathGraph = self.findMinValNotPassedNode()
        #t1 = time.time()
        # print(str(curNodeInPathGraph))
        valOfCurNode = self.notPassedNodeDict.pop(curNodeInPathGraph)
        gValOfCurNode = self.gValues.pop(curNodeInPathGraph)
        _ = self.hValues.pop(curNodeInPathGraph)
        #t2 = time.time()
        self.passedNodesSet.add(curNodeInPathGraph)
        #t3 = time.time()
        if self.isDestNode(curNodeInPathGraph):
            self.minVal = gValOfCurNode
            self.destNodeGenerated = curNodeInPathGraph
            return
        #t6 = time.time()
        nextNodeList = curNodeInPathGraph.generateNextNode(self.edgesGdf, self.uIdInEdges, self.destNode)
        #t4 = time.time()
        for nextNodeInPathGraph in nextNodeList:
            self.updateValOfNextNode(nextNodeInPathGraph, gValOfCurNode)
        #t5 = time.time()
        #print(t5-t0, t1-t0, t2-t1, t3-t2, t6-t3, t4-t6, t5-t4)

    def updateValOfNextNode(self, nextNodeInPathGraph, gValOfCurNode):
        if nextNodeInPathGraph not in self.passedNodesSet:
            gValOfNextNode = gValOfCurNode + nextNodeInPathGraph.calVal(self.estimationModel, self.edgesGdf)
            if nextNodeInPathGraph not in self.notPassedNodeDict:
                hValOfNextNode = self.calH(nextNodeInPathGraph)
                self.notPassedNodeDict[nextNodeInPathGraph] = gValOfNextNode + hValOfNextNode
                self.gValues[nextNodeInPathGraph] = gValOfNextNode
                self.hValues[nextNodeInPathGraph] = hValOfNextNode
            else:
                gValBefore = self.gValues.get(nextNodeInPathGraph)
                if gValOfNextNode < gValBefore:
                    hValOfNextNode = self.hValues.get(nextNodeInPathGraph)
                    _ = self.notPassedNodeDict.pop(nextNodeInPathGraph)
                    _ = self.gValues.pop(nextNodeInPathGraph)
                    self.notPassedNodeDict[nextNodeInPathGraph] = gValOfNextNode + hValOfNextNode
                    self.gValues[nextNodeInPathGraph] = gValOfNextNode
                    #self.hValues[nextNodeInPathGraph] = hValOfNextNode

    def calH(self, curNode):
        if curNode.node == -1:
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
        v = 5/3.6
        curPoint = (self.nodes.loc[curNode.node, 'x'], self.nodes.loc[curNode.node, 'y'])
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