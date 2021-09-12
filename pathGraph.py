import pandas as pd

from window import Window
import time
from spaitalShape import Point
from math import inf


class NodeInPathGraph:
    def __init__(self, window, node, prevNode, edge, shortPathEst = inf):
        self.window = window
        self.node = node
        self.prevNode = prevNode
        self.edge = edge
        self.shortPathEst = shortPathEst

    def __eq__(self, other):
        return self.window == other.window

    def __hash__(self):
        return hash(self.window.getTuple())

    def __str__(self):
        return "window:"+str(self.window)+" node:"+str(self.node)

    def valid(self):
        return self.window.valid()

    def calVal(self, estimationModel, edgesGdf):
        if self.window.midSeg == -1:
            return 0
        else:
            numericalFeatures, categoricalFeatures = self.window.extractFeatures(edgesGdf)
            return estimationModel.predict(numericalFeatures, categoricalFeatures)

    def generateNextNode(self, uToV, destNode):
        self.destNode = destNode
        if self.__isNoSucNode():
            return self.__dummySucNode()
        else:
            return self.__nextNodeFromEdge(uToV)

    def __isNoSucNode(self):
        return self.node == -1 or self.node == self.destNode

    def __dummySucNode(self):
        nextNodeId = -1
        nextWindow = Window(self.window.prevSeg, self.window.midSeg, self.window.sucSeg, -1)
        nextNodes = NodeInPathGraph(nextWindow, nextNodeId, self, -1)
        return [nextNodes]

    def __nextNodeFromEdge(self, uToV):
        listOfNodes = uToV[self.node]
        nextNodesList = []
        for edgeIdAndV in listOfNodes:
            edgeIdInGdf = edgeIdAndV[0]
            nextNodeId = edgeIdAndV[1]

            nextWindow = Window(self.window.prevSeg, self.window.midSeg, self.window.sucSeg, edgeIdInGdf)
            if nextWindow.valid():
                nextNodesList.append(NodeInPathGraph(nextWindow, nextNodeId, self, edgeIdInGdf))

        return nextNodesList


class NodeInAStarGraph(NodeInPathGraph):
    def __init__(self, window, node, prevNode, edge, gVal = inf, hVal = inf ):
        self.gVal = gVal
        self.hVal = hVal
        super().__init__(window, node, prevNode, edge,  self.gVal+self.hVal)