from window import Window
import time
from spaitalShape import Point

class NodeInPathGraph:
    def __init__(self, window, node, prevNode, edge):
        self.window = window
        self.node = node
        self.prevNode = prevNode
        self.edge = edge

    def __eq__(self, other):
        return self.window == other.window and self.edge == other.edge

    def __hash__(self):
        return hash((self.window.getTuple(), self.edge))

    def __str__(self):
        return "window:"+str(self.window)+" node:"+str(self.node)

    def calVal(self, estimationModel, edgesGdf):
        if self.window.midSeg == -1:
            return 0
        else:
            numericalFeatures, categoricalFeatures = self.window.extractFeatures(edgesGdf, self.prevNode.window.prevSeg)
            return estimationModel.predict(numericalFeatures, categoricalFeatures)

    def generateNextNode(self, edgesGdf, uIdInEdges, destNode):
        self.curNode = NodeInPathGraph(self.window,self.node,self.prevNode,self.edge)
        self.destNode = destNode
        if self.__isNoSucNode():
            return self.__dummySucNode()
        else:
            return self.__nextNodeFromEdge(edgesGdf, uIdInEdges)

    def __isNoSucNode(self):
        return self.node == -1 or self.node == self.destNode

    def __dummySucNode(self):
        nextNodeId = -1
        nextWindow = Window(self.window.midSeg, self.window.sucSeg, -1)
        nextPoint = Point(0,0)
        nextNodes = NodeInPathGraph(nextWindow, nextNodeId, self.curNode, -1)
        return [nextNodes]

    def __nextNodeFromEdge(self, edgesGdf, uIdInEdges):
        #edgesGdfFromNode = edgesGdf[edgesGdf['u'] == self.node]
        edgesGdfFromNode = edgesGdf.loc[uIdInEdges[self.node]]
        #print(edgesGdfFromNode.iloc[0])
        nextNodesList = []
        for edgeIdInGdf in list(edgesGdfFromNode.index):
            nextNodeId = edgesGdfFromNode.loc[edgeIdInGdf, 'v']
            nextWindow = Window(self.window.midSeg, self.window.sucSeg, edgeIdInGdf)
            nextPoint = Point(0,0)
            nextNodesList.append(NodeInPathGraph(nextWindow, nextNodeId, self.curNode, edgeIdInGdf))
        return nextNodesList
