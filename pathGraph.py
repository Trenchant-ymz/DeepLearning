from window import Window


class NodeInPathGraph:
    def __init__(self, window, node, prevNode):
        self.window = window
        self.node = node
        self.prevNode = prevNode

    def __eq__(self, other):
        return self.window == other.window

    def __hash__(self):
        return hash(self.window.getTuple())

    def __str__(self):
        return "window:"+str(self.window)+" node:"+str(self.node)

    def calVal(self, estimationModel, edgesGdf):
        if self.window.midSeg == -1:
            return 0
        else:
            numericalFeatures, categoricalFeatures = self.window.extractFeatures(edgesGdf, self.prevNode.window.prevSeg)
            return estimationModel.predict(numericalFeatures, categoricalFeatures)

    def generateNextNode(self, edgesGdf, destNode):
        self.curNode = NodeInPathGraph(self.window,self.node,self.prevNode)
        self.destNode = destNode
        if self.__isNoSucNode():
            return self.__dummySucNode()
        else:
            return self.__nextNodeFromEdge(edgesGdf)

    def __isNoSucNode(self):
        return self.node == -1 or self.node == self.destNode

    def __dummySucNode(self):
        nextNodeId = -1
        nextWindow = Window(self.window.midSeg, self.window.sucSeg, -1)
        nextNodes = NodeInPathGraph(nextWindow, nextNodeId, self.curNode)
        return [nextNodes]

    def __nextNodeFromEdge(self, edgesGdf):
        edgesGdfFromNode = edgesGdf[edgesGdf['u'] == self.node]
        nextNodesList = []
        for edgeIdInGdf in list(edgesGdfFromNode.index):
            nextNodeId = edgesGdfFromNode.loc[edgeIdInGdf, 'v']
            nextWindow = Window(self.window.midSeg, self.window.sucSeg, edgeIdInGdf)
            nextNodesList.append(NodeInPathGraph(nextWindow, nextNodeId, self.curNode))
        return nextNodesList
