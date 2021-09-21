import pandas as pd

from window import Window
import time
from spaitalShape import Point
from math import inf


class NodeInPathGraph:
    def __init__(self, window, node, prevNode, edge, shortPathEst=inf):
        self.window = window
        self.node = node
        self.prevNode = prevNode
        self.edge = edge
        self.shortPathEst = shortPathEst

    def __eq__(self, other):
        return self.window == other.window

    # this hack with additional ordering by key is needed to make it work with RBTree as TreeSet data structure
    def __lt__(self, other):
        if self.shortPathEst != other.shortPathEst:
            return self.shortPathEst < other.shortPathEst
        else:
            return str(self.window) < str(other.window)

    def __gt__(self, other):
        if self.shortPathEst != other.shortPathEst:
            return self.shortPathEst > other.shortPathEst
        else:
            return str(self.window) > str(other.window)

    def __hash__(self):
        return hash(str(self.window)+str(self.node))

    def __str__(self):
        return str(self.window) + str(self.node)

    def getStr(self):
        return str(self.window.minusSeg) + ',' + str(self.window) + ',' + str(self.node)

    def valid(self):
        return self.window.valid()


