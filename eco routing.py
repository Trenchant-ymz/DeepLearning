import numpy as np
import os
from osmgraph import GraphFromBbox, GraphFromHmlFile, GraphFromGdfs
from spaitalShape import Point, OdPair, Box

class LocationRequest:
    def __init__(self):
        # Murphy Depot
        #origin = Point(-93.2219, 44.979)
        origin = Point(-93.2466, 44.8959)
        # Shakopee East (Depot):
        #destination = Point(-93.4620, 44.7903)
        destination = Point(-93.4495, 44.8611)

        self.odPair = OdPair(origin, destination)
        self.boundingBox = Box(-93.4975, -93.1850, 44.7458, 45.0045)


def main():
    locationRequest = LocationRequest()
    osmGraphInBbox = extractGraphOf(locationRequest.boundingBox)
    graphInMurphy = extractGraphInMurphy(osmGraphInBbox)
    graphInMurphy.removeIsolateNodes()
    shortestPath = findShortestPath(graphInMurphy, locationRequest.odPair)
    ecoRoute, energyOnEcoRoute = findEcoPathAndCalEnergy(graphInMurphy, locationRequest.odPair)
    fastestPath, shortestTime = findFastestPathAndCalTime(graphInMurphy, locationRequest.odPair)
    print(len(fastestPath))


def extractGraphOf(boundingBox):
    folderOfGraph = r'GraphDataInBbox'
    if os.path.exists(folderOfGraph):
        print("reloading graph..")
        osmGraph = GraphFromHmlFile(os.path.join(folderOfGraph, 'osmGraph.graphml'))
    else:
        print("downloading graph..")
        osmGraph = GraphFromBbox(boundingBox)
        osmGraph.saveHmlTo(folderOfGraph)
    # fig, ax = ox.plot_graph(osmGraph.graph)
    return osmGraph


def extractGraphInMurphy(osmGraph):
    nodes, edges = osmGraph.graphToGdfs()
    edgesInMurphy = extractEdgesInMurphy(edges)
    graphInMurphy = GraphFromGdfs(nodes, edgesInMurphy)
    return graphInMurphy


def extractEdgesInMurphy(edges):
    edges['uvPair'] = edges.apply(lambda x: (x.u, x.v), axis=1)
    segmentElevationChange = np.load('segmentElevationChange.npy', allow_pickle=True).item()
    edges['isInMurphy'] = edges.uvPair.apply(lambda x: x in segmentElevationChange)
    return edges[edges.isInMurphy]


def findShortestPath(osmGraph, odPair):
    origNode, targetNode = osmGraph.getODNodesFromODPair(odPair)
    shortestPath = osmGraph.shortestPath(origNode,targetNode)
    print(shortestPath)
    # ox.plot_graph(osmGraph)
    osmGraph.plotPath(shortestPath)
    return shortestPath


def findEcoPathAndCalEnergy(osmGraph, odPair):
    origNode, targetNode = osmGraph.getODNodesFromODPair(odPair)
    ecoPath, ecoEnergy = osmGraph.ecoPath(origNode,targetNode)
    print("ecoPath:", ecoPath, "ecoEnergy:", ecoEnergy)
    osmGraph.plotPath(ecoPath)
    return ecoPath, ecoEnergy


def findFastestPathAndCalTime(osmGraph, odPair):
    origNode, targetNode = osmGraph.getODNodesFromODPair(odPair)
    fastestPath, shortestTime = osmGraph.fastestPath(origNode,targetNode)
    print("fastestPath:", fastestPath, "shortestTime:", shortestTime)
    osmGraph.plotPath(fastestPath)
    return fastestPath, shortestTime


if __name__ == '__main__':
    main()


