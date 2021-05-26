from spaitalShape import Point, OdPair, Box
import routingFunctions as rf


class LocationRequest:
    def __init__(self):
        # Murphy Depot
        origin = Point(-93.2219, 44.979)
        # test
        #origin = Point(-93.4254, 44.7888)
        #origin = Point(-93.2466, 44.8959)
        # Shakopee East (Depot):
        destination = Point(-93.4620, 44.7903)
        #destination = Point(-93.4495, 44.8611)

        self.odPair = OdPair(origin, destination)
        self.boundingBox = Box(-93.4975, -93.1850, 44.7458, 45.0045)


def main():
    locationRequest = LocationRequest()
    graphWithElevation, edges = rf.extractGraphAndPreprocessing(locationRequest)
    shortestNodePath = rf.findShortestPath(graphWithElevation, locationRequest.odPair)
    ecoRoute, energyOnEcoRoute, ecoEdgePath = rf.findEcoPathAndCalEnergy(graphWithElevation, locationRequest.odPair)
    fastestPath, shortestTime, fastestEdgePath = rf.findFastestPathAndCalTime(graphWithElevation, locationRequest.odPair)
    graphWithElevation.plotPathList([shortestNodePath, ecoRoute, fastestPath], 'routing result.pdf')


if __name__ == '__main__':
    main()


