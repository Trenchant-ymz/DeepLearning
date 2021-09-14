import numpy as np
import math


class Window:
    def __init__(self, minusSeg, prevSeg, midSeg, sucSeg):
        self.minusSeg = minusSeg
        self.prevSeg = prevSeg
        self.midSeg = midSeg
        self.sucSeg = sucSeg

    def __eq__(self, other):
        return self.prevSeg == other.prevSeg \
               and self.midSeg == other.midSeg and self.sucSeg == other.sucSeg

    def __str__(self):
        return str(self.prevSeg)+',' + str(self.midSeg) +',' + str(self.sucSeg)


    def valid(self):
        return self.prevSeg != self.sucSeg

    def extractFeatures(self, edgesDict):
        prevSegNumFeature, prevSegCatFeature = edgeFeature(self.prevSeg, edgesDict, self.minusSeg)
        midSegNumFeature, midSegCatFeature = edgeFeature(self.midSeg, edgesDict, self.prevSeg)
        sucSegNumFeature, sucSegCatFeature = edgeFeature(self.sucSeg, edgesDict, self.midSeg)
        numericalFeatures = [prevSegNumFeature, midSegNumFeature, sucSegNumFeature]
        categoricalFeatures = [prevSegCatFeature, midSegCatFeature, sucSegCatFeature]
        return numericalFeatures, categoricalFeatures


def edgeFeature(segmentIDInGdf, edgesDict, prevEdgeId):
    if segmentIDInGdf == -1:
        return [0] * 6, [0] * 7
    curEdge = edgesDict[segmentIDInGdf]
    catFeature = curEdge['categoricalFeature']
    previousOrientation = calPrevOrientation(edgesDict, curEdge, prevEdgeId)
    numFeature = [curEdge['speedLimit'], curEdge['mass'], curEdge['elevationChange'], previousOrientation,
                  curEdge['lengthNormalized'], curEdge['directionAngle']]
    return numFeature, catFeature


def calPrevOrientation(edgesDict, curEdge, prevEdgeId):
    if prevEdgeId is None or prevEdgeId == -1:
        orientation = 0
    else:
        prevEdge = edgesDict[prevEdgeId]
        a = prevEdge['points'][-2]
        b = curEdge['points'][0]
        c = curEdge['points'][1]
        orientation = ori_cal(a, b, c)
    orientation = (orientation + 1.46016587027665) / 33.3524612794841
    return orientation


def ori_cal(coor_a, coor_b, coor_c):
    """
    Calcuate the orientation change from vector ab to bc (0 in default, right turn > 0, left turn < 0)

    10 degree is the threshold of a turn.

    Args:
        coor_a: coordinate of point a
        coor_b: coordinate of point b
        coor_c: coordinate of point c

    Returns:
        's': straight
        'l': left-hand turn
        'r': right-hand turn
    """
    a = np.array(coor_a)
    b = np.array(coor_b)
    c = np.array(coor_c)
    v_ab = b - a
    v_bc = c - b
    cosangle = v_ab.dot(v_bc) / (np.linalg.norm(v_bc) * np.linalg.norm(v_ab) + 1e-16)
    res = math.acos(cosangle) * 180 / np.pi if np.cross(v_ab, v_bc) < 0 else -math.acos(cosangle) * 180 / np.pi
    return res if not math.isnan(res) else 0
