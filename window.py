import numpy as np
import math


class Window:
    def __init__(self, minusSeg, prevSeg, midSeg, sucSeg):
        self.minusSeg = minusSeg
        self.prevSeg = prevSeg
        self.midSeg = midSeg
        self.sucSeg = sucSeg

    def __eq__(self, other):
        return self.prevSeg == other.prevSeg and self.midSeg == other.midSeg and self.sucSeg == other.sucSeg

    def __str__(self):
        return str(self.prevSeg)+',' + str(self.midSeg) +',' + str(self.sucSeg)
        # return str(self.prevSeg) + ',' + str(self.midSeg) + ',' + str(self.sucSeg)

    def getTup(self):
        return tuple([self.minusSeg, self.prevSeg, self.midSeg, self.sucSeg])

    def valid(self):
        return self.prevSeg != self.sucSeg

    def extractFeatures(self, edgesDict):
        prevSegNumFeature, prevSegCatFeature = edgeFeature(self.prevSeg, edgesDict, self.minusSeg)
        midSegNumFeature, midSegCatFeature = edgeFeature(self.midSeg, edgesDict, self.prevSeg)
        sucSegNumFeature, sucSegCatFeature = edgeFeature(self.sucSeg, edgesDict, self.midSeg)
        numericalFeatures = [prevSegNumFeature, midSegNumFeature, sucSegNumFeature]
        categoricalFeatures = [prevSegCatFeature, midSegCatFeature, sucSegCatFeature]
        return numericalFeatures, categoricalFeatures


class WindowFromList(Window):
    def __init__(self, segList):
        self.minusSeg = segList[0]
        self.prevSeg = segList[1]
        self.midSeg = segList[2]
        self.sucSeg = segList[3]


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


def Beforeori_cal(coor_a, coor_b, coor_c):
    """
    Calcuate the orientation change from vector ab to bc (0 in default, right turn > 0, left turn < 0)

    10 degree is the threshold of a turn.

    Args:
        coor_a: coordinate of point a
        coor_b: coordinate of point b
        coor_c: coordinate of point c

    """
    a = np.array(coor_a)
    b = np.array(coor_b)
    c = np.array(coor_c)
    v_ab = b - a
    v_bc = c - b
    cosangle = v_ab.dot(v_bc) / (np.linalg.norm(v_bc) * np.linalg.norm(v_ab) + 1e-16)
    res = math.acos(cosangle) * 180 / np.pi if np.cross(v_ab, v_bc) < 0 else -math.acos(cosangle) * 180 / np.pi
    return res if not math.isnan(res) else 0

def ori_cal(coor_a, coor_b, coor_c):
    """
    Calcuate the orientation change from vector ab to bc (0 in default, right turn > 0, left turn < 0)

    10 degree is the threshold of a turn.

    Args:
        coor_a: coordinate of point a
        coor_b: coordinate of point b
        coor_c: coordinate of point c

    Returns:
    """
    x_a, y_a = coor_a
    x_b, y_b = coor_b
    x_c, y_c = coor_c
    v_ab_x = x_b - x_a
    v_ab_y = y_b - y_a
    v_bc_x = x_c - x_b
    v_bc_y = y_c - y_b
    v_bc_norm = math.sqrt(v_bc_x**2 + v_bc_y**2)
    v_ab_norm = math.sqrt(v_ab_x ** 2 + v_ab_y ** 2)
    cosangle = (v_ab_x*v_bc_x + v_ab_y*v_bc_y) / (v_bc_norm * v_ab_norm + 1e-16)
    cross = v_ab_x*v_bc_y - v_ab_y*v_bc_x
    res = math.acos(cosangle) * 180 / np.pi if cross < 0 else -math.acos(cosangle) * 180 / np.pi
    return res if not math.isnan(res) else 0