import numpy as np
import pandas as pd
import math

def edgePreprocessing(nodesGdf, edgesGdf, locationRequest):
    edgesGdf['odPair'] = edgesGdf.apply(lambda x: (x.u, x.v), axis=1)
    # mass
    edgesGdf['mass'] = locationRequest.mass
    edgesGdf['mass'] = edgesGdf['mass'].apply(lambda x: (x- 23185.02515) / 8227.65140266416) #8227.65140266416
    # speed limit
    edgesGdf['speedLimit'] = edgesGdf.apply(lambda x: calSpeedlimit(x), axis=1)
    edgesGdf['speedLimit'] = (edgesGdf['speedLimit'] - 80.5318397987996) / 21.7071763681126

    # elevation change
    if 'uElevation' in edgesGdf.columns:
        edgesGdf['elevationChange'] = edgesGdf.apply(lambda x: x['vElevation']-x['uElevation'], axis=1)
    else:
        segmentElevationChange = np.load('statistical data/segmentElevationChange.npy', allow_pickle=True).item()
        edgesGdf['elevationChange'] = edgesGdf.apply(lambda x: segmentElevationChange[(x.u, x.v)], axis=1)

    edgesGdf['elevationChange'] = (edgesGdf['elevationChange'] + 0.00450470150885644) / 8.62149031019689

    # previous orientation
    edgesGdf['points'] = edgesGdf['geometry'].apply(lambda x: pointsInSegment(x))


    # length no changes
    edgesGdf['lengthNormalized'] = (edgesGdf['length'] - 611.410287539911) / 903.292309592642

    # direction angle
    edgesGdf['directionAngle'] = edgesGdf['points'].apply(lambda x: directionAngle(x))
    edgesGdf['directionAngle'] = (edgesGdf['directionAngle'] - 1.67006008669261) / 102.77763894989


    # road type
    edgesGdf['roadtype'] = edgesGdf.apply(lambda x: highway_cal(x), axis=1)
    roadtypeDict = np.load('statistical data/road_type_dictionary.npy', allow_pickle=True).item()
    edgesGdf['roadtype'] = edgesGdf['roadtype'].apply(lambda x: roadtypeDict[x] if x in roadtypeDict else 0)
    roadtypeSet = {0, 1, 2, 3, 4, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 19, 20, 21}
    edgesGdf['roadtype'] = edgesGdf['roadtype'].apply(lambda x: x if x in roadtypeSet else 0)
    # time
    edgesGdf['timeOfTheDay'] = locationRequest.timeOfTheDay
    edgesGdf['timeOfTheDay'] = edgesGdf['timeOfTheDay'].apply(lambda x: calTimeStage(x))
    edgesGdf['dayOfTheWeek'] = locationRequest.dayOfTheWeek

    # lanes
    edgesGdf['lanes'] = edgesGdf.apply(lambda x: cal_lanes(x), axis=1)
    edgesGdf['lanes'] = edgesGdf['lanes'].apply(lambda x: x if x <= 8 else 8)

    #bridge
    edgesGdf['bridgeOrNot'] = edgesGdf['bridge'].apply(lambda x: bridgeOrNot(x))


    # endpoints
    edgesGdf['oriSignal'] = edgesGdf['u'].apply(lambda x: nodesGdf.loc[x, 'highway']).fillna("None")
    edgesGdf['destSignal'] = edgesGdf['v'].apply(lambda x: nodesGdf.loc[x, 'highway']).fillna("None")
    endpoints_dictionary = np.load('statistical data/endpoints_dictionary.npy', allow_pickle=True).item()
    edgesGdf['oriSignalCategoried'] = edgesGdf['oriSignal'].apply(lambda x: endpoints_dictionary[x] if x in endpoints_dictionary else 0)
    edgesGdf['destSignalCategoried'] = edgesGdf['destSignal'].apply(lambda x: endpoints_dictionary[x] if x in endpoints_dictionary else 0)
    endpointsSet = {0, 6, 4, 2, 1, 13}
    edgesGdf['oriSignalCategoried'] = edgesGdf['oriSignalCategoried'].apply(lambda x: x if x in endpointsSet else 0)
    edgesGdf['destSignalCategoried'] = edgesGdf['destSignalCategoried'].apply(lambda x: x if x in endpointsSet else 0)

    edgesGdf['categoricalFeature'] = edgesGdf.apply(lambda x: categoricalFeature(x), axis=1)

    return edgesGdf


def categoricalFeature(arraylike):
    return [arraylike.roadtype, arraylike.timeOfTheDay, arraylike.dayOfTheWeek, arraylike.lanes, arraylike.bridgeOrNot, arraylike.oriSignalCategoried, arraylike.destSignalCategoried]


def bridgeOrNot(bridge):
    if isinstance(bridge,float):
        return 0
    else:
        return 1


def directionAngle(pointsList):
    longitude_o, latitude_o = pointsList[0]
    longitude_d, latitude_d = pointsList[-1]
    direction = [latitude_d - latitude_o, longitude_d - longitude_o]
    direction_array = np.array(direction)
    if abs(np.linalg.norm(direction_array)) < 1e-16:
        return 0
    cosangle = direction_array.dot(np.array([1, 0])) / (np.linalg.norm(direction_array))
    if np.cross(direction_array, np.array([1, 0])) < 0:
        direction_angle = math.acos(cosangle) * 180 / np.pi
    else:
        direction_angle = -math.acos(cosangle) * 180 / np.pi
    return direction_angle


def pointsInSegment(geometry):
    pointsStringList = list(str(geometry)[12: -1].split(", "))
    for i,val in enumerate(pointsStringList):
        pointsStringList[i] = tuple(map(float, val.split(" ")))
    return pointsStringList


def highway_cal(network_seg):
    if 'highway' in network_seg and network_seg['highway']:
        if isinstance(network_seg['highway'], str):
            return network_seg['highway']
        elif isinstance(network_seg['highway'], list):
            return network_seg['highway'][0]
    else:
        return 'unclassified'


def calSpeedlimit(array_like):
    if isinstance(array_like['maxspeed'],float):
        if math.isnan(array_like['maxspeed']):
            if array_like['highway'] == "motorway":
                return 55 * 1.609
            elif array_like['highway'] == "motorway_link":
                return 50 * 1.609
            return 30 * 1.609
        else:
            return array_like['maxspeed'] * 1.609
    else:
        if isinstance(array_like['maxspeed'],list):
            t = array_like['maxspeed'][0]
        else:
            t = array_like['maxspeed']
        res = ''
        flag = 0
        for i in list(t):
            if i.isdigit():
                flag = 1
                res += i
            else:
                if flag == 1:
                    speed = int(res) * 1.609
                    return speed
        if flag == 1:
            speed = int(res) * 1.609
            return speed

def calTimeStage(timeOfTheDay):
    return timeOfTheDay // 4 + 1


def cal_lanes(array_like):
    if isinstance(array_like['lanes'],list):
        for i in array_like['lanes']:
            if i.isdecimal():
                return int(i)
    if isinstance(array_like['lanes'],int):
        return array_like['lanes']
    if pd.isna(array_like['lanes']):
        return 0
    if array_like['lanes'].isalpha():
        return 0
    if array_like['lanes'].isalnum():
        return int(array_like['lanes']) if int(array_like['lanes']) > 0 else 0
    else:
        return 0
