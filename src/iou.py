# -*- coding: utf-8 -*-
"""
Created on Wed Aug 11 00:28:06 2021

@author: SUBHADEEP
"""


#// Define junion of regions
def union(au, bu, areaIntersection):
	areaA = (au[2] - au[0]) * (au[3] - au[1])
	areaB = (bu[2] - bu[0]) * (bu[3] - bu[1])
	areaUnion = areaA + areaB - areaIntersection
	return areaUnion

#// Define intersection of regions
def intersection(ai, bi):
	x = max(ai[0], bi[0])
	y = max(ai[1], bi[1])
	w = min(ai[2], bi[2]) - x
	h = min(ai[3], bi[3]) - y
	if w < 0 or h < 0:
		return 0
	return w*h

#// Define Intersection of Union (IOU)
def iou(a, b):
	if a[0] >= a[2] or a[1] >= a[3] or b[0] >= b[2] or b[1] >= b[3]:
		return 0.0

	areaI = intersection(a, b)
	areaU = union(a, b, areaI)

	return float(areaI) / float(areaU + 1e-10)