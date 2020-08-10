import json
import os
import math
import subprocess

import torch
import numpy as np
from PIL import Image, ImageDraw

from Conifer_Trach_Fasterrcnn import get_prediction_boxes

MAX_TRANSFORM_SIZE = 8100
RING_DERIVATIVE_CONFIDENCE = 10  # value for derivative to at least be to signify change in size, color, etc for ring
RING_AREA_CHANGE_CONFIDENCE = 200 #400 #meh
IMAGEJ_SCRIPTS_DIRECTORY = r'imagej_scripts'
IMAGEJ = r"I:\Fiji.app\ImageJ-win64.exe"
IMAGEJ_GET_LINE = r'get_line.py'


# LINE/RECTANGLE
def line_rect(x1, y1, x2, y2, rx, ry, rw, rh):

    # check if the line has hit any of the rectangle's sides
    # uses the Line/Line function below
    left = line_line(x1,y1,x2,y2, rx,ry,rx, ry+rh)
    right = line_line(x1, y1, x2, y2, rx+rw, ry, rx+rw, ry+rh)
    top = line_line(x1, y1, x2, y2, rx, ry, rx+rw, ry)
    bottom = line_line(x1, y1, x2, y2, rx, ry+rh, rx+rw, ry+rh)

    # if ANY of the above are true, the line
    # has hit the rectangle
    if left or right or top or bottom:
        #TODO: return intersection points
        return True
    return None


# get prediction from the FasterRCNN model Conifer_Trach_Fasterrcnn
def get_prediction(image_name, model=None):

    return get_prediction_boxes(image_name, MAX_TRANSFORM_SIZE)


# LINE/LINE
def line_line(x1,  y1,  x2,  y2,  x3,  y3,  x4,  y4):

    # calculate the direction of the lines
    uA = ((x4-x3)*(y1-y3) - (y4-y3)*(x1-x3)) / ((y4-y3)*(x2-x1) - (x4-x3)*(y2-y1))
    uB = ((x2-x1)*(y1-y3) - (y2-y1)*(x1-x3)) / ((y4-y3)*(x2-x1) - (x4-x3)*(y2-y1))

    # if uA and uB are between 0-1, lines are colliding
    if uA >= 0 and uA <= 1 and uB >= 0 and uB <= 1:
        return True

    return False


class Line:
    slope = 0
    point = tuple()

    def __init__(self, points):
        self.slope = (points['y2'] - points['y1'])/(points['x2']-points['x1'])
        self.point = (points['x1'],  points['y1'])

    def __getitem__(self, item):
        return self.slope*(item - self.point[0]) + self.point[1]

    def in_box(self, box):
        # TODO: check if line goes through box


        return False


image_path = r"I:\Research\herotest_pmc10b_1571-90_5x_03-12-2020_18-14-1test2.jpg"

# Path to get_line macro for ImageJ
script_path = os.path.join(IMAGEJ_SCRIPTS_DIRECTORY, IMAGEJ_GET_LINE)
# x = subprocess.check_output([IMAGEJ, "-macro", script_path, image_path])

# Retreive line data that was drawn in Imagej
with open('I:\Research\dendrotools\dendrotools\imagej_scripts\data.txt') as json_file:
    data = json.load(json_file)

l1 = Line(data)
# slope = (data['y2'] - data['y1'])/(data['x2']-data['x1'])
# point = (data['x1'],  data['y1'])

# List that will hold onto formatted bounding boxes
bbox_list = list()

# Get unformatted tensor bounding boxes
bboxes = get_prediction(image_path)

# TODO: Loop through bounding boxes to put in better format

# Loop through bounding boxes to find those intersecting with drawn line
for bbox in bboxes:
    bbox = bbox.numpy()
    bbox_w = bbox[2] - bbox[0]
    bbox_h = bbox[3] - bbox[1]
    if line_rect(data['x1'], data['y1'], data['x2'], data['y2'], bbox[0], bbox[1], bbox_w, bbox_h):
        centerx = 0
        bbox_list.append([bbox[0], bbox[1], bbox_w, bbox_h, centerx])


bbox_list = sorted(bbox_list, key=lambda box: box[0])

im = Image.open(image_path)
box_img = ImageDraw.Draw(im)
# for box in bbox_list:
    # box_img.rectangle([box[0], box[1], box[0] + box[2], box[1] + box[3]], outline="red", width=2)

# im.show()
center2 = ()
center1 = ()

box = bbox_list[0]
box_img.rectangle([box[0], box[1], box[0] + box[2], box[1] + box[3]], outline="red", width=2)
for i in range(1, len(bbox_list)):
    box = bbox_list[i]
    box2 = bbox_list[i - 1]
    box_img.rectangle([box[0], box[1], box[0] + box[2], box[1] + box[3]], outline="red", width=2)
    area_diff = bbox_list[i][2]*bbox_list[i][3] - bbox_list[i-1][2]*bbox_list[i-1][3]
    box[4] = (box[0] + box[2] / 2, box[1] + box[3] / 2)
    box2[4] = (box2[0] + box2[2] / 2, box2[1] + box2[3] / 2)
    distance = math.sqrt((box2[4][1] - box[4][1])**2 + (box2[4][0] - box[4][0])**2)
    change_area_over_x = area_diff / distance
    # print(area_diff)
    # im.show()
    if change_area_over_x > RING_DERIVATIVE_CONFIDENCE: #and distance < 200:
        center1 = (box[0] + box[2]/2, box[1] + box[3]/2)
        center2 = (box2[0] + box2[2]/2, box2[1] + box2[3]/2)
        box_img.line(((center1[0] + center2[0])/2-1,
                      (center1[1] + center2[1])/2-1,
                      (center1[0] + center2[0])/2+1,
                      (center1[1] + center2[1])/2+1), fill="green", width=2)

        slope = (center2[1] - center1[1])/(center2[0] - center1[0])
        slope_perp = -1/slope
        intercept = (center1[1] + center2[1])/2 - slope_perp*(center1[0] + center2[0])/2
        w, h = im.size
        point1 = (h-intercept)/slope_perp, h
        point2 = (-1*intercept)/slope_perp, 0
        # box_img.line((point1, point2), fill='purple', width=2)
        break

slope_org_line = (data['y2'] - data['y1'])/(data['x2'] - data['x1'])
slope_for_new_line = -1/((data['y2'] - data['y1'])/(data['x2'] - data['x1']))
point = (((h-center1[1]+slope_for_new_line*center1[0])/slope_for_new_line), h)
point2 = ((-1*point[1]+slope_for_new_line*point[0])/slope_for_new_line, 0)

# box_img.line((point, point2), width=2, fill='red')
#  Show image for debugging purposes...
im.show()


bbox_list = list()
bbox_new = []
bboxes = get_prediction(image_path)
for bbox in bboxes:
    bbox = bbox.numpy()
    bbox_w = bbox[2] - bbox[0]
    bbox_h = bbox[3] - bbox[1]
    if line_rect(point[0], point[1], point2[0], point2[1], bbox[0], bbox[1], bbox_w, bbox_h):
        centerx = 0
        bbox_list.append([bbox[0], bbox[1], bbox_w, bbox_h, centerx])

bbox_list = sorted(bbox_list, key=lambda box: box[1])

for box in bbox_list:
    box_img.rectangle([box[0], box[1], box[0] + box[2], box[1] + box[3]], outline="red", width=2)

box_img.rectangle([bbox_list[0][0], bbox_list[0][1], bbox_list[0][0] + bbox_list[0][2], bbox_list[0][1] + bbox_list[0][3]], outline="green", width=2)
im.show()
# box = bbox_list[0]
# box_img.rectangle([box[0], box[1], box[0] + box[2], box[1] + box[3]], outline="red", width=2)
# for i in range(1, len(bbox_list)):


# mark point, left side of square
im.show()

lines = list()
for box in bbox_list:
    center = (box[0] + box[2]/2, box[1] + box[3]/2)
    point2 = ((-1 * point[1] + slope_org_line * point[0]) / slope_for_new_line, 0)
    lines.append(((center[0], center[1]), (0, slope_org_line*(-1*center[0])+center[1])))
    lines.append(((w, slope_org_line*(w-1*center[0])+center[1]), (0, slope_org_line*(-1*center[0])+center[1])))

bbox_list = [0]*len(lines)
bbox_new = []
line_count = 0
for bbox in bboxes:
    bbox = bbox.numpy()
    bbox_w = bbox[2] - bbox[0]
    bbox_h = bbox[3] - bbox[1]
    line_count = 0
    for line in lines:
        # box_img.line((line[0], line[1]), fill='pink', width=2)
        if not type(bbox_list[line_count]) == list:
            bbox_list[line_count] = []
        if line_rect(line[0][0], line[0][1], line[1][0], line[1][1], bbox[0], bbox[1], bbox_w, bbox_h):
            centerx = 0
            bbox_list[line_count].append([bbox[0], bbox[1], bbox_w, bbox_h, centerx])
            break
        line_count += 1

for l in range(len(lines)):
    bbox_list[l] = sorted(bbox_list[l], key=lambda box: box[0])
    for i in range(1, len(bbox_list[l])):
        # TODO: bad assumption here
        box = bbox_list[l][i]
        box2 = bbox_list[l][i - 1]
        box_img.rectangle([box[0], box[1], box[0] + box[2], box[1] + box[3]], outline="pink", width=2)
        area_diff = bbox_list[l][i][2]*bbox_list[l][i][3] - bbox_list[l][i-1][2]*bbox_list[l][i-1][3]
        box[4] = (box[0] + box[2] / 2, box[1] + box[3] / 2)
        box2[4] = (box2[0] + box2[2] / 2, box2[1] + box2[3] / 2)
        distance = math.sqrt((box2[4][1] - box[4][1]) ** 2 + (box2[4][0] - box[4][0]) ** 2)
        change_area_over_x = area_diff / distance
        # print(area_diff)
        # im.show()
        if change_area_over_x > RING_DERIVATIVE_CONFIDENCE and area_diff > RING_AREA_CHANGE_CONFIDENCE: #and distance < 200:
            center1 = (box[0] + box[2]/2, box[1] + box[3]/2)
            center2 = (box2[0] + box2[2]/2, box2[1] + box2[3]/2)
            box_img.line(((center1[0] + center2[0])/2-1,
                          (center1[1] + center2[1])/2-1,
                          (center1[0] + center2[0])/2+1,
                          (center1[1] + center2[1])/2+1), fill="green", width=2)

            slope = (center2[1] - center1[1])/(center2[0] - center1[0])
            slope_perp = -1/slope
            intercept = (center1[1] + center2[1])/2 - slope_perp*(center1[0] + center2[0])/2
            w, h = im.size
            point1 = (h-intercept)/slope_perp, h
            point2 = (-1*intercept)/slope_perp, 0
            # box_img.line((point1, point2), fill='purple', width=2)
            break


# box_img.line(((w, slope_org_line*(w-1*center1[0])+center1[1]), (0, slope_org_line*(-1*center1[0])+center1[1])), fill='red', width=2)
im.show()


def from_perpendicular_slope():
    None


def average_perpendicular_slope():
    None


def from_multiple_points():
    None
