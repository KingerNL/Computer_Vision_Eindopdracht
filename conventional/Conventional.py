# This document isn't written in PEP8, will try and change this to a later date.
# -=-=-=- IMPORT LIBRARY'S -=-=-=- #
import numpy as np
import cv2 as cv
import os
import glob
import csv
from typing import List
from math import pi

# -=-=-=-=- CLASSES -=-=-=-=- #

class image():
    
    def __init__(self, cv_image, doc_name: str):
        self.contours: list = []
        self.cv_image       = cv_image
        self.name           = doc_name
    
    def add_contour(self, contour):
        self.contours.append(contour)

class object():
    def __init__(self, outline, kind_of_object: str, position: tuple, oriëntation: float, color: str):
        self.kind_of_object = kind_of_object
        self.color          = color
        self.position       = position
        self.oriëntation    = oriëntation
        self.outline        = outline

# -=-=-=-=- DECLARE FUNCTIONS -=-=-=-=- #

def roundness(contour, moments) -> float:
    length = cv.arcLength(contour, True)
    k = (length * length) / (moments['m00'] * 4 * pi)
    # print(k)
    return k

def find_object(contour, moment) -> str:
    k_value = roundness(contour, moment)
    if k_value < 3:
        chosen_object = 'ring'
    else:
        chosen_object = 'unknown_object'
    return chosen_object

def find_color(contour, original_image) -> str:
    # TODO: Check mean color and compare to pre defined colors
    return "unknown_color"

# -=-=-=- VARIABLES -=-=-=- #

# Folder variables
images: List[image]  = []
root_dir    = os.path.abspath("./")
input_dir   = os.path.join(root_dir,'input_conventional','*.jpg')
output_dir  = None # later defined 

#             Hue, Saturation, Value
lower_color = (10,     20,      35)
upper_color = (30,     200,     180)

# how big the smallest area should be, the other ones get filtered.
filter_contour_area = 900

# how big the kernel should be for the dilation / erosion.
kernel = np.ones((4, 4), np.uint8)

# pre-defined colors & min rmse
white  = np.array(["red",    (255,255,255)], dtype=object)
black  = np.array(["black",  (0, 0, 0)],     dtype=object)
pink   = np.array(["pink",   (300, 90 ,53)], dtype=object)
metal  = np.array(["metal",  (0, 67 ,67)],   dtype=object)
colors = np.array([white, black, pink, metal])
min_rmse = 1000000

# -=-=-=- READ IMAGES  -=-=-=- #

for img in glob.glob(input_dir):
    cv_img  = cv.imread(img)
    img     = image(cv_img, os.path.basename(img))
    images.append(img)

# -=-=-=- MASK IMAGES -=-=-=- #

for img in images:
    
    # set image in hue color space
    hsv_image = cv.cvtColor(img.cv_image, cv.COLOR_BGR2HSV)
    
    # erode / dilate images for better quality masks
    mask = cv.inRange(hsv_image, lower_color, upper_color)
    mask = cv.bitwise_not(mask)
    
    cv.erode(mask, kernel, iterations=1)
    cv.erode(mask, kernel, iterations=1)

    cv.dilate(mask, kernel, iterations=1)
    cv.dilate(mask, kernel, iterations=1)
    cv.dilate(mask, kernel, iterations=1)
    
    # -=-=- FIND CONTOURS -=-=- #
    contours, hierarchy = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    
    # -=- filter and add information to contours -=- #
    for contour in contours:
        if (cv.contourArea(contour) > filter_contour_area):
            # -=- find Moment and center point -=- #
            M = cv.moments(contour)
            cX = int(M['m10'] / M['m00'])
            cY = int(M['m01'] / M['m00'])

            # -=- find oriëntation -=- #
            (x,y),(MA,ma),angle = cv.fitEllipse(contour)
            
            # -=- find closest color -=- #
            color = find_color(img.cv_image , contour)
            
            # -=- find kind of object -=- #
            identified_item = find_object(contour, M)
            
            # -=- add to list -=- #
            item = object(contour, identified_item, (cX, cY), angle, color)
            img.add_contour(item)

    # print("found", len(img.contours), "contours in", img.name)

    # -=-=- draw contours and put text -=-=- #    
    for contour in range(len(img.contours)):
        # to make a specific contour, use cnt = contours[1], and cnt as a var (instead of img.contours)
        cv.drawContours(img.cv_image, img.contours[contour].outline, -1, (255,0,255), 3)
        cv.putText(img.cv_image, img.contours[contour].kind_of_object, img.contours[contour].position, cv.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 4)
    
    # -=-=- save image -=-=- #
    output_dir  = os.path.join(root_dir,'output_conventional', img.name)
    cv.imwrite(output_dir, img.cv_image)
    
    # -=-=- save data to csv -=-=- #
    # with open('./output_data.csv', 'a', encoding='UTF8', newline='') as csv_file:
    #     csv_writer = csv.writer(csv_file)
    #     for contour in range(len(img.contours)):
    #         csv_writer.writerow((img.name, img.contours[contour].kind_of_object, img.contours[contour].position, img.contours[contour].oriëntation, img.contours[contour].color))
            
        

print("done!")