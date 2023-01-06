# This document isn't written in PEP8, will try and change this to a later date.
# -=-=-=- IMPORT LIBRARY'S -=-=-=- #
import numpy as np
import cv2 as cv
import os
import glob
import csv
from typing import List
from math import pi, sqrt

# -=-=-=-=- BEGIN SCREEN -=-=-=-=- #

ascii_file = open('../image_objects_ordered/coding.txt', 'r')
print(''.join([line for line in ascii_file]))
input('press enter to start detection...')

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

# -=-=-=- VARIABLES -=-=-=- #

# Folder variables
images: List[image]  = []
root_dir    = os.path.abspath("./")
input_dir   = os.path.join(root_dir,'input_conventional','*.jpg')
output_dir  = None # later defined 

#                        Hue, Saturation, Value
lower_background_color = (10,     20,      35)
upper_background_color = (30,     200,     170)
lower_blue             = (80,     120,     20)
upper_blue             = (100,    255,     255)
# how big the smallest area should be, the other ones get filtered.
filter_contour_area = 900

# how big the kernel should be for the dilation / erosion.
kernel = np.ones((4, 4), np.uint8)

# pre-defined colors (BGR)
#TODO: Check if color value's corresponts. Or change the heuristic.
white: np.ndarray  = np.array(["white",  (210,210 ,210 )],   dtype=object)
black: np.ndarray  = np.array(["black",  (30, 30  , 30 )],    dtype=object)
pink:  np.ndarray  = np.array(["pink",   (120, 90 , 220)], dtype=object)
metal: np.ndarray  = np.array(["metal",  (120, 120,120 )], dtype=object)
blue:  np.ndarray  = np.array(["blue",   (255  , 100  ,10 )],  dtype=object)
colors: list = (white, black, pink, metal)

# -=-=-=-=- DECLARE FUNCTIONS -=-=-=-=- #

def roundness(contour, moments) -> float:
    length = cv.arcLength(contour, True)
    k = (length * length) / (moments['m00'] * 4 * pi)
    return k

# TODO: Find ways to detect the other objects
def find_object(contour, moment, mask) -> str:
    k_value = roundness(contour, moment)
    
    blue_mask = np.zeros(mask.shape[:2], dtype="uint8")
    cv.drawContours(blue_mask, [contour], -1, 255, -1)
    individual_masks = cv.bitwise_and(mask, mask, mask = blue_mask)
    blue_mask = cv.inRange(individual_masks, lower_blue, upper_blue)
    
    # print(np.mean(blue_mask))
    if np.mean(blue_mask) > 0:
        return 'check valve'
    elif k_value < 2.5:
        return 'ring'
    else:
        return '?'

def find_color(contour, original_image) -> str:
    
    mask = np.zeros(original_image.shape[:2], np.uint8)
    cv.drawContours(mask, contour, -1, 255, -1)
    
    # mean returned: BGR = Blue, Green, Red
    mean = cv.mean(original_image, mask=mask)[0:3]
    min_rmse = 1000000
    # print(mean)
    for color in colors:
        bb = color[1][0]
        gg = color[1][1]
        rr = color[1][2]
        rmse = sqrt( ( (mean[2]-rr)*(mean[2]-rr) + (mean[1]-gg)*(mean[1]-gg) + (mean[0]-bb)*(mean[0]-bb) )/3 )
        colorname = color[0]
        # print(colorname,rmse)
        if rmse < min_rmse:
            min_rmse = rmse
            match_color = color[0]
    
    # print("")
    # print("match_color:", match_color)
    # print("rmse:", min_rmse)
    # print("")
    return match_color

# -=-=-=- READ IMAGES  -=-=-=- #

for img in glob.glob(input_dir):
    cv_img  = cv.imread(img)
    img     = image(cv_img, os.path.basename(img))
    images.append(img)

# -=-=-=- MASK IMAGES -=-=-=- #

for img in images:
    
    # set image in hue color space
    hsv_image = cv.cvtColor(img.cv_image, cv.COLOR_BGR2HSV)
    
    # filter image on HSV color ranges
    mask = cv.inRange(hsv_image, lower_background_color, upper_background_color)
    # erode / dilate images for better quality masks
    mask_binary = cv.bitwise_not(mask)
    mask_with_color = cv.bitwise_and(hsv_image, img.cv_image, mask = mask_binary)

    cv.erode(mask_binary, kernel, iterations=1)
    cv.erode(mask_binary, kernel, iterations=1)

    cv.dilate(mask_binary, kernel, iterations=1)
    cv.dilate(mask_binary, kernel, iterations=1)
    cv.dilate(mask_binary, kernel, iterations=1)

    # -=-=- FIND CONTOURS -=-=- #
    contours, hierarchy = cv.findContours(mask_binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

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
            color = find_color(contour, img.cv_image)
            
            # -=- find kind of object -=- #
            identified_item = find_object(contour, M, mask_with_color)
            
            # -=- add to list -=- #
            item = object(contour, identified_item, (cX, cY), angle, color)
            img.add_contour(item)

    print("found", len(img.contours), "contour(s) in:", img.name)

    # -=-=- draw contours and put text -=-=- #    
    for contour in range(len(img.contours)):
        if img.contours[contour].kind_of_object == 'check valve':
            color = blue[1]
        elif img.contours[contour].color == 'white':
            color = white[1]
        elif img.contours[contour].color == 'black':
            color = black[1]
        elif img.contours[contour].color == 'pink':
            color = pink[1]
        else:
            color = metal[1]

        cv.drawContours(img.cv_image, img.contours[contour].outline, -1, color, 3)
        
        if img.contours[contour].color == 'black':
            color = white[1]
        else:
            color = black[1]
        
        cv.putText(img.cv_image, img.contours[contour].kind_of_object, img.contours[contour].position, cv.FONT_HERSHEY_SIMPLEX, 1.2, color, 4)
    
    # -=-=- save image -=-=- #
    output_dir  = os.path.join(root_dir,'output_conventional', img.name)
    cv.imwrite(output_dir, img.cv_image)
    
    # -=-=- save data to csv -=-=- #
    with open('./output_data.csv', 'a', encoding='UTF8', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        for contour in range(len(img.contours)):
            csv_writer.writerow((img.name, img.contours[contour].kind_of_object, 1, img.contours[contour].position, img.contours[contour].oriëntation, img.contours[contour].color))
            
print("done! data saved to: output_data.csv")