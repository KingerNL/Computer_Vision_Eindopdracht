# This document isn't written in PEP8, will try and change this to a later date.
# Made by: Mart Veldkamp
# -=-=-=- IMPORT LIBRARY'S -=-=-=- #
import numpy as np
import cv2 as cv
import os
import glob
import csv
from typing import List
from math import pi, sqrt

# -=-=-=-=- BEGIN SCREEN -=-=-=-=- #

print(''.join([line for line in open('../image_objects_ordered/coding.txt', 'r')]))
input('press enter to start detection...')

# -=-=-=-=- CLASSES -=-=-=-=- #

class image():
    
    def __init__(self, cv_image, doc_name: str):
        self.contours: list = []
        self.cv_image       = cv_image
        self.name           = doc_name
    
    def add_contour(self, contour):
        self.contours.append(contour)
    
    def draw_contour(self, original_image, contour, kind_of_object: str, color: str):
        if kind_of_object == 'check valve':
            contour_color = blue[1]
        elif color == 'white':
            contour_color = white[1]
        elif color == 'black':
            contour_color = black[1]
        elif color == 'pink':
            contour_color = pink[1]
        else:
            contour_color = metal[1]

        cv.drawContours(original_image, contour, -1, contour_color, 4)

    def draw_bounding(self, original_image, contour, kind_of_object, color):
        if kind_of_object == 'check valve':
            contour_color = blue[1]
        elif color == 'white':
            contour_color = white[1]
        elif color == 'black':
            contour_color = black[1]
        elif color == 'pink':
            contour_color = pink[1]
        else:
            contour_color = metal[1]

        # -=-=- Draw bounding box with text -=-=- #
        x,y,w,h = cv.boundingRect(contour)
        cv.rectangle(original_image, (x,y), (x+w,y+h), contour_color, 3)
        cv.putText(original_image, kind_of_object, (x, y-10), cv.FONT_HERSHEY_SIMPLEX, 1.2, black[1], 4)

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
lower_background_color = (13,     145,      20)
upper_background_color = (35,     255,     255)
lower_blue             = (30,     120,      20)
upper_blue             = (120,    255,     255)
lower_pink             = (130,    50,      20)
upper_pink             = (180,    255,     255)

# how big the smallest area should be, the other ones get filtered.
filter_contour_area = 900

# mask for identifying nuts
contours_nut, _ = cv.findContours(cv.bitwise_not(cv.inRange(cv.cvtColor(cv.imread("./labeled_images/nut.jpg"), cv.COLOR_BGR2HSV), lower_background_color, upper_background_color)), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
nut_contour_list    = []
for contour in contours_nut:
    if (cv.contourArea(contour) > 1000):
        nut_contour_list.append(contour)

# how big the kernel should be for the dilation / erosion.
kernel = np.ones((4, 4), np.uint8)

# pre-defined colors (BGR)
#TODO: Check if color value's corresponts. Or change the heuristic.
white: np.ndarray  = np.array(["white",  (210, 210, 210)],   dtype=object)
black: np.ndarray  = np.array(["black",  (30,  30,  30 )],    dtype=object)
metal: np.ndarray  = np.array(["metal",  (120, 120, 120)], dtype=object)

pink:  np.ndarray  = np.array(["pink",   (120, 90,  220)], dtype=object)
blue:  np.ndarray  = np.array(["blue",   (255, 100 ,10 )],  dtype=object)

colors: list = (white, black, metal)

# -=-=-=-=- DECLARE FUNCTIONS -=-=-=-=- #

def roundness(contour, moments) -> float:
    length = cv.arcLength(contour, True)
    k = (length * length) / (moments['m00'] * 4 * pi)
    return k

# TODO: Find ways to detect the other objects
def find_object(contour, moment, mask) -> str:
    
    # check valve check
    blue_mask = np.zeros(mask.shape[:2], dtype="uint8")
    cv.drawContours(blue_mask, [contour], -1, 255, -1)
    squirness = roundness(contour, moment)
    individual_masks = cv.bitwise_and(mask, mask, mask = blue_mask)
    blue_mask = cv.inRange(individual_masks, lower_blue, upper_blue)

    # ring check
    k_value = roundness(contour, moment)
    # nut check
    perc_nut = cv.matchShapes(nut_contour_list[0], contour, cv.CONTOURS_MATCH_I1, 0.0)

    if np.mean(blue_mask) > 0:
        return 'check valve'
    elif k_value < 1.6:
        return 'ring'
    elif perc_nut < 0.04:
        return 'nut'
    elif squirness < 1.4:
        return 'metal attachment'
    else:
        return 'bolt'

def find_color(contour, original_image, cont_mask) -> str:
    
    mask = np.zeros(original_image.shape[:2], dtype="uint8")
    cv.drawContours(mask, [contour], -1, 255, -1)
    
    # check pink
    pink_mask = np.zeros(cont_mask.shape[:2], dtype="uint8")
    cv.drawContours(pink_mask, [contour], -1, 255, -1)
    individual_masks = cv.bitwise_and(cont_mask, cont_mask, mask = pink_mask)
    pink_mask = cv.inRange(individual_masks, lower_pink, upper_pink)
    
    if np.mean(pink_mask) > 0:
        return 'pink'

    mean = cv.mean(original_image, mask=mask)[0:3]
    min_rmse = 1000000
    # print(mean)
    for color in colors:
        bb = color[1][0]
        gg = color[1][1]
        rr = color[1][2]
        rmse = sqrt( ( (mean[2]-rr)*(mean[2]-rr) + (mean[1]-gg)*(mean[1]-gg) + (mean[0]-bb)*(mean[0]-bb) )/3 )

        if rmse < min_rmse:
            min_rmse = rmse
            match_color = color[0]
    
    return match_color

# -=-=-=- READ IMAGES  -=-=-=- #

for img in glob.glob(input_dir):
    images.append(image(cv.imread(img), os.path.basename(img)))

# -=-=-=- MASK IMAGES -=-=-=- #

for img in images:
    
    # set image in hue color space
    hsv_image = cv.cvtColor(img.cv_image, cv.COLOR_BGR2HSV)
    
    # filter image on HSV color ranges
    mask = cv.inRange(hsv_image, lower_background_color, upper_background_color)
    
    # dilate images for better quality masks
    mask_binary = cv.bitwise_not(mask)
    mask_with_color = cv.bitwise_and(hsv_image, img.cv_image, mask = mask_binary)

    cv.dilate(mask_binary, kernel, iterations=1)

    # -=-=- FIND CONTOURS -=-=- #
    contours, _ = cv.findContours(mask_binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

    # -=- filter and add information to contours -=- #
    for contour in contours:
        if (cv.contourArea(contour) > filter_contour_area):
            # -=- find Moment and center point -=- #
            moment = cv.moments(contour)
            cX = int(moment['m10'] / moment['m00'])
            cY = int(moment['m01'] / moment['m00'])

            # -=- find oriëntation -=- #
            (_,_),(_,_),angle = cv.fitEllipse(contour)

            # -=- find closest color -=- #
            color = find_color(contour, img.cv_image, mask_with_color)
            
            # -=- find kind of object -=- #
            identified_item = find_object(contour, moment, mask_with_color)
            
            # -=- add object to list -=- #
            img.add_contour(object(contour, identified_item, (cX, cY), angle, color))

    print("found", len(img.contours), "contour(s) in:", img.name)

    # -=-=- draw contours (bounding box optional) and put text -=-=- #    
    for contour in range(len(img.contours)):
        img.draw_contour(img.cv_image, img.contours[contour].outline, img.contours[contour].kind_of_object, img.contours[contour].color)
        img.draw_bounding(img.cv_image, img.contours[contour].outline, img.contours[contour].kind_of_object, img.contours[contour].color)

# -=-=-=- SAVE IMAGES -=-=-=- #
for img in images:
    output_dir  = os.path.join(root_dir,'output_conventional', img.name)
    cv.imwrite(output_dir, img.cv_image)
    
    # -=-=- save data to csv -=-=- #
    # with open('./output_data.csv', 'a', encoding='UTF8', newline='') as csv_file:
    #     csv_writer = csv.writer(csv_file)
    #     for contour in range(len(img.contours)):
    #         csv_writer.writerow((img.name, img.contours[contour].kind_of_object, 1, img.contours[contour].position, img.contours[contour].oriëntation, img.contours[contour].color))

print("done! data saved to: output_data.csv")