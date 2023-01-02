# -=-=-=- IMPORT LIBRARY'S -=-=-=- #
import numpy as np
import cv2 as cv
import os
import glob

# -=-=-=-=- IMAGE CLASS -=-=-=-=-

class image():
    def __init__(self, cv_image, name):
        self.cv_image = cv_image
        self.name = name
    
    def set_color(self, color):
        self.color = color
    
    def set_contours(self, contour):
        self.contour = contour
        
            
# -=-=-=- VARIABELEN -=-=-=-

# Folder variabelen
images      = []
root_dir    = os.path.abspath("./")
input_dir   = os.path.join(root_dir,'input_conventional','*.jpg')

# Dit zijn de:         Hue, Saturation, Value
lower_color = np.array([10,     20,      35])
upper_color = np.array([30,     200,     180])

kernel = np.ones((4, 4), np.uint8)

# -=-=-=- IMAGES UITLEZEN -=-=-=- #

for img in glob.glob(input_dir):
    cv_img = cv.imread(img)
    img = image(cv_img, os.path.basename(img))
    images.append(img)

# -=-=-=- IMAGES MASKEN -=-=-=- #

for img in images:

    # zet afbeelding in hue color space
    hsv_image = cv.cvtColor(img.cv_image, cv.COLOR_BGR2HSV)
    
    # erode / dilate afbeelding voor betere kwaliteid masks
    mask = cv.inRange(hsv_image, lower_color, upper_color)
    mask = cv.bitwise_not(mask)
    
    mask = cv.erode(mask, kernel, iterations=1)
    mask = cv.erode(mask, kernel, iterations=1)

    mask = cv.dilate(mask, kernel, iterations=1)
    mask = cv.dilate(mask, kernel, iterations=1)
    mask = cv.dilate(mask, kernel, iterations=1)
    
    # -=-=- contours vinden -=-=- #
    contours, hierarchy = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

    img.contour = cv.drawContours(img.cv_image, contours, -1, (255,0,255), 5)

    # -=-=- afbeelding opslaan -=-=- #
    output_dir  = os.path.join(root_dir,'output_conventional', img.name)
    
    cv.imwrite(output_dir, img.contour)
