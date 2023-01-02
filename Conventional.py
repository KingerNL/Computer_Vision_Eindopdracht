# -=-=-=- IMPORT LIBRARY'S -=-=-=- #
import numpy as np
import cv2 as cv
import os
import glob

# -=-=-=-=- IMAGE CLASS -=-=-=-=- #

class image():
    
    def __init__(self, cv_image, name):
        self.contours   = []
        self.cv_image   = cv_image
        self.name       = name
    
    def add_contour(self, contour):
        self.contours.append(contour)

class object():
    def __init__(self):
        self.kind_of_object = None
        self.color          = None
        self.position       = None
        self.oriëntation    = None
    
# -=-=-=- VARIABLES -=-=-=- #

# Folder variables
images      = []
root_dir    = os.path.abspath("./")
input_dir   = os.path.join(root_dir,'input_conventional','*.jpg')
output_dir  = None # later defined 

#                      Hue, Saturation, Value
lower_color = np.array([10,     20,      35])
upper_color = np.array([30,     200,     180])

contour_area = 900

kernel = np.ones((4, 4), np.uint8)

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
    
    mask = cv.erode(mask, kernel, iterations=1)
    mask = cv.erode(mask, kernel, iterations=1)

    mask = cv.dilate(mask, kernel, iterations=1)
    mask = cv.dilate(mask, kernel, iterations=1)
    mask = cv.dilate(mask, kernel, iterations=1)
    
    # -=-=- find contours -=-=- #
    contours, hierarchy = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    
    for contour in contours:
        if (cv.contourArea(contour) > contour_area):
            img.add_contour(contour)
            contour = object()
            
    # print("found", len(img.contours), "contours in", img.name)

    # -=-=- find oriëntation -=-=- #
    # TODO: Check this function
    for contour in img.contours:
        (x,y),(MA,ma),angle = cv.fitEllipse(contour)
        object.oriëntation = angle
        
    # -=-=- find closest color -=-=- #
    # TODO: make this function
    
    # -=-=- find kind of object -=-=- #
    # TODO: Make this function
    

    # -=-=- draw contours and put text -=-=- #
    # om een specifieke contour te maken, gebruik cnt = contours[1], en cnt als var
    img_with_contour = cv.drawContours(img.cv_image, img.contours, -1, (255,0,255), 3)
    
    for contour in range(len(img.contours)):
        M = cv.moments(img.contours[contour])

        cX = int(M['m10'] / M['m00'])
        cY = int(M['m01'] / M['m00'])
        
        # TODO: Change to which object we need to indentify
        text = "object"
        
        img_with_text = cv.putText(img_with_contour, text, (cX, cY), cv.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 255), 3)
    
    # -=-=- save image -=-=- #
    output_dir  = os.path.join(root_dir,'output_conventional', img.name)
    
    cv.imwrite(output_dir, img_with_text)
