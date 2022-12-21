# -=-=-=- IMPORT LIBRARY'S -=-=-=- #
import numpy as np
import cv2 as cv

# -=-=-=- IMAGES UITLEZEN -=-=-=- #
original_image = cv.imread('./test_images_2/foto_mat (1).jpg')

hsv_image = cv.cvtColor(original_image, cv.COLOR_BGR2HSV)
    
# -=-=-=- CONTOURS VINDEN -=-=-=- #
# Dit zijn de:         Hue, Saturation, Value
lower_color = np.array([10,     20,      35])
upper_color = np.array([30,     200,     180])

# -=-=-=- IMAGE PRE-PROCESSING -=-=-=- #
"""
    1. haal alle aangegeven kleuren weg
    2. inverteer afbeelding
"""
mask = cv.inRange(hsv_image, lower_color, upper_color)
mask = cv.bitwise_not(mask)    

# -=-=-=- ERODE / DILATE -=-=-=- #
kernel = np.ones((4, 4), np.uint8)

mask = cv.erode(mask, kernel, iterations=1)
mask = cv.erode(mask, kernel, iterations=1)

mask = cv.dilate(mask, kernel, iterations=1)
mask = cv.dilate(mask, kernel, iterations=1)


# -=-=-=- CONTOURS VINDEN -=-=-=- #
contours, hierarchy = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

cv.drawContours(original_image, contours, -1, (255,0,255), 5)

# -=-=-=- AFBEELDINGEN OPSLAAN -=-=-=- #
cv.imwrite("./result_conventional/result1.jpg", original_image)