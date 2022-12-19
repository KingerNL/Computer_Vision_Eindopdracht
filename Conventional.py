import numpy as np
import cv2 as cv

frame = cv.imread('./test_images_perfect/Image_Perfect (1).jpg')

hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    
# Threshold of blue in HSV space
lower_blue = np.array([30, 30, 100])
upper_blue = np.array([70, 190, 210])

# preparing the mask to overlay
mask = cv.inRange(hsv, lower_blue, upper_blue)

# invert mask naar black background, white object    
mask = cv.bitwise_not(mask)    

kernel = np.ones((5, 5), np.uint8)

mask = cv.erode(mask, kernel, iterations=1)
mask = cv.erode(mask, kernel, iterations=1)

mask = cv.dilate(mask, kernel, iterations=1)
mask = cv.dilate(mask, kernel, iterations=1)


cv.imwrite("./result_conventional/result1.jpg", mask)