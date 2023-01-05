import os
import cv2
import numpy as np

# Uses BGR
def labelRoadmark(percentage):                 
    if percentage == 0:
        return ['A+', (0, 255, 0)]   # Green
    elif percentage <= 5:
        return ['A', (51, 255, 102)]  # Light Green
    elif percentage <= 20:
        return ['B', (0, 255, 255)]  # Yellow
    elif percentage <= 30:
        return ['C', (0, 153, 255)]  # Orange
    elif percentage > 30:
        return ['D', (0, 0, 255)]    # Red

SAVE_PATH = os.path.abspath("../detectron2-final/output")

# Set the lower range value of color in HLS
lower_range = np.array([0, 125, 0])
# Set the upper range value of color in HLS
upper_range = np.array([255, 255, 255])

fontFace = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 1
borderThickness = 2

def classify(images, masks):
    for index, (img, mask) in enumerate(zip(images, masks)):

        imgHLS = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        mask = mask.astype(np.uint8)
        mask[mask > 0] = 255

        for i in range(mask.shape[0]):
            # Creating threshold of the manually created mask and then finding contours of roadmarks
            gray = mask[i,:,:]
            thresh = cv2.threshold(
                gray, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)[1]
            contour = cv2.findContours(
                thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
            contour = max(contour, key=cv2.contourArea)

            # Create three channel mask for single part of the roadmark
            mask_part = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

            # Remove the concavities
            hull = cv2.convexHull(contour)
            hull = [tuple(p[0]) for p in hull]
            
            # Find all of the four corners
            tr = max(hull, key=lambda x: x[0] - x[1])
            tl = min(hull, key=lambda x: x[0] + x[1])
            br = max(hull, key=lambda x: x[0] + x[1])
            bl = min(hull, key=lambda x: x[0] - x[1])

            width = max(abs(tr[0] - tl[0]), abs(br[0] - bl[0]))
            height = max(abs(bl[1] - tl[1]), abs(br[1] - tr[1]))

            # Tranform img and mask to top down view
            pts1 = np.float32([tr, tl, br, bl])
            # Height start in top left corner and goes downwards
            pts2 = np.float32(
                [[width, 0], [0, 0], [width, height], [0, height]])
            matrix = cv2.getPerspectiveTransform(pts1, pts2)
            tf_imgHLS = cv2.warpPerspective(imgHLS, matrix, (width, height))
            tf_mask_part = cv2.warpPerspective(mask_part, matrix, (width, height))

            # Count every pixel in mask
            pixels = np.sum(gray > 0)

            # Combine image and mask to remove noise
            #result = cv2.bitwise_and(tf_imgHLS, tf_mask_part) # Bird's-eye view not always working
            result = cv2.bitwise_and(imgHLS, mask_part)

            # Filter the white
            roadmark = cv2.inRange(result, lower_range, upper_range)

            white_pixels = np.sum(roadmark > 0)
            damage = (pixels - white_pixels) / pixels * 100

            # Highlight roadmarks and labels them on damage
            cv2.drawContours(img, [contour], -1,
                             labelRoadmark(damage)[1], borderThickness)                           
            cv2.putText(img, labelRoadmark(damage)[
                        0], (contour[0][0] - 5), fontFace, fontScale, labelRoadmark(damage)[1], borderThickness)
            print('Image[{}] damaged: {:.2f}% [{}]'.format(
                index, damage, labelRoadmark(damage)[0]))

        cv2.imwrite('{}/Image{}-labeled.jpg'.format(SAVE_PATH, index), img)