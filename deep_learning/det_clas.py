import os
import cv2
import numpy as np
import csv
from custom_config_detectron2 import *
from math import sqrt

SAVE_PATH = os.path.abspath("../deep_learning/output_test")

fontFace = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 1
borderThickness = 2

def classify(images, img_names):

    white = np.array(["white",(210, 210, 210)],dtype=object)
    black = np.array(["black",(30,  30,  30 )],dtype=object)
    metal = np.array(["metal",(120, 120, 120)],dtype=object)
    colors: list = (white, black, metal)
    lower_pink= (130,50,20)
    upper_pink= (180,255,255)

    # Run code over just one image of the input list of images
    for i in range(len(images)):
        img = images[i]
        # Run prediction over image
        result = predictor(img)
        img_name = img_names[i]
        
        # Extract mask out of prediction
        mask = result['instances'].pred_masks.numpy()
        imgHLS = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # Convert binary mask to integers and make a 'black white' mask
        mask = mask.astype(np.uint8)
        mask[mask > 0] = 255

        # Run code over each individual mask, and classify its class, position, orientation and color
        for j in range(mask.shape[0]):

            # Creating threshold of the manually created mask and then finding contours of roadmarks
            gray = mask[j,:,:]
            thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)[1]
            contour2 = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
            cntrs = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contour = max(contour2, key=cv2.contourArea)
            # Create three channel mask for single part of the roadmark
            mask_part = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            # Combine image and mask to remove noise
            result_mask = cv2.bitwise_and(imgHLS, mask_part)
                  
            #MIN_AREA_RECT COMPUTES SMALLEST BOUNDING BOX AND THUS ALSO IS ROTATED

            # contour = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cntrs = cntrs[0] if len(cntrs) == 2 else cntrs[1]

            # get rotated rectangle from outer contour
            rotrect = cv2.minAreaRect(cntrs[0])
            box = cv2.boxPoints(rotrect)
            box = np.int0(box)

            width = int(rotrect[1][0])
            height = int(rotrect[1][1])
            angle = int(rotrect[2])

            if width < height:
                angle = 90 - angle
            else:
                angle = -angle


            # Extract classes out of predicion, get the class and replace it for the corresponding label
            classes = result['instances'].pred_classes.cpu().detach().numpy()
            kind_of_object = classes[j]
            kind_of_object = str(kind_of_object)
            kind_of_object = kind_of_object.replace('0', 'nut')
            kind_of_object = kind_of_object.replace('1', 'bolt')
            kind_of_object = kind_of_object.replace('2', 'ring')
            kind_of_object = kind_of_object.replace('3', 'metal attachment')
            kind_of_object = kind_of_object.replace('4', 'valve')


            # Extract bounding boxes of objects out of prediction and calculate its centerpoint for th location
            bounding = result['instances'].pred_boxes.tensor.numpy()
            box = bounding[j]
            centerx,centery = (np.average(box[:2]),np.average(box[2:]))


            # Run color identification on the mask to identify the objects color
            mask5 = mask[j,:,:]
            mask_with_color = cv2.bitwise_and(hsv_image,img, mask=mask5)
            # cv2.namedWindow("Resized_Window", cv2.WINDOW_KEEPRATIO)
            # cv2.imshow("Resized_Window", mask_with_color)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
        
            contours, _ = cv2.findContours(mask5, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            for contour in contours:
                maskr = np.zeros(img.shape[:2], dtype="uint8")
                cv2.drawContours(maskr, [contour], -1, 255, -1)
                pink_mask = cv2.inRange(mask_with_color, lower_pink, upper_pink)
                if np.mean(pink_mask) > 0:
                        match_color = 'pink'
                mean = cv2.mean(img, mask=maskr)[0:3]
                min_rmse = 1000000
                # print(mean)
                for color in colors:
                    bb = color[1][0]
                    gg = color[1][1]
                    rr = color[1][2]
                    rmse = sqrt( ( (mean[2]-rr)*(mean[2]-rr) + (mean[1]-gg)*(mean[1]-gg) + (mean[0]-bb)*(mean[0]-bb) )/3 )
                    
                    if np.mean(pink_mask) > 0:
                        match_color = 'pink'
                    elif rmse < min_rmse:
                        min_rmse = rmse
                        match_color = color[0]
                    
                cv2.putText(img, str(j), (contour[0][0] - 5), fontFace, fontScale, (0,0,0), borderThickness)
                
            with open('./output_data_det.csv', 'a', encoding='UTF8', newline='') as csv_file:
                            csv_writer = csv.writer(csv_file)
                            # for contour in range(len(classes)):
                            csv_writer.writerow((img_name,str(j), kind_of_object, mask.shape[0], (centerx,centery), angle, match_color))

        cv2.imwrite('{}/Image{}-labeled.jpg'.format(SAVE_PATH, i), img)