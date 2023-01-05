from typing import final
import warnings

from numpy.lib.type_check import imag
warnings.filterwarnings("ignore")


from classification import classify
from custom_config_detectron import *
import skimage.io
import numpy as np
import sys
import os
import glob
import matplotlib.pyplot as plt
import cv2
import glob


# Root directory of the project
ROOT_DIR = os.path.abspath("../detectron2-final")   # Go one folder up

# Path to trained weights
##WEIGHTS_PATH = os.path.join(ROOT_DIR, "weight/model_last.pth")  # TODO: update this path

# Device to load the neural network on.
# Useful if you're training a model on the same
# machine, in which case use CPU and leave the
# GPU for training.
##DEVICE = "/cpu:0"  # /cpu:0 or /gpu:0

##config = InferenceConfig()
# config.display() # Shows config stats

# LOAD MODEL
# Create model in inference mode
##with tf.device(DEVICE): model = modellib.MaskRCNN(mode="inference", model_dir=DEFAULT_LOGS_DIR,config=config)

# Load COCO weights or load the last model you trained
weights_path = cfg.MODEL.WEIGHTS
cfg.MODEL.DEVICE = "cpu"
# Load weights
print("\nLoading weights", weights_path.split('/')[-1])



import cv2
import glob

images = []

for img in glob.glob(os.path.join(ROOT_DIR,'input','*.jpg')):
      cv_img = cv2.imread(img)
      images.append(cv_img)

print('\nPredicting masks...')

masks = []   

for image in images:
        # Predict masks
        result = predictor(image)
        #extract mask out of result
        mask_array = result['instances'].pred_masks.numpy()
        # Extract masks out of result
        masks.append(mask_array)

print('Done predicting masks.')

# Classify image with predicted mask
print('\nClassifying images...')
classify(images, masks)
print('Done classifying masks.\n')
