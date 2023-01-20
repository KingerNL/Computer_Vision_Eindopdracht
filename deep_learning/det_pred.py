from typing import final
import warnings

from numpy.lib.type_check import imag
warnings.filterwarnings("ignore")

from det_clas import classify
from custom_config_detectron2 import *
import numpy as np
import sys
import os
import glob
import matplotlib.pyplot as plt
import cv2
import glob


# Root directory of the projects
ROOT_DIR = os.path.abspath("../deep_learning")   # Go one folder up
# Path to trained weights
WEIGHTS_PATH = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
# set model device
cfg.MODEL.DEVICE = "cpu"
# Load weights
print("\nLoading weights", WEIGHTS_PATH.split('/')[-1])

images = []
image_names = []

for img in glob.glob(os.path.join(ROOT_DIR,'test_set','*.jpg')):
      cv_img = cv2.imread(img)
      img_namee = os.path.basename(img)
      images.append(cv_img)
      image_names.append(img_namee)

# Classify image with predicted mask
print('\nLabeling images...')
classify(images, image_names)
print('Done\n')

