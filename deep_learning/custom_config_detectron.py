import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
 
import numpy as np
import cv2
import random
import os
 
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog

#training
from detectron2.data.catalog import DatasetCatalog
from detectron2.engine import DefaultTrainer

from detectron2.engine import DefaultPredictor

# Root directory of the project
ROOT_DIR = os.path.abspath("../detectron2-final") # Go one folder up

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("my_dataset_train",)
cfg.DATASETS.TEST = ("my_dataset_val",)
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025
cfg.SOLVER.MAX_ITER = 4000
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
 
cfg.MODEL.WEIGHTS = os.path.join(ROOT_DIR, "weight/model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5   # set the testing threshold for this model
cfg.DATASETS.TEST = (os.path.join(ROOT_DIR, "test"), )
cfg.MODEL.DEVICE = "cpu"         #kan je uitcommenten als je een CUDA GPU hebt, maar heeft voor predicten geen toegevoegde waarden om het via de GPU te doen
predictor = DefaultPredictor(cfg)

test_metadata = MetadataCatalog.get(os.path.join(ROOT_DIR, "test"))