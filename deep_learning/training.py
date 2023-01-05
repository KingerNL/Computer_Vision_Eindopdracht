import os
import sys
import detectron2
from detectron2.utils.logger import setup_logger

# import some common libraries
import numpy as np
import cv2
import random

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.data.catalog import DatasetCatalog
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg

# Root directory of the project
ROOT_DIR = os.path.join("../detectron2-final") # Go one folder up

#dataset 
from detectron2.data.datasets import register_coco_instances
register_coco_instances("my_dataset_train", {}, os.path.join(ROOT_DIR, 'train', "train.json"),os.path.join(ROOT_DIR,'train'))
register_coco_instances("my_dataset_val", {}, os.path.join(ROOT_DIR,'valid', "valid.json"),os.path.join(ROOT_DIR,'valid'))

#config file for model.
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
cfg.MODEL.DEVICE = "cpu" #kan je uitcommenten als je een CUDA GPU hebt, maar heeft voor predicten geen toegevoegde waarden om het via de GPU te doen


weights_path = cfg.MODEL.WEIGHTS

print("Loading weights ", weights_path)

if __name__ == '__main__':
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg) 
    trainer.resume_or_load(resume=True) #resume staat nu op true --> opnieuw trainen moet je hem op False zetten
    trainer.train()
