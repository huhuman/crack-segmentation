# import some common libraries
import numpy as np
import cv2
import random
import matplotlib.pyplot as plt
import tensorflow as tf
import datetime
import os
import time

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, transforms
import numpy as np
import json
from detectron2.structures import BoxMode
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg

# TRAIN_DIR = "/home/aicenter/Documents/hsu/data/public_dataset/CrackForest-dataset/"
# TRAIN_DIR = "/home/aicenter/Documents/hsu/data/Crack_Augmented/"
# TRAIN_DIR = "/home/aicenter/Documents/hsu/data/Crack_Augmented_Trans/"
TRAIN_DIR = "/home/aicenter/Documents/hsu/data/public_dataset/crack_segmentation_dataset/"


def get_cracks_dicts(img_dir):
    dataset_dicts = []
    image_paths = list(sorted(os.listdir(img_dir + 'image/')))
    mask_paths = None
    if os.path.exists(img_dir+'mask/'):
        mask_paths = list(sorted(os.listdir(img_dir+'mask/')))
    for idx, v in enumerate(image_paths):
        record = {}

        filename = os.path.join(img_dir, 'image/', v)
        height, width = cv2.imread(filename).shape[:2]

        record["file_name"] = filename
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width

        if mask_paths:
            mask_path = img_dir + "mask/" + mask_paths[idx]
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # HxWxC
            contours, _ = cv2.findContours(
                mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            objs = []
            for cnt in contours:
                approx = cv2.approxPolyDP(
                    cnt, 0.009 * cv2.arcLength(cnt, True), True)
                if approx.shape[0] < 3:
                    continue
                # cv2.drawContours(image, [approx], 0, (0, 0, 255), 1)
                # get bounding box coordinates for each mask
                px = [pos[0][0] for pos in approx]
                py = [pos[0][1] for pos in approx]
                poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
                poly = [p for x in poly for p in x]
                obj = {
                    "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                    "bbox_mode": BoxMode.XYXY_ABS,
                    "segmentation": [poly],
                    "category_id": 0,
                    "iscrowd": 0
                }
                objs.append(obj)
            record["annotations"] = objs
        else:
            record["annotations"] = [{
                "bbox": [0, 0, 0, 0],
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": [(0, 0), (0, 0), (0, 0)],
                "category_id": 0,
                "iscrowd": 0
            }]
        dataset_dicts.append(record)
    return dataset_dicts


def main():
    for d in ["train"]:
        DatasetCatalog.register(
            "crack_" + d, lambda d=d: get_cracks_dicts(TRAIN_DIR))
        MetadataCatalog.get(
            "crack_" + d).set(thing_classes=["crack"], thing_colors=[(96, 151, 255)])

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(
        "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("crack_train",)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml")  # Let training initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.0002  # pick a good LR
    cfg.SOLVER.MAX_ITER = 150000
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (crack)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()
