# import some common libraries
import numpy as np
import cv2
import tensorflow as tf
import datetime
import os
import time

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import hooks
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import DatasetCatalog, MetadataCatalog
from dataset.crack_images import get_cracks_dicts
from my_trainer import CrackTrainer

DATA_DIR = {"train": "/home/aicenter/Documents/hsu/data/CC213_Augmented/",
            "val": "/home/aicenter/Documents/hsu/data/CC213_validation/"}


def main():
    for d in ["train", "val"]:
        DatasetCatalog.register(
            "crack_" + d, lambda d=d: get_cracks_dicts(DATA_DIR[d]))
        MetadataCatalog.get(
            "crack_" + d).set(thing_classes=["crack"], thing_colors=[(96, 151, 255)])

    #### training parameters ####
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(
        "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("crack_train",)
    cfg.DATASETS.TEST = ("crack_val", )
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml")  # Let training initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.0002  # pick a good LR
    cfg.SOLVER.MAX_ITER = 100
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512  # ??? need to discuss further
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.TEST.EVAL_PERIOD = 20
    #### end of setting ####

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    trainer = CrackTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()


if __name__ == "__main__":
    main()
