# import some common libraries
import numpy as np
import cv2
import os
import sys
from datetime import datetime

from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader
from detectron2.structures import BoxMode
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.structures.instances import Instances


def inference(cfg, predictor, image_path):
    # cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST
    im = cv2.imread(image_path)
    outputs = predictor(im)
    v = Visualizer(im[:, :, ::-1],
                   MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
                   scale=1,
                   instance_mode=ColorMode.IMAGE
                   )
    pred_outputs = outputs["instances"].to("cpu")
    person_detections = pred_outputs[pred_outputs.pred_classes == 0]
    out = v.draw_instance_predictions(person_detections)
    return out.get_image()[:, :, ::-1], len(person_detections)


def test(test_dir, usrname):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(
        "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml")
    # set the testing threshold for this model
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    predictor = DefaultPredictor(cfg)
    print('Processing %s...' % (test_dir))
    prediction, numbers = inference(cfg, predictor, test_dir)
    print(numbers)
    file_savename = '/home/aicenter/Documents/hsu/demo/game-server/images/result/%s_%s_%s.jpg' % (
        datetime.now().strftime("%H:%M"), usrname, numbers)
    cv2.imwrite(file_savename, prediction)


if __name__ == "__main__":
    test(sys.argv[1], sys.argv[2])
