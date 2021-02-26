# import some common libraries
import numpy as np
import cv2
import os
import argparse

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from dataset.crack_images import get_cracks_dicts_instance
from my_trainer import CrackTrainer

'''
python3 train_maskrcnn.py --tdir '/home/aicenter/Documents/hsu/data/CECI_Project/chungliao/cropped_fixed_512_Augmented/' \
    --vdir '/home/aicenter/Documents/hsu/data/CECI_Project/chungliao/val_cropped_fixed_512/' \
    --weight '/home/aicenter/Documents/hsu/crack-detection/models/CC140+73+22_Augmented/model_final.pth' \
    --output 'chungliao-finetune/' --iter 50000 --batch 4
'''


def main(args):
    data_dir = {"train": args.train_dir, "val": args.val_dir}
    for d in ["train", "val"]:
        DatasetCatalog.register(
            "crack_" + d, lambda d=d: get_cracks_dicts_instance(data_dir[d]))
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
        "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml") if args.model_path == None else args.model_path
    cfg.SOLVER.IMS_PER_BATCH = args.batch_size
    cfg.SOLVER.BASE_LR = args.lr
    cfg.SOLVER.MAX_ITER = args.iteration
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512  # ??? need to discuss further
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.TEST.EVAL_PERIOD = 20
    #### end of setting ####
    if args.output_dir != None:
        cfg.OUTPUT_DIR = args.output_dir
    else:
        cfg.OUTPUT_DIR = "Mask R-CNN/"
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    trainer = CrackTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Mask R-CNN on Detectron2: Code for both training and predicting')
    parser.add_argument('--tdir', dest='train_dir', type=str, default=None, required=True,
                        help='root directory of training data. Note that there must be subfolder image/ and mask/ under the directory.')
    parser.add_argument('--vdir', dest='val_dir', type=str, default=None, required=True,
                        help='root directory of validation data. Note that there must be subfolder image/ and mask/ under the directory.')
    parser.add_argument('--weight', dest='model_path', type=str,
                        default=None, help='path of weight file.')
    parser.add_argument('--lr', dest='lr', type=float,
                        default=2e-4, help='learning rate')
    parser.add_argument('--batch', dest='batch_size',
                        type=int, default=2, help='batch size')
    parser.add_argument('--iter', dest='iteration',
                        type=int, default=100, help='number of iteratinon(s)')
    parser.add_argument('--output', dest='output_dir',
                        type=str, default=None, help='path for saving the model')
    args = parser.parse_args()
    print("Command Line Args:", args)
    main(args)
