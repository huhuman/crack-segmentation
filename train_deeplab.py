#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""
DeepLab Training Script.

This script is a simplified version of the training script in detectron2/tools.
"""

import os
import torch

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import DatasetMapper, DatasetCatalog, MetadataCatalog, build_detection_train_loader
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch

from deeplab import add_deeplab_config

from dataset.crack_images import get_cracks_dicts_semantic
from my_trainer import DeepLabv3_Plus_Trainer

'''
python3 train_deeplab.py \
    --config-file ./deeplab/configs/Cityscapes-SemanticSegmentation/deeplab_v3_plus_R_103_os16_mg124_poly_90k_bs16.yaml \
    --tdir /home/aicenter/Documents/hsu/data/CECI_Project/chungliao/semantic/semantic_cropped_fixed_512_Augmented/    \
    --vdir /home/aicenter/Documents/hsu/data/CECI_Project/chungliao/semantic/semantic_val_cropped_fixed_512/ \
    --weight ./deeplab/configs/DeepLabV3+_R103-DC5.pkl --batch 4 --iter 200
'''


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    if args.output_dir != None:
        cfg.OUTPUT_DIR = args.output_dir
    else:
        cfg.OUTPUT_DIR = "DeepLabv3+/"
    add_deeplab_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)
    data_dir = {"train": args.train_dir, "val": args.val_dir}
    for d in ["train", "val"]:
        DatasetCatalog.register(
            "crack_" + d, lambda d=d: get_cracks_dicts_semantic(data_dir[d]))
        MetadataCatalog.get(
            "crack_" + d).set(evaluator_type="sem_seg", stuff_classes=["crack"])

    cfg.MODEL.WEIGHTS = args.model_path

    cfg.SOLVER.IMS_PER_BATCH = args.batch_size
    cfg.SOLVER.BASE_LR = args.lr  # pick a good LR
    cfg.SOLVER.MAX_ITER = args.iteration
    cfg.DATASETS.TRAIN = ("crack_train",)
    cfg.DATASETS.TEST = ("crack_val", )

    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 2
    cfg.MODEL.SEM_SEG_HEAD.LOSS_TYPE = "customed"
    cfg.INPUT.CROP.SIZE = (256, 512)
    cfg.TEST.EVAL_PERIOD = 20

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    trainer = DeepLabv3_Plus_Trainer(cfg)
    trainer.resume_or_load(resume=False)
    return trainer.train()


if __name__ == "__main__":
    parser = default_argument_parser()
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
    # launch(
    #     main,
    #     args.num_gpus,
    #     num_machines=args.num_machines,
    #     machine_rank=args.machine_rank,
    #     dist_url=args.dist_url,
    #     args=(args,),
    # )
