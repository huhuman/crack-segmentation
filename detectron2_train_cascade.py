#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Detection Training Script.
This scripts reads a given config file and runs the training or evaluation.
It is an entry point that is made to train standard models in detectron2.
In order to let one script support training of many models,
this script contains logic that are specific to these built-in models and therefore
may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".
Therefore, we recommend you to use detectron2 as an library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other customizations.
"""

import logging
import os
from collections import OrderedDict
import torch

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.config import CfgNode as CN
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch
from dataset.crack_images import get_cracks_dicts
from detectron2.evaluation import verify_results
from my_trainer import CascadeTrainer

'''
python detectron2_train_cascade.py  --num-gpus 1 --config-file ./models/Cascade_MaskRCNN/mask_cascade_rcnn_ResNeSt_200_FPN_dcn_syncBN_all_tricks_3x.yaml
'''

DATA_DIR = {"train": "/home/aicenter/Documents/hsu/data/EWSHM/exp3_Augmented/",
            "val": "/home/aicenter/Documents/hsu/data/EWSHM/exp3/"}


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)
    cfg.OUTPUT_DIR = './cascade_output'

    if args.eval_only:
        model = CascadeTrainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = CascadeTrainer.test(cfg, model)
        if cfg.TEST.AUG.ENABLED:
            res.update(CascadeTrainer.test_with_TTA(cfg, model))
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    for d in ["train", "val"]:
        DatasetCatalog.register(
            "crack_" + d, lambda d=d: get_cracks_dicts(DATA_DIR[d]))
        MetadataCatalog.get("crack_" + d).set(thing_classes=["crack"])
    cfg.DATASETS.TRAIN = ("crack_train",)
    cfg.DATASETS.TEST = ("crack_val", )
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = "https://s3.us-west-1.wasabisys.com/resnest/detectron/mask_cascade_rcnn_ResNeSt_200_FPN_dcn_syncBN_all_tricks_3x-e1901134.pth"
    cfg.SOLVER.IMS_PER_BATCH = 1
    cfg.SOLVER.MAX_ITER = 10000
    cfg.TEST.EVAL_PERIOD = 2000
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    """
    If you'd like to do anything fancier than the standard training logic,
    consider writing your own training loop or subclassing the trainer.
    """
    trainer = CascadeTrainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    if cfg.TEST.AUG.ENABLED:
        trainer.register_hooks(
            [hooks.EvalHook(
                0, lambda: trainer.test_with_TTA(cfg, trainer.model))]
        )
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
