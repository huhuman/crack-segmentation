#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""
DeepLab Training Script.

This script is a simplified version of the training script in detectron2/tools.
"""

import os
import torch

import detectron2.data.transforms as T
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import DatasetMapper, DatasetCatalog, MetadataCatalog, build_detection_train_loader
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.evaluation import CityscapesSemSegEvaluator, DatasetEvaluators, SemSegEvaluator

from deeplab import add_deeplab_config, build_lr_scheduler

from dataset.crack_images import get_cracks_dicts_semantic

'''
python3 train_deeplab.py \
    --config-file ./deeplab/configs/Cityscapes-SemanticSegmentation/deeplab_v3_plus_R_103_os16_mg124_poly_90k_bs16.yaml \
    --tdir /home/aicenter/Documents/hsu/data/CECI_Project/chungliao/cropped_fixed_512/ \
    --vdir /home/aicenter/Documents/hsu/data/CECI_Project/chungliao/val_cropped_fixed_512/ \
    --weight ./deeplab/configs/DeepLabV3+_R103-DC5.pkl --batch 2
'''


def build_sem_seg_train_aug(cfg):
    augs = [
        T.ResizeShortestEdge(
            cfg.INPUT.MIN_SIZE_TRAIN, cfg.INPUT.MAX_SIZE_TRAIN, cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING
        )
    ]
    if cfg.INPUT.CROP.ENABLED:
        augs.append(
            T.RandomCrop_CategoryAreaConstraint(
                cfg.INPUT.CROP.TYPE,
                cfg.INPUT.CROP.SIZE,
                cfg.INPUT.CROP.SINGLE_CATEGORY_MAX_AREA,
                cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
            )
        )
    augs.append(T.RandomFlip())
    return augs


class Trainer(DefaultTrainer):
    """
    We use the "DefaultTrainer" which contains a number pre-defined logic for
    standard training workflow. They may not work for you, especially if you
    are working on a new research project. In that case you can use the cleaner
    "SimpleTrainer", or write your own training loop.
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each builtin dataset.
        For your own dataset, you can simply create an evaluator manually in your
        script and do not have to worry about the hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        if evaluator_type == "sem_seg":
            return SemSegEvaluator(
                dataset_name,
                distributed=True,
                num_classes=cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
                ignore_label=cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
                output_dir=output_folder,
            )
        if evaluator_type == "cityscapes_sem_seg":
            assert (
                torch.cuda.device_count() >= comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesSemSegEvaluator(dataset_name)
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        if len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)

    @classmethod
    def build_train_loader(cls, cfg):
        if "SemanticSegmentor" in cfg.MODEL.META_ARCHITECTURE:
            mapper = DatasetMapper(
                cfg, is_train=True, augmentations=build_sem_seg_train_aug(cfg))
        else:
            mapper = None
        return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        """
        It now calls :func:`detectron2.solver.build_lr_scheduler`.
        Overwrite it if you'd like a different scheduler.
        """
        return build_lr_scheduler(cfg, optimizer)


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
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 2
    data_dir = {"train": args.train_dir, "val": args.val_dir}
    for d in ["train", "val"]:
        DatasetCatalog.register(
            "crack_" + d, lambda d=d: get_cracks_dicts_semantic(data_dir[d]))
        MetadataCatalog.get(
            "crack_" + d).set(evaluator_type="sem_seg")

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        return res

    if args.model_path:
        cfg.MODEL.WEIGHTS = args.model_path

    cfg.SOLVER.IMS_PER_BATCH = args.batch_size
    cfg.SOLVER.BASE_LR = args.lr  # pick a good LR
    cfg.SOLVER.MAX_ITER = args.iteration
    cfg.DATASETS.TRAIN = ("crack_train",)
    cfg.DATASETS.TEST = ("crack_val", )
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    trainer = Trainer(cfg)
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
