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
import numpy as np
from PIL import Image

from detectron2.data import DatasetCatalog
from detectron2.utils.events import get_event_storage
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import DatasetEvaluator
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling import GeneralizedRCNNWithTTA
from utils.metrics import Evaluator

from fvcore.common.file_io import PathManager

# DeepLab
import detectron2.data.transforms as T
from detectron2.evaluation import DatasetEvaluators, SemSegEvaluator
from detectron2.data import DatasetMapper, MetadataCatalog, build_detection_train_loader
from deeplab import add_deeplab_config, build_lr_scheduler


class CrackInstanceSegEvaluator(DatasetEvaluator):

    def __init__(self, dataset_name, distributed, num_classes, conf_threshold=0.5, output_dir=None):
        """
        Args:
            dataset_name (str): name of the dataset to be evaluated.
            distributed (True): if True, will collect results from all ranks for evaluation.
                Otherwise, will evaluate the results in the current process.
            num_classes (int): number of classes
            ignore_label (int): value in semantic segmentation ground truth. Predictions for the
            corresponding pixels should be ignored.
            output_dir (str): an output directory to dump results.
        """
        self._dataset_name = dataset_name
        self._output_dir = output_dir
        self._num_classes = num_classes
        self._N = num_classes + 1
        self._conf_threshold = conf_threshold
        self.confusion_matrix = np.zeros((self._N, self._N), dtype=np.int64)

        self._cpu_device = torch.device("cpu")
        self._logger = logging.getLogger(__name__)

        def image_path_to_mask(path):
            split_path = path.split('/')
            return '/'.join(split_path[:-2]) + '/mask/' + split_path[-1]
        self.input_file_to_gt_file = {
            dataset_record["file_name"]: image_path_to_mask(
                dataset_record["file_name"])
            for dataset_record in DatasetCatalog.get(dataset_name)
        }

    def reset(self):
        self.confusion_matrix = np.zeros((self._N, self._N), dtype=np.int64)

    def process(self, inputs, outputs):
        for input, output in zip(inputs, outputs):
            masks = output['instances'].pred_masks.cpu().numpy()
            masks = masks[output['instances'].scores.cpu().numpy()
                          > self._conf_threshold]
            pre_image = (np.sum(masks, axis=0) > 0).astype(int)
            pre_image = pre_image.reshape(-1)
            with PathManager.open(self.input_file_to_gt_file[input["file_name"]], "rb") as f:
                gt_image = (np.array(Image.open(f).convert('L')) /
                            255).astype(int)
                gt_image = gt_image.reshape(-1)

            assert gt_image.shape == pre_image.shape
            self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self._N)
        label = self._N * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self._N**2)
        confusion_matrix = count.reshape(self._N, self._N)
        return confusion_matrix

    def Pixel_Accuracy(self):
        Acc = np.diag(self.confusion_matrix).sum() / \
            self.confusion_matrix.sum()
        return Acc

    def Pixel_Accuracy_Class(self):
        Acc = np.diag(self.confusion_matrix) / \
            self.confusion_matrix.sum(axis=1)
        Acc = np.nanmean(Acc)
        return Acc

    def Mean_Intersection_over_Union(self):
        MIoU = np.diag(self.confusion_matrix) / (
            np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
            np.diag(self.confusion_matrix))
        MIoU = np.nanmean(MIoU)
        return MIoU

    def Frequency_Weighted_Intersection_over_Union(self):
        freq = np.sum(self.confusion_matrix, axis=1) / \
            np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
            np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
            np.diag(self.confusion_matrix))

        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def evaluate(self):
        res = {}
        res["Acc"] = self.Pixel_Accuracy()
        res["Acc_class"] = self.Pixel_Accuracy_Class()
        res["mIoU"] = self.Mean_Intersection_over_Union()
        res["FWIoU"] = self.Frequency_Weighted_Intersection_over_Union()

        results = OrderedDict({"sem_seg": res})
        self._logger.info(results)
        return results


class CrackTrainer(DefaultTrainer):
    """
    We use the "DefaultTrainer" which contains pre-defined default logic for
    standard training workflow. They may not work for you, especially if you
    are working on a new research project. In that case you can write your
    own training loop. You can use "tools/plain_train_net.py" as an example.
    """
    @classmethod
    def build_evaluator(cls, cfg, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return CrackInstanceSegEvaluator(
            cfg.DATASETS.TEST[0],
            distributed=True,
            num_classes=1,
            conf_threshold=cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST,
            output_dir=output_folder,
        )


class CrackSemanticSegEvaluator(DatasetEvaluator):

    def __init__(self, dataset_name, distributed, num_classes, output_dir=None):
        """
        Args:
            dataset_name (str): name of the dataset to be evaluated.
            distributed (True): if True, will collect results from all ranks for evaluation.
                Otherwise, will evaluate the results in the current process.
            num_classes (int): number of classes
            ignore_label (int): value in semantic segmentation ground truth. Predictions for the
            corresponding pixels should be ignored.
            output_dir (str): an output directory to dump results.
        """
        self._dataset_name = dataset_name
        self._output_dir = output_dir
        self._num_classes = num_classes
        self._N = num_classes + 1
        self.confusion_matrix = np.zeros((self._N, self._N), dtype=np.int64)

        self._cpu_device = torch.device("cpu")
        self._logger = logging.getLogger(__name__)

        def image_path_to_mask(path):
            split_path = path.split('/')
            return '/'.join(split_path[:-2]) + '/mask/' + split_path[-1]
        self.input_file_to_gt_file = {
            dataset_record["file_name"]: image_path_to_mask(
                dataset_record["file_name"])
            for dataset_record in DatasetCatalog.get(dataset_name)
        }

    def reset(self):
        self.confusion_matrix = np.zeros((self._N, self._N), dtype=np.int64)

    def process(self, inputs, outputs):
        for input, output in zip(inputs, outputs):
            masks = output['sem_seg'].cpu().numpy()
            masks = np.argmax(masks, axis=0)
            pre_image = masks.reshape(-1)
            with PathManager.open(self.input_file_to_gt_file[input["file_name"]], "rb") as f:
                gt_image = (np.array(Image.open(f).convert('L'))).astype(int)
                gt_image = gt_image.reshape(-1)
            # print(np.unique(gt_image, return_counts=True))
            # print(np.unique(pre_image, return_counts=True))
            assert gt_image.shape == pre_image.shape
            self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self._N)
        label = self._N * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self._N**2)
        confusion_matrix = count.reshape(self._N, self._N)
        return confusion_matrix

    def Pixel_Accuracy(self):
        Acc = np.diag(self.confusion_matrix).sum() / \
            self.confusion_matrix.sum()
        return Acc

    def Pixel_Accuracy_Class(self):
        Acc = np.diag(self.confusion_matrix) / \
            self.confusion_matrix.sum(axis=1)
        Acc = np.nanmean(Acc)
        return Acc

    def Mean_Intersection_over_Union(self):
        MIoU = np.diag(self.confusion_matrix) / (
            np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
            np.diag(self.confusion_matrix))
        MIoU = np.nanmean(MIoU)
        return MIoU

    def Frequency_Weighted_Intersection_over_Union(self):
        freq = np.sum(self.confusion_matrix, axis=1) / \
            np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
            np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
            np.diag(self.confusion_matrix))

        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def evaluate(self):
        res = {}
        res["Acc"] = self.Pixel_Accuracy()
        res["Acc_class"] = self.Pixel_Accuracy_Class()
        res["mIoU"] = self.Mean_Intersection_over_Union()
        res["FWIoU"] = self.Frequency_Weighted_Intersection_over_Union()

        results = OrderedDict({"sem_seg": res})
        self._logger.info(results)
        return results


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


class DeepLabv3_Plus_Trainer(DefaultTrainer):
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

        return CrackSemanticSegEvaluator(
            cfg.DATASETS.TEST[0],
            distributed=True,
            num_classes=1,
            output_dir=output_folder,
        )

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
