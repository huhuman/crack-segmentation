
import os
import torch
import numpy as np
import cv2
import argparse

import detectron2.data.transforms as T
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import DatasetMapper, DatasetCatalog, MetadataCatalog, build_detection_test_loader
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.engine import DefaultPredictor

from deeplab import add_deeplab_config, build_lr_scheduler

from dataset.crack_images import get_cracks_dicts_semantic

'''
python test_deeplab.py --config-file ./deeplab/configs/Cityscapes-SemanticSegmentation/deeplab_v3_plus_R_103_os16_mg124_poly_90k_bs16.yaml\
    --path /home/aicenter/Documents/hsu/data/CECI_Project/chungliao/semantic/semantic_val_cropped_fixed_512/ \
    --weight /home/aicenter/Documents/hsu/crack-detection/DeepLabv3+/model_final.pth
'''


def inference(predictor, image_path, mask_color, confidence=0.5):
    print(image_path)
    im = cv2.imread(image_path)
    outputs = predictor(im)
    masks = outputs['sem_seg'].cpu().numpy()
    masks = np.argmax(masks, axis=0)
    mask = masks > 0
    roi = im[mask]
    im[mask] = ((0.5 * mask_color) + (0.5 * roi)).astype("uint8")
    return im, mask


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


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_deeplab_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # cfg.freeze()
    default_setup(cfg, args)
    return cfg


def test(args):
    test_dir = args.test_path
    assert os.path.exists(test_dir), 'Files not found'

    cfg = setup(args)
    cfg.MODEL.WEIGHTS = args.model_path
    # cfg.SOLVER.IMS_PER_BATCH = args.batch_size
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 2
    cfg.INPUT.CROP.SIZE = (256, 512)
    predictor = DefaultPredictor(cfg)

    mask_color = np.array((0, 255, 0), dtype="uint8")
    if os.path.isdir(test_dir):
        save_dir = test_dir + '/result/'
        if not os.path.exists(test_dir + 'image/'):
            save_dir = test_dir + '../result/'
        os.makedirs(save_dir, exist_ok=True)
        DatasetCatalog.clear()
        DatasetCatalog.register(
            "cracks_test", lambda d="test": get_cracks_dicts_semantic(test_dir))
        # MetadataCatalog.get("cracks_test").set(
        #     thing_classes=["crack"], thing_colors=[(96, 151, 255)])
        cfg.DATASETS.TEST = ("cracks_test", )
        test_loader = build_detection_test_loader(cfg, "cracks_test")
        if args.eval:
            assert os.path.exists(
                test_dir + 'mask/'), print("Files not exist: %s" % (test_dir + 'mask/'))
            from utils.metrics import Evaluator
            evaluator = Evaluator(2)
            evaluator.reset()
            for idx, d in enumerate(test_loader):
                file_name = d[0]["file_name"]
            #     if idx % 25 == 0:
            #         print('Processing %s...' % (file_name))
            #     visualization, mask = inference(
            #         predictor, file_name, mask_color,  cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST)
            #     target = cv2.imread(d[0]['seg_file_name'],
            #                         cv2.IMREAD_GRAYSCALE)
            #     evaluator.add_batch(target/255, mask)
            #     if not args.save:
            #         cv2.imwrite(save_dir + file_name.split('/')
            #                     [-1], visualization)
            # print_evaluation(evaluator)
        else:
            for idx, d in enumerate(test_loader):
                file_name = d[0]["file_name"]
                if idx % 25 == 0:
                    print('Processing %s...' % (file_name))
                visualization, _ = inference(
                    predictor, file_name, mask_color,  cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST)
                cv2.imwrite(save_dir + file_name.split('/')
                            [-1], visualization)
    else:
        save_path = test_dir[:-4] + "_prediction" + test_dir[-4:]
        print('Processing %s...' % (test_dir))
        prediction = inference(
            predictor, test_dir, mask_color,  cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST)
        cv2.imwrite(save_path, prediction)


if __name__ == "__main__":
    parser = default_argument_parser()
    parser.add_argument('--path', dest='test_path', type=str, default=None, required=True,
                        help='Root directory of testing data (or direct image path). Note that there must be subfolder image/ under the directory.')
    parser.add_argument('--weight', dest='model_path', type=str, required=True,
                        default=None, help='path of weight file.')
    parser.add_argument("--eval", action="store_true",
                        help="add it to perform evaluation")
    parser.add_argument("--save", action="store_true",
                        help="add it to save predictions")
    args = parser.parse_args()
    test(args)
