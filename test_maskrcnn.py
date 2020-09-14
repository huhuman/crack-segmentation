# import some common libraries
import numpy as np
import cv2
import os
import argparse

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader
from dataset.crack_images import get_cracks_dicts_instance
from detectron2.engine import DefaultPredictor


def inference(predictor, image_path, mask_color, confidence=0.5):
    im = cv2.imread(image_path)
    outputs = predictor(im)
    masks = outputs['instances'].pred_masks.cpu().numpy()
    masks = masks[outputs['instances'].scores.cpu().numpy() > confidence]
    mask = np.sum(masks, axis=0) > 0
    roi = im[mask]
    im[mask] = ((0.5 * mask_color) + (0.5 * roi)).astype("uint8")
    return im


def test(args):
    test_dir = args.test_path

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(
        "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (crack)
    cfg.MODEL.WEIGHTS = args.model_path
    # set the testing threshold for this model
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    predictor = DefaultPredictor(cfg)

    mask_color = np.array((0, 255, 0), dtype="uint8")

    if os.path.isdir(test_dir):
        save_dir = test_dir + '/result/'
        os.makedirs(save_dir, exist_ok=True)
        DatasetCatalog.clear()
        DatasetCatalog.register(
            "cracks_test", lambda d="test": get_cracks_dicts_instance(test_dir))
        MetadataCatalog.get("cracks_test").set(
            thing_classes=["crack"], thing_colors=[(96, 151, 255)])
        cfg.DATASETS.TEST = ("cracks_test", )
        test_loader = build_detection_test_loader(cfg, "cracks_test")

        for idx, d in enumerate(test_loader):
            file_name = d[0]["file_name"]
            if idx % 25 == 0:
                print('Processing %s...' % (file_name))
            prediction = inference(
                predictor, file_name, mask_color,  cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST)
            cv2.imwrite(save_dir + file_name.split('/')[-1], prediction)
    else:
        save_path = test_dir[:-4] + "_prediction" + test_dir[-4:]
        print('Processing %s...' % (test_dir))
        prediction = inference(
            predictor, test_dir, mask_color,  cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST)
        cv2.imwrite(save_path, prediction)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Mask R-CNN on Detectron2: Code for both training and predicting')
    parser.add_argument('--path', dest='test_path', type=str, default=None, required=True,
                        help='Root directory of testing data (or direct image path). Note that there must be subfolder image/ under the directory.')
    parser.add_argument('--weight', dest='model_path', type=str, required=True,
                        default=None, help='path of weight file.')
    args = parser.parse_args()
    test(args)
