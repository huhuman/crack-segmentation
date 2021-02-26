# import some common libraries
import numpy as np
import cv2
import os
import sys

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader
from dataset.crack_images import get_cracks_dicts_instance
from detectron2.engine import DefaultPredictor


def inference(predictor, image_path, mask_color, confidence=0.5, isgray=True):
    im = None
    if isgray:
        im = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    else:
        im = cv2.imread(image_path, )
    outputs = predictor(im)
    masks = outputs['instances'].pred_masks.cpu().numpy()
    masks = masks[outputs['instances'].scores.cpu().numpy() > confidence]
    mask = np.sum(masks, axis=0) > 0
    im = cv2.imread(image_path, )
    roi = im[mask]
    im[mask] = ((0.5 * mask_color) + (0.5 * roi)).astype("uint8")
    return im, mask


def resizeImageAndWrite(im, file_path):
    size_list = [(1000, 600), (250, 150)]
    export_path = ['/home/aicenter/Documents/hsu/demo/server/images/original/',
                   '/home/aicenter/Documents/hsu/demo/server/images/thumbnail/']

    h, w, _ = im.shape
    for target_size, path in zip(size_list, export_path):
        img = im.copy()
        ratio_h = int(target_size[0]/w*h)
        img = cv2.resize(img, (target_size[0], ratio_h))
        crop_h_upper = min(ratio_h, int(ratio_h/2 + target_size[1]/2))
        crop_h_lower = max(0, int(ratio_h/2 - target_size[1]/2))
        print(crop_h_upper, crop_h_lower)
        img = img[crop_h_lower:crop_h_upper, :, :]
        filename = file_path.split('/')[-1]
        save_name = filename.split('.')[0] + '_%sx%s.%s' % (
            target_size[0], target_size[1], filename.split('.')[-1])
        sid = len(os.listdir(path)) + 1
        save_path = '%s/%03d.%s' % (path, sid, save_name.split('.')[-1])
        print(save_path)
        cv2.imwrite(save_path, img)


def test(test_dir):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(
        "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (crack)
    cfg.MODEL.WEIGHTS = '/home/aicenter/Documents/hsu/crack-detection/models/Mask R-CNN/CC140+73+22_Augmented/model_final.pth'
    # set the testing threshold for this model
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    predictor = DefaultPredictor(cfg)

    mask_color = np.array((0, 255, 0), dtype="uint8")

    print('Processing %s...' % (test_dir))
    prediction, _ = inference(
        predictor, test_dir, mask_color,  cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST)
    resizeImageAndWrite(prediction, test_dir)


if __name__ == "__main__":
    test(sys.argv[1])
