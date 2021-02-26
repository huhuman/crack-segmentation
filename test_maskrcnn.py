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

'''
python3 test_maskrcnn.py --path "/home/aicenter/Documents/hsu/data/CECI_Project/chungliao/val_cropped_fixed_512/" \
    --weight "/home/aicenter/Documents/hsu/crack-detection/models/Mask R-CNN/ongoing_experiments/chungliao/model_final.pth"

python3 test_maskrcnn.py --path "/home/aicenter/Documents/hsu/data/CECI_Project/chungliao/val_cropped_fixed_512/" \
    --weight "/home/aicenter/Documents/hsu/crack-detection/models/Mask R-CNN/ongoing_experiments/chungliao-finetune/model_final.pth"
'''


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


def print_evaluation(evaluator):
    MIoU = np.diag(evaluator.confusion_matrix) / (
        np.sum(evaluator.confusion_matrix, axis=1) + np.sum(evaluator.confusion_matrix, axis=0) -
        np.diag(evaluator.confusion_matrix))
    Acc = evaluator.Pixel_Accuracy()
    Acc_class = evaluator.Pixel_Accuracy_Class()
    mIoU = evaluator.Mean_Intersection_over_Union()
    FWIoU = evaluator.Frequency_Weighted_Intersection_over_Union()
    print('Validation:')
    print('mIoU:{}'.format(MIoU))
    print("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}".format(
        Acc, Acc_class, mIoU, FWIoU))


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
        if not os.path.exists(test_dir + 'image/'):
            save_dir = test_dir + '../result/'
        os.makedirs(save_dir, exist_ok=True)
        DatasetCatalog.clear()
        DatasetCatalog.register(
            "cracks_test", lambda d="test": get_cracks_dicts_instance(test_dir))
        MetadataCatalog.get("cracks_test").set(
            thing_classes=["crack"], thing_colors=[(96, 151, 255)])
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
                if idx % 25 == 0:
                    print('Processing %s...' % (file_name))
                visualization, mask = inference(
                    predictor, file_name, mask_color,  cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST)
                target = cv2.imread(d[0]['seg_file_name'],
                                    cv2.IMREAD_GRAYSCALE)
                evaluator.add_batch(target/255, mask)
                if not args.save:
                    cv2.imwrite(save_dir + file_name.split('/')
                                [-1], visualization)
            print_evaluation(evaluator)
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
        prediction, _ = inference(
            predictor, test_dir, mask_color,  cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST)

        cv2.imwrite(save_path, prediction)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Mask R-CNN on Detectron2: Code for both training and predicting')
    parser.add_argument('--path', dest='test_path', type=str, default=None, required=True,
                        help='Root directory of testing data (or direct image path). Note that there must be subfolder image/ under the directory.')
    parser.add_argument('--weight', dest='model_path', type=str, required=True,
                        default=None, help='path of weight file.')
    parser.add_argument("--eval", action="store_true",
                        help="add it to perform evaluation")
    parser.add_argument("--save", action="store_true",
                        help="add it to save predictions")
    parser.add_argument("--crop", action="store_true",
                        help="add it to crop input image into 512x512 sub-images")
    args = parser.parse_args()
    test(args)
