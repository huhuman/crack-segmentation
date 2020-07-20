from dataloaders.utils import decode_seg_map_sequence
from modeling.deeplab import *

import torch
from torchvision.utils import make_grid
from torchvision import transforms
import argparse

import numpy as np
from PIL import Image
import cv2
import os
'''
execute script example:
python test_tunnel.py --backbone resnet --use-sbd --gpu-ids 0
'''

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
MODEL_DIR = 'run/customed/deeplab-resnet/CC200_Augmented/model_best.pth.tar'
CROP_SIZE = (256, 256)  # (w, h)
TEST_DIR = '/home/aicenter/Documents/hsu/data/test_dataset/test_ceci/640xh/%sx%s/image/' % (
    CROP_SIZE[0], CROP_SIZE[1])
SOURCE_IMAGE_PATH = '/home/aicenter/Documents/hsu/data/test_dataset/test_ceci/640xh/gt_640.png'


def transform_val(image):
    composed_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return composed_transforms(image)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PyTorch DeeplabV3Plus Training")
    parser.add_argument('--backbone', type=str, default='resnet',
                        choices=['resnet', 'xception', 'drn', 'mobilenet'],
                        help='backbone name (default: resnet)')
    parser.add_argument('--out-stride', type=int, default=16,
                        help='network output stride (default: 8)')
    parser.add_argument('--use-sbd', action='store_true', default=True,
                        help='whether to use SBD dataset (default: True)')
    parser.add_argument('--sync-bn', type=bool, default=None,
                        help='whether to use sync bn (default: auto)')
    parser.add_argument('--freeze-bn', type=bool, default=False,
                        help='whether to freeze bn parameters (default: False)')
    # cuda, seed and logging
    parser.add_argument('--no-cuda', action='store_true',
                        default=False, help='disables CUDA training')
    parser.add_argument('--gpu-ids', type=str, default='0',
                        help='use which gpu to train, must be a \
                        comma-separated list of integers only (default=0)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    if args.cuda:
        try:
            args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
        except ValueError:
            raise ValueError(
                'Argument --gpu_ids must be a comma-separated list of integers only')

    if args.sync_bn is None:
        if args.cuda and len(args.gpu_ids) > 1:
            args.sync_bn = True
        else:
            args.sync_bn = False

    print(args)
    torch.manual_seed(args.seed)
    # Define network
    model = DeepLab(num_classes=2,
                    backbone=args.backbone,
                    output_stride=args.out_stride,
                    sync_bn=args.sync_bn,
                    freeze_bn=args.freeze_bn)
    if args.cuda:
        model.load_state_dict(torch.load(MODEL_DIR)['state_dict'])
        model.to(DEVICE)

    # preparation for tunnel image
    height, width, _ = cv2.imread(SOURCE_IMAGE_PATH).shape
    x_num, y_num = int(width/CROP_SIZE[0]), int(height/CROP_SIZE[1])
    whole_image = np.array([])
    v_im = np.array([])

    model.eval()
    for filename in sorted(os.listdir(TEST_DIR)):
        count_id = int(filename.split('.')[0])
        image_path = TEST_DIR+filename
        inputs = Image.open(image_path).convert('RGB')
        inputs = transform_val(inputs)
        inputs = inputs.unsqueeze(0)

        if args.cuda:
            inputs = inputs.to(DEVICE)
        with torch.no_grad():
            output = model(inputs)
        pred = output.data.cpu().numpy()
        pred = np.argmax(pred, axis=1)

        image = cv2.imread(image_path)
        if len(np.unique(pred)) > 1:
            color = np.array((0, 0, 255), dtype="uint8")
            mask = pred[0] == 1
            roi = image[mask]
            image[mask] = (
                (0.5 * color) + (0.5 * roi)).astype("uint8")

        if (count_id+1) % y_num == 0:
            whole_image = v_im if whole_image.size == 0 else np.hstack(
                (whole_image, v_im))
            v_im = np.array([])
        else:
            v_im = image if v_im.size == 0 else np.vstack((v_im, image))

    save_path = SOURCE_IMAGE_PATH.split('.png')[0] + '_seg_deeplab.png'
    print('Saved at', save_path)
    cv2.imwrite(save_path, whole_image)
