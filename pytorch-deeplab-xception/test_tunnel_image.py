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
python test_tunnel_image.py --backbone resnet --use-sbd --gpu-ids 0
'''

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
MODEL_DIR = 'run/customed/deeplab-resnet-weight/CC140+73+22_Augmented/checkpoint.pth.tar'
CROP_SIZE = (256, 256)  # (w, h)
# TUNNEL_IMAGE_PATH = '/home/aicenter/Documents/hsu/data/test_dataset/test_tunnel/3yi_tunnel.png'
TUNNEL_IMAGE_PATH = '/home/aicenter/Documents/hsu/data/test_dataset/test_tunnel/t3_tunnel.png'


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

    model.eval()
    tunnel_image = cv2.imread(TUNNEL_IMAGE_PATH)
    height, width, _ = tunnel_image.shape

    crop_width, crop_height = CROP_SIZE
    x_num = int(width/crop_width) + 1
    y_num = int(height/crop_height) + 1
    max_x = width - crop_width
    max_y = height - crop_height
    for x in range(x_num):
        for y in range(y_num):
            start_x = min(x*crop_width, max_x)
            end_x = start_x + crop_width
            start_y = min(y*crop_height, max_y)
            end_y = start_y + crop_height
            test_image = tunnel_image[start_y:end_y, start_x:end_x, :]
            test_image = Image.fromarray(
                cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB))
            inputs = transform_val(test_image)
            inputs = inputs.unsqueeze(0)

            if args.cuda:
                inputs = inputs.to(DEVICE)
            with torch.no_grad():
                output = model(inputs)
            pred = output.data.cpu().numpy()
            pred = np.argmax(pred, axis=1)

            if len(np.unique(pred)) > 1:
                test_image = np.array(test_image)
                color = np.array((255, 0, 0), dtype="uint8")  # RGB
                mask = pred[0] == 1
                roi = test_image[mask]
                test_image[mask] = (
                    (0.5 * color) + (0.5 * roi)).astype("uint8")

            tunnel_image[start_y:end_y, start_x:end_x, :] = cv2.cvtColor(
                np.asarray(test_image), cv2.COLOR_RGB2BGR)

    save_path = TUNNEL_IMAGE_PATH.split('.png')[0] + '_seg.png'
    print('Saved at', save_path)
    cv2.imwrite(save_path, tunnel_image)
