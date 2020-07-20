
from dataloaders.datasets.crack_images import TestCrackImagesForSegmentation
from dataloaders.utils import decode_seg_map_sequence
from modeling.deeplab import *

import torch
from torchvision.utils import make_grid
from tensorboardX import SummaryWriter

import argparse
from tqdm import tqdm
import numpy as np
import cv2
import os
'''
execute script example:
python test_my_dataset.py --backbone resnet  --workers 4 \
    --use-sbd --batch-size 1 --gpu-ids 0 --checkname deeplab-resnet --dataset customed
'''

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
MODEL_DIR = 'run/customed/deeplab-resnet/CC213_Augmented/model_best.pth.tar'
# TEST_DIR = '/home/aicenter/Documents/hsu/data/test_dataset/test_ceci/640xh/256x256/'
TEST_DIR = '/home/aicenter/Documents/hsu/data/CC213_validation/'


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PyTorch DeeplabV3Plus Training")
    parser.add_argument('--backbone', type=str, default='resnet',
                        choices=['resnet', 'xception', 'drn', 'mobilenet'],
                        help='backbone name (default: resnet)')
    parser.add_argument('--out-stride', type=int, default=16,
                        help='network output stride (default: 8)')
    parser.add_argument('--dataset', type=str, default='pascal',
                        choices=['customed', 'pascal', 'coco', 'cityscapes'],
                        help='dataset name (default: pascal)')
    parser.add_argument('--use-sbd', action='store_true', default=True,
                        help='whether to use SBD dataset (default: True)')
    parser.add_argument('--workers', type=int, default=4,
                        metavar='N', help='dataloader threads')
    parser.add_argument('--sync-bn', type=bool, default=None,
                        help='whether to use sync bn (default: auto)')
    parser.add_argument('--freeze-bn', type=bool, default=False,
                        help='whether to freeze bn parameters (default: False)')
    # training hyper params
    parser.add_argument('--batch-size', type=int, default=None,
                        metavar='N', help='input batch size for \
                                training (default: auto)')
    # cuda, seed and logging
    parser.add_argument('--no-cuda', action='store_true',
                        default=False, help='disables CUDA training')
    parser.add_argument('--gpu-ids', type=str, default='0',
                        help='use which gpu to train, must be a \
                        comma-separated list of integers only (default=0)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    # checking point
    parser.add_argument('--checkname', type=str, default=None,
                        help='set the checkpoint name')

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

    if args.batch_size is None:
        args.batch_size = 4 * len(args.gpu_ids)

    if args.checkname is None:
        args.checkname = 'deeplab-'+str(args.backbone)
    print(args)
    torch.manual_seed(args.seed)
    dataset_test = TestCrackImagesForSegmentation(TEST_DIR)
    test_loader = torch.utils.data.DataLoader(
        dataset_test, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    # Define network
    model = DeepLab(num_classes=dataset_test.NUM_CLASSES,
                    backbone=args.backbone,
                    output_stride=args.out_stride,
                    sync_bn=args.sync_bn,
                    freeze_bn=args.freeze_bn)
    if args.cuda:
        model.load_state_dict(torch.load(MODEL_DIR)['state_dict'])
        model.to(DEVICE)
    model.eval()
    tbar = tqdm(test_loader, desc='\r')
    test_loss = 0.0
    count = 0
    for i, sample in enumerate(tbar):
        image = sample['image']
        if args.cuda:
            image = image.to(DEVICE)
        with torch.no_grad():
            output = model(image)
        pred = output.data.cpu().numpy()
        pred = np.argmax(pred, axis=1)

        save_dir = TEST_DIR + '/result/'
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        original_image = None
        if 'Trans' in TEST_DIR:
            foldername = TEST_DIR.split('/')[-2].split('_')[0]
            original_image = cv2.imread(
                TEST_DIR + '../' + foldername + '/image/' + sample['filename'][0])
        else:
            original_image = cv2.imread(
                TEST_DIR + 'image/' + sample['filename'][0])
        if len(np.unique(pred)) > 1:
            color = np.array((0, 0, 255), dtype="uint8")
            mask = pred[0] == 1
            roi = original_image[mask]
            original_image[mask] = (
                (0.5 * color) + (0.5 * roi)).astype("uint8")

        cv2.imwrite(save_dir + sample['filename']
                    [0], original_image)  # batch_size =1
