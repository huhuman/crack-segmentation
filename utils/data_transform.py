import cv2
import os
import numpy as np


def TRANSFORM_MASK_Binary2RGB(mask_file_path):
    mask = cv2.imread(mask_file_path, cv2.IMREAD_GRAYSCALE)
    mask[mask > 0] = 255
    cv2.imwrite(mask_file_path.split('/')[-1], mask)


def TRANSFORM_MASK_RGB2Binary(mask_file_path):
    mask = cv2.imread(mask_file_path, cv2.IMREAD_GRAYSCALE)
    mask = mask/255
    save_dir = os.path.join(
        '/'.join(mask_file_path.split('/')[:-1]), '../semantic_mask/')
    os.makedirs(save_dir, exist_ok=True)
    cv2.imwrite(save_dir + mask_file_path.split('/')[-1], mask)


if __name__ == "__main__":
    root = '/home/aicenter/Documents/hsu/data/CECI_Project/chungliao/val_cropped_fixed_512/mask/'
    for filename in os.listdir(root):
        TRANSFORM_MASK_RGB2Binary(root + filename)
