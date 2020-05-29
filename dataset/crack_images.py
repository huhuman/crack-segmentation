import torch
from torch.utils.data import Dataset
import cv2
import os
import numpy as np


def isimage(filename):
    IMAGE_FORMAT = [
        'jpg', 'png', 'JPEG', 'PNG', 'jpeg'
    ]
    return filename.split('.')[-1] in IMAGE_FORMAT


class CrackImagesForSegmentation(Dataset):
    """Created for Crackdataset"""

    NUM_CLASSES = 1

    def __init__(self, root_dir=None, transforms=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root = root_dir
        self.imgs = list(sorted(os.listdir(root_dir+'image/')))
        self.masks = list(sorted(os.listdir(root_dir+'mask/')))
        self.transforms = transforms

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_path = self.root + "image/" + self.imgs[idx]
        mask_path = self.root + "mask/" + self.masks[idx]
        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path)/255  # HxWxC
        obj_ids = np.unique(mask)
        obj_ids = obj_ids[:1]
        masks = mask == obj_ids[:, None, None]

        # get bounding box coordinates for each mask
        pos = np.where(masks[0])
        xmin = np.min(pos[1])
        xmax = np.max(pos[1])
        ymin = np.min(pos[0])
        ymax = np.max(pos[0])
        boxes = [[xmin, ymin, xmax, ymax]]

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((self.NUM_CLASSES,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = torch.zeros((1,), dtype=torch.int64)

        if self.transforms is not None:
            img, target = self.transforms(img, target)
        return img, target


class CrackImagesForClassification(Dataset):
    """Created for Crack Dataset in SDNET2018"""

    def __init__(self, root_dir=None, transforms=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.annotation = []
        self.image_path = []
        imgfolder_difftype = os.listdir(root_dir)
        for imgfolder in imgfolder_difftype:
            # 'C' for cracked and 'U' for uncracked
            full_path = '%s%s/C%s/' % (root_dir, imgfolder, imgfolder)
            crack_img = os.listdir(full_path)
            crack_path = [
                full_path + filename for filename in crack_img if isimage(filename)]
            full_path = '%s%s/U%s/' % (root_dir, imgfolder, imgfolder)
            noncrack_img = os.listdir(full_path)

            # cut in 1/4
            np.random.shuffle(noncrack_img)
            split_end = int(len(noncrack_img)/4)
            noncrack_img = noncrack_img[:split_end]

            noncrack_path = [
                full_path + filename for filename in noncrack_img if isimage(filename)]

            # pack into variable
            self.image_path += crack_path + noncrack_path
            self.annotation += [1]*len(crack_path)
            self.annotation += [0]*len(noncrack_path)

        self.transforms = transforms

    def __len__(self):
        return len(self.image_path)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        annotation = self.annotation[idx]
        img_name = self.image_path[idx]
        image = cv2.imread(img_name)
        if self.transforms:
            image = self.transforms(image)

        # sample = {'image': image, 'annotation': annotation}

        return image, annotation
