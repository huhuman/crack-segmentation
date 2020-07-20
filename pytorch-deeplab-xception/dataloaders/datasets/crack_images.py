import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os
import numpy as np

from dataloaders import custom_transforms as tr
from utils.summaries import TensorboardSummary


def Isimage(filename):
    IMAGE_FORMAT = [
        'jpg', 'png', 'JPEG', 'PNG', 'jpeg'
    ]
    return filename.split('.')[-1] in IMAGE_FORMAT


def CrackImagesTrainValSplit(self, root_dir, train_val_split=0.7):
    """
    Created for Generating Two ImageDataset: Train & Val
    Args:
        root_dir (string): Directory with all the images.
        train_val_split (double between 0 and 1): Ratio of training set to the whole dataset 
    """
    self.imgs = list(sorted(os.listdir(root_dir+'image/')))
    self.masks = list(sorted(os.listdir(root_dir+'mask/')))


class ImageSetWithFilenameSet(Dataset):
    """Created for Crackdataset"""

    NUM_CLASSES = 2  # should include background

    def __init__(self, args, root_dir, img_files, mask_files):
        """
        Args:
            root_dir (string): Directory with all the images.
        """
        self.args = args
        self.root = root_dir
        self.imgs = img_files  # only filename
        self.masks = mask_files  # only filename

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_path = self.root + "image/" + self.imgs[idx]
        mask_path = self.root + "mask/" + self.masks[idx]
        # img = cv2.imread(img_path)
        # mask = cv2.imread(mask_path)/255  # HxWxC
        _img = Image.open(img_path).convert('RGB')
        mask = np.array(Image.open(mask_path).convert('L'))/255
        # mask = np.array(Image.open(mask_path))/255
        _target = Image.fromarray(mask)

        sample = {'image': _img, 'label': _target}

        return self.transform_tr(sample)

    def transform_tr(self, sample):
        composed_transforms = transforms.Compose([
            tr.RandomHorizontalFlip(),
            tr.RandomScaleCrop(base_size=self.args.base_size,
                               crop_size=self.args.crop_size),
            tr.RandomGaussianBlur(),
            tr.Normalize(mean=(0.485, 0.456, 0.406),
                         std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        return composed_transforms(sample)


class CrackImagesForSegmentation(Dataset):
    """Created for Crackdataset"""

    NUM_CLASSES = 2  # should include background

    def __init__(self, args, root_dir=None, mode='train'):
        """
        Args:
            root_dir (string): Directory with all the images.
        """
        self.args = args
        self.root = root_dir
        self.imgs = list(sorted(os.listdir(root_dir+'image/')))
        self.masks = list(sorted(os.listdir(root_dir+'mask/')))
        self.mode = mode

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_path = self.root + "image/" + self.imgs[idx]
        mask_path = self.root + "mask/" + self.masks[idx]
        # img = cv2.imread(img_path)
        # mask = cv2.imread(mask_path)/255  # HxWxC
        _img = Image.open(img_path).convert('RGB')
        mask = np.array(Image.open(mask_path).convert('L'))/255
        # mask = np.array(Image.open(mask_path))/255
        _target = Image.fromarray(mask)

        sample = {'image': _img, 'label': _target}
        if self.mode == 'train':
            return self.transform_tr(sample)
        else:
            return self.transform_val(sample)

    def transform_tr(self, sample):
        composed_transforms = transforms.Compose([
            tr.RandomHorizontalFlip(),
            tr.RandomScaleCrop(base_size=self.args.base_size,
                               crop_size=self.args.crop_size),
            tr.RandomGaussianBlur(),
            tr.Normalize(mean=(0.485, 0.456, 0.406),
                         std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        return composed_transforms(sample)

    def transform_val(self, sample):

        composed_transforms = transforms.Compose([
            tr.FixScaleCrop(crop_size=513),
            tr.Normalize(mean=(0.485, 0.456, 0.406),
                         std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        return composed_transforms(sample)


class ValCrackImagesForSegmentation(Dataset):
    """Created for Crackdataset"""

    NUM_CLASSES = 2  # should include background

    def __init__(self, root_dir=None):
        """
        Args:
            root_dir (string): Directory with all the images.
        """
        self.root = root_dir
        self.imgs = list(sorted(os.listdir(root_dir+'image/')))
        self.masks = list(sorted(os.listdir(root_dir+'mask/')))

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_path = self.root + "image/" + self.imgs[idx]
        mask_path = self.root + "mask/" + self.masks[idx]
        # img = cv2.imread(img_path)
        # mask = cv2.imread(mask_path)/255  # HxWxC
        _img = Image.open(img_path).convert('RGB')
        mask = np.array(Image.open(mask_path).convert('L'))/255
        # mask = np.array(Image.open(mask_path))/255
        _target = torch.from_numpy(
            np.array(Image.fromarray(mask)).astype(np.float32)).float()
        return {'image': self.transform_val(_img), 'label': _target, 'filename': self.imgs[idx]}

    def transform_val(self, image):
        # using torchvision.transform instead of customized tr
        composed_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        return composed_transforms(image)


class TestCrackImagesForSegmentation(Dataset):
    """Created for Crackdataset"""

    NUM_CLASSES = 2  # should include background

    def __init__(self, root_dir=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root = root_dir
        self.imgs = list(sorted(os.listdir(root_dir+'image/')))

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_path = self.root + "image/" + self.imgs[idx]
        _img = Image.open(img_path).convert('RGB')

        return {'image': self.transform_val(_img), 'filename': self.imgs[idx]}

    def transform_val(self, image):
        # using torchvision.transform instead of customized tr
        composed_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        return composed_transforms(image)
