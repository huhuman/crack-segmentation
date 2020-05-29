import cv2
import mxnet as mx
import os
import matplotlib.pyplot as plt
import numpy as np

# DATA_DIR = "/home/aicenter/Documents/hsu/data/public_dataset/CrackForest-dataset/"
# DATA_DIR = "/home/aicenter/Documents/hsu/data/public_dataset/crack_segmentation_dataset/"
DATA_DIR = "/home/aicenter/Documents/hsu/data/Concrete-Crack/"


def positional_augmentation(joint):
    # Random crop
    crop_height = 200
    crop_width = 200
    # Watch out: weight before height in size param!
    aug = mx.image.RandomCropAug(size=(crop_width, crop_height))
    aug_joint = aug(joint)
    # Deterministic resize
    resize_size = 100
    aug = mx.image.ResizeAug(resize_size)
    aug_joint = aug(aug_joint)
    # Add more translation/scale/rotation augmentations here...
    return aug_joint


def color_augmentation_bright(base):
    # Only applied to the base image, and not the mask layers.
    aug = mx.image.BrightnessJitterAug(brightness=0.5)
    aug_base = aug(base)
    # Add more color augmentations here...
    return aug_base


def color_augmentation_contrast(base):
    aug = mx.image.ContrastJitterAug(contrast=0.5)
    aug_base = aug(base)
    return aug_base


def horizontal_flip(base):
    aug = mx.image.HorizontalFlipAug(p=1)
    aug_base = aug(base)
    return aug_base


def vertical_flip(base):
    aug = mx.image.HorizontalFlipAug(p=1)
    aug_base = aug(base.swapaxes(0, 1)).swapaxes(0, 1)
    return aug_base


def joint_transform(base, mask):
    aug_base_list, aug_mask_list = [], []
    # Convert types
    base = base.astype('float32')/255
    mask = mask.astype('float32')/255

    # Join
    # Concatinate on channels dim, to obtain an 6 channel image
    # (3 channels for the base image, plus 3 channels for the mask)
    base_channels = base.shape[2]  # so we know where to split later on
    joint = mx.nd.concat(base, mask, dim=2)

    # Augmentation Part 1: positional
    aug_joint = positional_augmentation(joint)
    # Split
    aug_base = aug_joint[:, :, :base_channels]
    aug_mask = aug_joint[:, :, base_channels:]
    aug_base_list.append(aug_base)
    aug_mask_list.append(aug_mask)

    # Augmentation 2: color brightness
    aug_base = color_augmentation_bright(base.copy())
    aug_base_list.append(aug_base)
    aug_mask_list.append(mask)

    # Augmentation 3: color contrast
    aug_base = color_augmentation_contrast(base.copy())
    aug_base_list.append(aug_base)
    aug_mask_list.append(mask)

    # Augmentation 4: horizontal flip
    aug_joint = horizontal_flip(joint)
    aug_base = aug_joint[:, :, :base_channels]
    aug_mask = aug_joint[:, :, base_channels:]
    aug_base_list.append(aug_base)
    aug_mask_list.append(aug_mask)

    # # Augmentation 5: vertical flip
    aug_joint = vertical_flip(joint)
    aug_base = aug_joint[:, :, :base_channels]
    aug_mask = aug_joint[:, :, base_channels:]
    aug_base_list.append(aug_base)
    aug_mask_list.append(aug_mask)

    # # Augmentation: test
    # aug_base = test_augmentation(base.copy())
    # aug_base_list.append(aug_base)
    # aug_mask_list.append(mask)

    return aug_base_list, aug_mask_list


def plot_mx_arrays(arrays):
    """
    Array expected to be height x width x 3 (channels), and values are floats between 0 and 255.
    """
    plt.subplots(figsize=(12, 4))
    row_num = len(arrays)
    col_num = len(arrays[0])
    for ri, sub_arrays in enumerate(arrays):
        for idx, array in enumerate(sub_arrays):
            assert array.shape[2] == 3, "RGB Channel should be last"
            plt.subplot(row_num, col_num, idx+1+col_num*ri)
            plt.imshow((array.clip(0, 255)/255).asnumpy())
    plt.savefig('augment.png')


if __name__ == "__main__":
    image_dir = DATA_DIR + "image/"
    mask_dir = DATA_DIR + "mask/"
    img_paths = sorted(
        [image_dir + img_name for img_name in os.listdir(image_dir)])
    mask_paths = sorted(
        [mask_dir + mask_name for mask_name in os.listdir(mask_dir)])

    count = 1
    save_dir = DATA_DIR[:-1] + '_Augmented/'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
        os.mkdir(save_dir+'image/')
        os.mkdir(save_dir+'mask/')
    for img_path, mask_path in zip(img_paths, mask_paths):
        print(img_path, mask_path)
        image = mx.image.imread(img_path)
        mask = mx.image.imread(mask_path)
        trans_imgs, trans_masks = joint_transform(image, mask)
        output_images = [image.astype('float32')]
        output_images += [img*255 for img in trans_imgs]
        output_masks = [mask.astype('float32')]
        output_masks += [msk*255 for msk in trans_masks]
        # plot_mx_arrays([output_images, output_masks])
        # break
        for image, mask in zip(output_images, output_masks):
            # image = cv2.cvtColor(np.float32(image), cv2.COLOR_RGB2BGR)
            cv2.imwrite('%simage/%04d.png' %
                        (save_dir, count), image.asnumpy())
            # mask = cv2.cvtColor(mask.astype('uint8'), cv2.COLOR_RGB2BGR)
            cv2.imwrite('%smask/%04d.png' %
                        (save_dir, count), mask.asnumpy())
            count += 1
