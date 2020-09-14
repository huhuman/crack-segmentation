import cv2
import mxnet as mx
import os
import matplotlib.pyplot as plt
import numpy as np
import argparse


def crop_augmentation(joint, crop_width, crop_height):
    # Random crop
    # Watch out: weight before height in size param!
    aug = mx.image.RandomCropAug(size=(crop_width, crop_height))
    aug_joint = aug(joint)
    # Deterministic resize
    ratio = 0.7
    resize_size = int(min(crop_width, crop_height)*ratio)
    aug = mx.image.ResizeAug(resize_size)
    aug_joint = aug(aug_joint)
    # Add more translation/scale/rotation augmentations her
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


def joint_transform(image, mask):
    aug_base_list, aug_mask_list = [], []
    # Convert types
    image = image.astype('float32')/255
    mask = mask.astype('float32')/255

    mx_image = mx.nd.array(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    mx_mask = mx.nd.array(mask)
    # Join
    # Concatinate on channels dim, to obtain an 6 channel image
    # (3 channels for the base image, plus 3 channels for the mask)
    base_channels = mx_image.shape[2]  # so we know where to split later on
    joint = mx.nd.concat(mx_image, mx_mask, dim=2)

    # Augmentation Part 1: positional
    (h, w) = image.shape[:2]
    aug_joint = crop_augmentation(joint, int(w*0.5), int(h*0.5))
    # Split
    aug_base = aug_joint[:, :, :base_channels]
    aug_mask = aug_joint[:, :, base_channels:]
    aug_base_list.append(aug_base)
    aug_mask_list.append(aug_mask)

    # Augmentation Part 1-1: positional again
    aug_joint = crop_augmentation(joint, int(w*0.75), int(h*0.75))
    # Split
    aug_base = aug_joint[:, :, :base_channels]
    aug_mask = aug_joint[:, :, base_channels:]
    aug_base_list.append(aug_base)
    aug_mask_list.append(aug_mask)

    # Augmentation 2: color brightness
    aug_base = color_augmentation_bright(mx_image.copy())
    aug_base_list.append(aug_base)
    aug_mask_list.append(mx_mask)

    # Augmentation 3: color contrast
    aug_base = color_augmentation_contrast(mx_image.copy())
    aug_base_list.append(aug_base)
    aug_mask_list.append(mx_mask)

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

    # # Augmentation 6: rotate
    (h, w) = image.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), 30, 1.0)
    aug_base = cv2.warpAffine(image, M, (w, h))
    aug_mask = cv2.warpAffine(mask, M, (w, h))
    aug_base = mx.nd.array(cv2.cvtColor(aug_base, cv2.COLOR_BGR2RGB))
    aug_mask = mx.nd.array(aug_mask)
    aug_base_list.append(aug_base)
    aug_mask_list.append(aug_mask)
    M = cv2.getRotationMatrix2D((w/2, h/2), 120, 1.0)
    aug_base = cv2.warpAffine(image, M, (w, h))
    aug_mask = cv2.warpAffine(mask, M, (w, h))
    aug_base = mx.nd.array(cv2.cvtColor(aug_base, cv2.COLOR_BGR2RGB))
    aug_mask = mx.nd.array(aug_mask)
    aug_base_list.append(aug_base)
    aug_mask_list.append(aug_mask)

    return aug_base_list, aug_mask_list


def plot_mx_arrays(arrays):
    """
    Array expected to be height x width x 3 (channels), and values are floats between 0 and 255.
    """
    plt.subplots(figsize=(12, 4))
    row_num = len(arrays)
    col_num = len(arrays[0])
    print('Row=', row_num, ' Col=', col_num)
    for ri, sub_arrays in enumerate(arrays):
        for idx, array in enumerate(sub_arrays):
            assert array.shape[2] == 3, "RGB Channel should be last"
            plt.subplot(row_num, col_num, idx+1+col_num*ri)
            # make sure array type equals to mxnet ndarray
            plt.imshow((array.clip(0, 255)/255).asnumpy())
    plt.savefig('augment.png')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Data augmentation')
    parser.add_argument('--imgdir', dest='data_dir', type=str, default=None, required=True,
                        help='Root directory of data. Note that there must be subfolder image/ and mask/ under the directory.')
    parser.add_argument('--log', action='store_true', default=False,
                        help='Decide to or not to print logging information. (default: fasle)')
    args = parser.parse_args()

    image_dir = args.data_dir + "image/"
    mask_dir = args.data_dir + "mask/"
    img_paths = sorted(
        [image_dir + img_name for img_name in os.listdir(image_dir)])
    mask_paths = sorted(
        [mask_dir + mask_name for mask_name in os.listdir(mask_dir)])

    count = 1
    save_dir = args.data_dir[:-1] + '_Augmented/'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
        os.mkdir(save_dir+'image/')
        os.mkdir(save_dir+'mask/')
    for img_path, mask_path in zip(img_paths, mask_paths):
        if(args.log):
            print(img_path, mask_path)
        image = cv2.imread(img_path)
        mask = cv2.imread(mask_path)
        trans_imgs, trans_masks = joint_transform(image, mask)
        mx_image = mx.nd.array(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        mx_mask = mx.nd.array(mask)
        output_images = [mx_image.astype('float32')]
        output_images += [img*255 for img in trans_imgs]
        output_masks = [mx_mask.astype('float32')]
        output_masks += [msk*255 for msk in trans_masks]
        # plot_mx_arrays([output_images, output_masks])
        # break
        for image, mask in zip(output_images, output_masks):
            # image = cv2.cvtColor(np.float32(image), cv2.COLOR_RGB2BGR)
            cv2.imwrite('%simage/%04d.png' %
                        (save_dir, count), cv2.cvtColor(image.asnumpy(), cv2.COLOR_RGB2BGR))
            # mask = cv2.cvtColor(mask.astype('uint8'), cv2.COLOR_RGB2BGR)
            cv2.imwrite('%smask/%04d.png' %
                        (save_dir, count), mask.asnumpy())
            count += 1
