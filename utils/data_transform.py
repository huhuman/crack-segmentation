import cv2
import os
import numpy as np


def TRANSFORM_MASK_Binary2RGB(mask_file_path):
    mask = cv2.imread(mask_file_path, cv2.IMREAD_GRAYSCALE)
    mask[mask > 0] = 255
    save_dir = os.path.join(
        '/'.join(mask_file_path.split('/')[:-1]), '../binary_mask/')
    os.makedirs(save_dir, exist_ok=True)
    cv2.imwrite(save_dir + mask_file_path.split('/')[-1], mask)


def TRANSFORM_MASK_RGB2Binary(mask_file_path):
    mask = cv2.imread(mask_file_path, cv2.IMREAD_GRAYSCALE)
    mask = mask/255
    save_dir = os.path.join(
        '/'.join(mask_file_path.split('/')[:-1]), '../rgb_mask/')
    os.makedirs(save_dir, exist_ok=True)
    cv2.imwrite(save_dir + mask_file_path.split('/')[-1], mask)


def TRANSFORM_MASK_SEMANTIC(mask_file_path):
    mask = cv2.imread(mask_file_path, cv2.IMREAD_GRAYSCALE)
    mask = (mask-255)*255    # 255->0, 0->255
    save_dir = os.path.join(
        '/'.join(mask_file_path.split('/')[:-1]), '../semantic_mask/')
    os.makedirs(save_dir, exist_ok=True)
    cv2.imwrite(save_dir + mask_file_path.split('/')[-1], mask)


def OverlayVisualization(base_img_path, apply_img_path):
    mask_color = np.array((255, 255, 0), dtype="uint8")
    b_img = cv2.imread(base_img_path)
    fix_h1, fix_w1, _ = b_img.shape
    a_img = cv2.imread(apply_img_path, cv2.IMREAD_GRAYSCALE)
    fix_h2, fix_w2 = a_img.shape
    if fix_w1 > fix_w2:
        a_img = cv2.resize(a_img, (fix_w1, fix_h1))
    else:
        b_img = cv2.resize(b_img, (fix_w2, fix_h2))
    print(a_img.shape)
    print(b_img.shape)
    a_img = cv2.erode(a_img, np.ones((3, 3), np.uint8), iterations=3)
    mask = a_img != 255
    roi = b_img[mask]
    b_img[mask] = ((1 * mask_color) + (0 * roi)).astype("uint8")
    cv2.imwrite(base_img_path[:-4] + '_overlap_result.png', b_img)


def CropFixedSizeImage(root, size=512):
    image_files = sorted([file for file in os.listdir(root + "image/")])
    export_path = root + 'cropped_fixed_%s/' % (size)
    os.makedirs(export_path, exist_ok=True)
    os.makedirs(export_path + 'image/', exist_ok=True)
    os.makedirs(export_path + 'mask/', exist_ok=True)
    for file in image_files:
        print(file)
        mask = cv2.imread(root + "mask/" + file, cv2.IMREAD_GRAYSCALE)
        image = cv2.imread(root + "image/" + file)
        height, width, _ = image.shape
        h_split_number = int(height/size)
        w_split_number = int(width/size)
        count = 1
        for w_num in range(w_split_number):
            start_x = w_num*size
            end_x = start_x + size
            for h_num in range(h_split_number):
                start_y = h_num*size
                end_y = start_y + size
                cropped_image = image[start_y:end_y, start_x:end_x, :]
                cropped_mask = mask[start_y:end_y, start_x:end_x]
                if np.sum(cropped_mask) > 0:
                    cv2.imwrite('%simage/%s_%02d.%s' % (export_path,
                                                        file[:-4], count, file[-3:]), cropped_image)
                    cv2.imwrite('%smask/%s_%02d.%s' % (export_path,
                                                       file[:-4], count, file[-3:]), cropped_mask)
                    count += 1


if __name__ == "__main__":
    CropFixedSizeImage(
        '/home/aicenter/Documents/hsu/data/project_home_inspection/cracks/label/images/', 512)
    # start_mile, end_mile = '378k+900', '379K+000'
    # OverlayVisualization('/home/aicenter/Documents/hsu/data/test_dataset/test_tunnel/CL/北上_%s_%sm_真值.png' % (start_mile, end_mile),
    #                      '/home/aicenter/Documents/hsu/data/test_dataset/test_tunnel/CL/北上_%s~%sm_segonly_gray_without_train.png' % (start_mile, end_mile))
    # OverlayVisualization('/home/aicenter/Documents/hsu/data/test_dataset/test_tunnel/CL/北上_%s_%sm_真值.png' % (start_mile, end_mile),
    #                      '/home/aicenter/Documents/hsu/data/test_dataset/test_tunnel/CL/北上_%s~%sm_segonly_gray_with_train_v1.png' % (start_mile, end_mile))
    # root = '/home/aicenter/Documents/hsu/data/project_home_inspection/cracks/label_data/mask/'
    # for filename in os.listdir(root):
    #     TRANSFORM_MASK_Binary2RGB(root + filename)
    #     # TRANSFORM_MASK_RGB2Binary(root + filename)
    #     # TRANSFORM_MASK_SEMANTIC(root + filename)
