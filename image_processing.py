from utils.utils import ImagePreprocessing
import cv2
import os

DATA_DIR = '/home/aicenter/Documents/hsu/data/Crack_Augmented/image/'


def main():
    '''
    apply any transform method to the training image for image enhancement or noise reduction
    '''
    image_paths = [DATA_DIR+filename for filename in os.listdir(DATA_DIR)]
    folder_name = DATA_DIR.split('/')[-3]
    save_dir = DATA_DIR + '../../' + folder_name + '_Trans/'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    save_dir += 'image/'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    for image_path in image_paths:
        img = cv2.imread(image_path)
        img_name = image_path.split('/')[-1]
        transformed_img = ImagePreprocessing.preprocessing(img)
        print(save_dir+img_name)
        cv2.imwrite(save_dir+img_name, transformed_img)


if __name__ == "__main__":
    main()
