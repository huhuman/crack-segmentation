from torchvision import transforms, models
from torch.utils.data import DataLoader
import torch
import cv2
import os
import math
import numpy as np
import shutil


def Predicts(model, batch_size, inputs):
    assert type(
        batch_size) == int, 'Parameter Error: batch_size should be interger'
    preds = []
    with torch.set_grad_enabled(False):
        batch_num = int(len(inputs)/batch_size)
        for num in range(batch_num):
            outputs = model(inputs[num*batch_size:(num+1)*batch_size])
            _, batch_preds = torch.max(outputs, 1)
            preds += batch_preds.tolist()
        if len(inputs) % batch_size != 0:
            outputs = model(inputs[batch_size*batch_num:])
            _, batch_preds = torch.max(outputs, 1)
            preds += batch_preds.tolist()
    return preds


def ReConstructTunnelImage(num_w, num_h, crop_images, preds):
    output_image = np.array([])
    for i in range(num_w):
        tmp = crop_images[i*num_h]
        for j in range(1, num_h):
            index = i*num_h + j
            if preds[index] == 1:
                image = crop_images[index]
                tmp = np.vstack((tmp, image))
            else:
                tmp = np.vstack((tmp, np.zeros((crop_h, crop_w, 3))))
        if output_image.size == 0:
            output_image = tmp
            continue
        output_image = np.hstack((output_image, tmp))
    print(output_image.shape)
    return output_image


if __name__ == "__main__":
    # tunnel_image = cv2.imread('../data/test/tunnel.tif')
    # height, width, _ = tunnel_image.shape
    # crop_h, crop_w = (256, 256)
    # test_path = '../data/test_ceci/640xh/%sx%s/' % (crop_h, crop_w)
    test_path = '/home/aicenter/Documents/hsu/data/public_dataset/crack_segmentation_dataset/image/'

    # Check numbers of noncrack images
    # img_files = [1 for filename in os.listdir(
    #     test_path) if 'noncrack_noncrack' in filename]
    # print('Noncrack images number=', sum(img_files))
    # assert False, 'Stopping'

    img_files = [test_path + filename for filename in os.listdir(test_path)]
    img_files.sort()

    data_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    all_predictions = np.array([])
    split_size = 25
    unit_len = int(len(img_files)/split_size) + 1
    for n in range(split_size):
        end_index = min((n+1)*unit_len, len(img_files))
        img_files_split = img_files[n*unit_len:end_index]
        inputs, images = [], []
        for img_file in img_files_split:
            image = cv2.imread(img_file)
            # images.append(image)
            image = data_transforms(image)
            inputs.append(image)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        inputs = torch.stack(inputs)
        inputs = inputs.to(device)
        res_model = models.resnet18(pretrained=False)
        num_ftrs = res_model.fc.in_features
        # Here the size of each output sample is set to 2 : crack/uncrack
        res_model.fc = torch.nn.Linear(num_ftrs, 2)
        res_model.load_state_dict(torch.load(
            'models/classification/classify_05061718.pkl'))
        res_model.to(device)
        res_model.eval()

        # num_w = math.ceil(width/crop_w)
        # num_h = math.ceil(height/crop_h)

        _preds = Predicts(model=res_model, batch_size=10, inputs=inputs)
        all_predictions = np.append(all_predictions, _preds)

        # for pred, img_file in zip(preds, img_files):
        #     split_path = img_file.split('/')
        #     img_name = split_path[-1]
        #     save_dir = '/'.join(split_path[:-2])

        #     save_dir += '/result_%sx%s/' % (crop_h, crop_w)
        #     os.makedirs(save_dir, exist_ok=True)
        #     save_dir_crack = save_dir + 'crack/'
        #     os.makedirs(save_dir_crack, exist_ok=True)
        #     save_dir_noncrack = save_dir + 'noncrack/'
        #     os.makedirs(save_dir_noncrack, exist_ok=True)
        #     if pred == 1:
        #         shutil.copyfile(img_file,
        #                         os.path.join(save_dir_crack, img_name))
        #     else:
        #         shutil.copyfile(img_file,
        #                         os.path.join(save_dir_noncrack, img_name))

    conf_matrix = np.zeros((2, 2))
    for pred, img_file in zip(all_predictions, img_files):
        split_path = img_file.split('/')
        img_name = split_path[-1]
        gt = int('noncrack_noncrack' not in img_name)
        conf_matrix[int(pred), gt] += 1
    print(conf_matrix)
    Acc = (conf_matrix[0, 0]+conf_matrix[1, 1])/sum(conf_matrix)
    Precision = (conf_matrix[1, 1])/(conf_matrix[1, 1]+conf_matrix[1, 0])
    Recall = (conf_matrix[1, 1])/(conf_matrix[1, 1]+conf_matrix[0, 1])
    F1_Score = 2/(1/Precision+1/Recall)
    print('Acc=', Acc)
    print('Precision=', Precision)
    print('Recall=', Recall)
    print('F1_Score=', F1_Score)
