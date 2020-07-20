# Semantic Segmentation on Crack Image Dataset
![demo](https://github.com/huhuman/crack-segmentation/blob/master/demo/result.png?raw=true)
## Update Notes
* *2020.05.29: Create repo*
* *2020.07.20: Add more detail of methods and some codes for testing*

## Requiremnts
The code was tested under the below environment
* torch==1.5.0, torchvision==0.6.0
* cuda=10.2
* gcc=7.5.0
* GPU=GeForce RTX 2080 SUPER
## Installation
1. To install required python packages
```bash=
pip3 -r requirement.txt # add --user if you do not have root permission
```
2. To build [***detectron2***](https://github.com/facebookresearch/detectron2) benchmark, pls follow instruction in the original git repo: [INSTALL.md](https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md)
(*Note: make sure that GPU is available during the whole build process; otherwise there will be GPU compilation error while training.*)
## Dataset
#### Our dataset mostly originates from https://github.com/khanhha/crack_segmentation
However, there are some duplications, we review the dataset again and refine it to the current version. Moreover, the dataset was combined with our self-annotated data. In total, ~?? images are available for training and testing.

**[Download our dataset here](https://drive.google.com/drive/folders/17Tt-qNx_k1P_bJTaM1XcQDdED7JAb-dx?usp=sharing)**

## Getting Started
1. To run maskrcnn inference

Please refer to the [detectron2_test.ipynb](https://github.com/huhuman/crack-segmentation/blob/master/detectron2_test.ipynb)

2. To run deeplabv3+ inference
```bash=
python test_my_dataset.py --backbone resnet  --workers 4 \
    --use-sbd --batch-size 1 --gpu-ids 0 --checkname deeplab-resnet --dataset customed
```

## Model
### [MaskRCNN](https://arxiv.org/abs/1703.06870)
#### Result
* transfer learing on pre-trained weight: [X101-FPN](https://github.com/facebookresearch/detectron2/blob/master/configs/COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml)

| Backbone | Dataset | <center>Val</center>mIoU | Weight |
| - | - | - | - |
| ResNet101 | 183 images     | 75.5%     | [Drive]()

### [DeepLabv3+]()

#### Result
* transfer learing on pre-trained weight: [resnet101](https://download.pytorch.org/models/resnet101-5d3b4d8f.pth)

| Backbone | Dataset | <center>Val</center>mIoU | Weight |
| - | - | - | - |
| ResNet101 | 183 images     | 74.1%     | [Drive]()

## Reference
* DeepLabv3+ based on Pytorch from https://github.com/jfzhang95/pytorch-deeplab-xception
* Crack Images from *CECI Engineering Consultants, Inc. Taiwan*

## License
The source code and model is released under [Apache 2.0 license](./LICENSE).
