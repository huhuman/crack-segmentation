# Segmentation on Crack Image Dataset
![demo](https://github.com/huhuman/crack-segmentation/blob/master/demo/result.png?raw=true)
## Introduction
This repo can provide those who wants to devlop the automation in inspection with a strong foundation for crack identification. Based on [Detectron2](##Reference), we had implemented two well-know algorithms of image segmentation, [Mask R-CNN](##Reference) and [DeepLabv3+](##Reference). Furthermore, a limited dataset combining public sources and our own sources is also open to reproduce our work, and the pre-trained weights are also available for any extended application. The detail of code usage (e.g. how to train/fine-tune a model, predict an image, evaluate the performance) is well-documented in the below.

**The following shows publications based on the repository**
* EWSHM2020
* IPC-SHM-P1

If this repo helps your work, pls [:star2: Cite us! :star2: ](#Citation)

## Update Notes
* *2020.05.29: Create repo*
* *2020.07.20: Add more detail of methods and some codes for testing*
* *2020.09.16: Implementation of Mask R-CNN*

## TODO
- [ ] prune unneccessary packages in requirment.txt
- [ ] implement DeepLabv3+

## Requiremnts
The code was tested under the below environment
* detectron2=0.2.1
* torch=1.5.0, torchvision=0.6.0
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
### DeepCrack
Pls refer to https://github.com/yhlleo/DeepCrack
Note that only 140 images were selected by us.
### Tunnel **[[Link]](https://drive.google.com/file/d/1xXB3FFToH4-YyaAX3Hd7y-cs9xFuWFud/view?usp=sharing)**
| Image | <img width="100" alt="tunnel" src="https://github.com/huhuman/crack-segmentation/blob/master/demo/tunnel/1.png?raw=true"><img width="100" alt="tunnel" src="https://github.com/huhuman/crack-segmentation/blob/master/demo/tunnel/2.png?raw=true"><img width="100" alt="tunnel" src="https://github.com/huhuman/crack-segmentation/blob/master/demo/tunnel/3.png?raw=true.png"><img width="100" alt="tunnel" src="https://github.com/huhuman/crack-segmentation/blob/master/demo/tunnel/4.png?raw=true.png"><img width="100" alt="tunnel" src="https://github.com/huhuman/crack-segmentation/blob/master/demo/tunnel/5.png?raw=true"><img width="100" alt="tunnel" src="https://github.com/huhuman/crack-segmentation/blob/master/demo/tunnel/6.png?raw=true">  | 
| - | - |
| **Mask** | <img width="100" alt="tunnel" src="https://github.com/huhuman/crack-segmentation/blob/master/demo/tunnel/1_m.png?raw=true"><img width="100" alt="tunnel" src="https://github.com/huhuman/crack-segmentation/blob/master/demo/tunnel/2_m.png?raw=true"><img width="100" alt="tunnel" src="https://github.com/huhuman/crack-segmentation/blob/master/demo/tunnel/3_m.png?raw=true.png"><img width="100" alt="tunnel" src="https://github.com/huhuman/crack-segmentation/blob/master/demo/tunnel/4_m.png?raw=true.png"><img width="100" alt="tunnel" src="https://github.com/huhuman/crack-segmentation/blob/master/demo/tunnel/5_m.png?raw=true"><img width="100" alt="tunnel" src="https://github.com/huhuman/crack-segmentation/blob/master/demo/tunnel/6_m.png?raw=true"> | 

### Welding Joints
> Sources from Project 1 of The 1st International Project Competition for Structural Health Monitoring, IPC-SHM (2020)

| Image | <img width="100" alt="welding" src="https://github.com/huhuman/crack-segmentation/blob/master/demo/welding/1.png?raw=true"><img width="100" alt="welding" src="https://github.com/huhuman/crack-segmentation/blob/master/demo/welding/2.png?raw=true"><img width="100" alt="welding" src="https://github.com/huhuman/crack-segmentation/blob/master/demo/welding/3.png?raw=true.png"><img width="100" alt="welding" src="https://github.com/huhuman/crack-segmentation/blob/master/demo/welding/4.png?raw=true.png"><img width="100" alt="welding" src="https://github.com/huhuman/crack-segmentation/blob/master/demo/welding/5.png?raw=true"><img width="100" alt="welding" src="https://github.com/huhuman/crack-segmentation/blob/master/demo/welding/6.png?raw=true">  | 
| - | - |
| **Mask** | <img width="100" alt="welding" src="https://github.com/huhuman/crack-segmentation/blob/master/demo/welding/1_m.png?raw=true"><img width="100" alt="welding" src="https://github.com/huhuman/crack-segmentation/blob/master/demo/welding/2_m.png?raw=true"><img width="100" alt="welding" src="https://github.com/huhuman/crack-segmentation/blob/master/demo/welding/3_m.png?raw=true.png"><img width="100" alt="welding" src="https://github.com/huhuman/crack-segmentation/blob/master/demo/welding/4_m.png?raw=true.png"><img width="100" alt="welding" src="https://github.com/huhuman/crack-segmentation/blob/master/demo/welding/5_m.png?raw=true"><img width="100" alt="welding" src="https://github.com/huhuman/crack-segmentation/blob/master/demo/welding/6_m.png?raw=true"> | 

### <font color=gray>*Optional*
https://github.com/khanhha/crack_segmentation integrated multiple sources of crack images and most of them belong to road surface. Because some duplications exist, data cleaning should be applied before utlization. In addition, DeepCrack is also included by them.</font>

## Getting Started
1. To run Mask R-CNN inference:
```bash=
python test_maskrcnn.py --path [TEST_FOLDER | IMAGE_PATH] \
                        --weight [WEIGHT_PATH]
```
2. To run DeepLabv3+ inference:
```bash=

```
## Model
| Model| Backbone | Dataset | <center>Val</center>mIoU | <center>Links</center> |
| - | - | - | - | - | - |
|<font color=green>Mask R-CNN</font>| [X101+FPN](https://github.com/facebookresearch/detectron2/blob/master/configs/COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml) | Tunnel | 65.0% | [[Publisher]](http://www.ewshm2020.com/) [[Paper]]() [[Weight]](https://drive.google.com/file/d/1V05QQJVTyFOJoVJtCeJJp7_7RRB18K_F/view?usp=sharing)
|<font color=red>DeepLabv3+</font>| [R103-DC5]((https://github.com/facebookresearch/detectron2/tree/master/projects/DeepLab)) | Tunnel | - | [[Publisher]](http://www.ewshm2020.com/) [[Paper]]() [Weight]
|<font color=green>Mask R-CNN</font> | [X101+FPN](https://github.com/facebookresearch/detectron2/blob/master/configs/COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml) | Welding | 75.5% | [[Publisher]](http://www.schm.org.cn/#/IPC-SHM,2020/project1) [[Paper]]() [[Weight]](https://drive.google.com/file/d/1F5NPdm0lQccahXmrYG1Rzj777sZNjRuQ/view?usp=sharing) |
|<font color=red>DeepLabv3+</font>| [R103-DC5]((https://github.com/facebookresearch/detectron2/tree/master/projects/DeepLab)) | Welding | - | [[Publisher]](http://www.schm.org.cn/#/IPC-SHM,2020/project1) [[Paper]]() [Weight]
## Train/Fine-tune
1. Mask R-CNN:
```bash=
python train_maskrcnn.py --tdir [TRAINING_DATA_FOLDER_PATH] \
                --vdir [VALIDATION_DATA_FOLDER_PATH] \
                --weight [WEIGHT_PATH] \
                --lr [LEARNING_RATE] \
                --batch [BATCH_SIZE] \
                --iter [NUMBER_OF_ITERATION] \
                --output [OUTPUT_DIR]
```
## Evaluate
```python=

```

> The code here partially originates from https://github.com/jfzhang95/pytorch-deeplab-xception 

## Reference
* Liu, Y., Yao, J., Lu, X., Xie, R., & Li, L. (2019). DeepCrack: A Deep Hierarchical Feature Learning Architecture for Crack SegmentationNeurocomputing, 338, 139–153. [[Code](https://github.com/yhlleo/DeepCrack)]
* Crack Images from ***CECI Engineering Consultants, Inc. Taiwan***

## Citation
If you use any source code or models of the repository in your work, please use the following BibTeX entry.
*   DeepLabv3+:
```
@inproceedings{deeplabv3plus2018,
  title={Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation},
  author={Liang-Chieh Chen and Yukun Zhu and George Papandreou and Florian Schroff and Hartwig Adam},
  booktitle={ECCV},
  year={2018}
}
```
* Mask R-CNN
```
@inproceedings{he2017mask,
  title={Mask r-cnn},
  author={He, Kaiming and Gkioxari, Georgia and Doll{\'a}r, Piotr and Girshick, Ross},
  booktitle={Proceedings of the IEEE international conference on computer vision},
  pages={2961--2969},
  year={2017}
}
```
* EWSHM2020
```
```
* IPCSHM
```
```
## License
The source code and the models are released under [Apache 2.0 license](./LICENSE).
