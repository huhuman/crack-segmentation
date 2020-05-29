# Semantic Segmentation on Crack Image Dataset
## Update Notes
* *2020.05.29: Create repo*

## Requiremnts
The code was tested under the environment
* torch==1.5.0, torchvision==0.6.0
* cuda=10.2
* gcc=7.5.0
* GPU=GeForce RTX 2080 SUPER
## Installation
1. To install required python packages
```bash=
pip3 -r requirement.txt # add --user if u do not have permission
```
2. To build [***detectron2***](https://github.com/facebookresearch/detectron2) benchmark, pls follow instruction in the original git repo: [INSTALL.md](https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md)
(*Note: make sure that GPU is availablethe during the whole build process; otherwise there will be GPU compilation error while training.*)
## Dataset
#### Our dataset mostly originates from https://github.com/khanhha/crack_segmentation
However, there are some duplications, we review the dataset again and refine it to the current version.

## Getting Started
1. To run maskrcnn inference

Open the [detectron2_test.ipynb]()

2. To run deeplabv3+ inference
```bash=
python test_my_dataset.py --backbone resnet  --workers 4 \
    --use-sbd --batch-size 1 --gpu-ids 0 --checkname deeplab-resnet --dataset customed
```

## Model
### MaskRCNN
[paper](https://arxiv.org/abs/1703.06870)
#### Performance

### DeepLabv3+
[paper]()

#### Performance


## License
The source code and model is released under Apache 2.0 license.

