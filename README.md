# Automatic_Greenscreen_AI (자동_그린스크린_AI)


[![Rick Astley - Never Gonna Give You Up](http://img.youtube.com/vi/f7tSVyHK0FU/0.jpg)](https://youtu.be/f7tSVyHK0FU?t=0s)
[![BTS](http://img.youtube.com/vi/DPrAzZhtE48/0.jpg)](https://youtu.be/DPrAzZhtE48?t=0s) 


이 인공지능은 아래 논문과 모델을 기반으로 만들었습니다.

This AI is based on the paper and model below.


# BlendMask: Top-Down Meets Bottom-Up for Instance Segmentation

    BlendMask: Top-Down Meets Bottom-Up for Instance Segmentation;
    Hao Chen, Kunyang Sun, Zhi Tian, Chunhua Shen, Yongming Huang, and Youliang Yan;
    In: Proc. IEEE Conf. Computer Vision and Pattern Recognition (CVPR), 2020.

[[`Paper`](https://arxiv.org/abs/2001.00309)] [[`BibTeX`](#citing-blendmask)]

## Installation(설치방법)

First install Detectron2 following the official guide: [INSTALL.md](https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md).

*Please use Detectron2 with commit id [9eb4831](https://github.com/facebookresearch/detectron2/commit/9eb4831f742ae6a13b8edb61d07b619392fb6543) if you have any issues related to Detectron2.*

Then build Automatic_Greenscreen_AI with:

```
git clone https://github.com/so45jj45/Automatic_Greenscreen_AI.git
cd Automatic_Greenscreen_AI
python setup.py build develop
```

## Models
### COCO Instance Segmentation Baselines

Model | Name | inf. time | box AP | mask AP | download
--- |:---|:---:|:---:|:---:|:--:|
Mask R-CNN |[R_50_1x](https://github.com/facebookresearch/detectron2/blob/master/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml) | 13 FPS | 38.6 | 35.2 |
BlendMask |[R_50_1x](R_50_1x.yaml) | 14 FPS | 39.9 | 35.8 | [model](https://cloudstor.aarnet.edu.au/plus/s/zoxXPnr6Hw3OJgK/download)
Mask R-CNN |[R_50_3x](https://github.com/facebookresearch/detectron2/blob/master/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml) | 13 FPS | 41.0 | 37.2 | 
BlendMask |[R_50_3x](R_50_3x.yaml) | 14 FPS | 42.7 | 37.8 | [model](https://cloudstor.aarnet.edu.au/plus/s/ZnaInHFEKst6mvg/download)
Mask R-CNN |[R_101_3x](https://github.com/facebookresearch/detectron2/blob/master/configs/COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml) | 10 FPS | 42.9 | 38.6 |
BlendMask |[R_101_3x](R_101_3x.yaml) | 11 FPS | 44.8 | 39.5 | [model](https://cloudstor.aarnet.edu.au/plus/s/e4fXrliAcMtyEBy/download)
BlendMask |[R_101_dcni3_5x](R_101_dcni3_5x.yaml) | 10 FPS | 46.8 | 41.1 | [model](https://cloudstor.aarnet.edu.au/plus/s/vbnKnQtaGlw8TKv/download)

### BlendMask Real-time Models

Model | Name | inf. time | box AP | mask AP | download
--- |:---|:---:|:---:|:---:|:---:
Mask R-CNN |[550_R_50_3x](../RCNN/550_R_50_FPN_3x.yaml) | 16 FPS | 39.1 | 35.3 |
BlendMask |[550_R_50_3x](550_R_50_3x.yaml) | 28 FPS | 38.7 | 34.5 | [model](https://cloudstor.aarnet.edu.au/plus/s/R3Qintf7N8UCiIt/download)
BlendMask |[RT_R_50_4x_syncbn_shtw](RT_R_50_4x_syncbn_shtw.yaml) | 31 FPS | 39.3 | 35.1 | [model](https://cloudstor.aarnet.edu.au/plus/s/iNAQQmfOJlTaBRk/download)
BlendMask |[RT_R_50_4x_bn-head_syncbn_shtw](RT_R_50_4x_bn-head_syncbn_shtw.yaml) | 31 FPS | 39.3 | 35.1 | [model](https://cloudstor.aarnet.edu.au/plus/s/hI15l4ChWFqWvHp/download)
BlendMask |[DLA_34_4x](DLA_34_syncbn_4x.yaml) | 32 FPS | 40.8 | 36.3 | [model](https://cloudstor.aarnet.edu.au/plus/s/JO2xPUGMSbUkKFZ/download)


## Quick Start

### Demo

만약 입력 영상이 웹캠일때

If the input is a webcam

```
python demo/demo3.py --config-file configs/BlendMask/DLA_34_syncbn_4x.yaml --webcam --opts MODEL.WEIGHTS {Model_PATH}/DLA_34_syncbn_4x.pth

```

만약 입력 영상이 비디오 일때

If the input is a video

```
python demo/demo3.py --config-file configs/BlendMask/R_101_dcni3_5x.yaml --video-input {Video_PATH}/rickroll1080p.mp4 --output {Output_PATH}/rickroll1080pone.mp4 --opts MODEL.WEIGHTS checkpoints/R_101_dcni3_5x.pth

```


만약 가장 큰 객체 한개만 출력하려면 demo3 대신 demo2를 사용하세요.

If you want to print only the largest object, use demo2 instead of demo3.


