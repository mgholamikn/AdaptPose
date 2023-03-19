# [CVPR 2022] AdaptPose: Cross-Dataset Adaptation for 3D Human Pose Estimation by Learnable Motion Generation

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/adaptpose-cross-dataset-adaptation-for-3d/weakly-supervised-3d-human-pose-estimation-on)](https://paperswithcode.com/sota/weakly-supervised-3d-human-pose-estimation-on?p=adaptpose-cross-dataset-adaptation-for-3d)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1d_bcHtBm-rR0mTVJ7484WVJ5Gmk5NkbB#scrollTo=Xtvy95RHLbrd)


This implementation is based on [VideoPsoe3D](https://github.com/facebookresearch/VideoPose3D) and [PoseAug](https://github.com/jfzhang95/PoseAug). Experiemnts on 4 datatsets: Human3.6M, 3DHP, 3DPW, and Ski are provided. Adaptpose is aimed to improve accuracy of 3D pose estimators in cross-dataset scenarios. 
 
<p align="center">.
<img  src="Figures/Tiser.jpg" width="400">
<p/>

## Google Colab
If you do not have a suitable environment to run this project then you could give Google Colab a try. It allows you to run the project in the cloud, free of charge. You may try our Colab demo using the notebook we have prepared: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1d_bcHtBm-rR0mTVJ7484WVJ5Gmk5NkbB#scrollTo=Xtvy95RHLbrd)

## Environment 

```
cd AdaptPose
conda create -n adaptpose python=3.6.9
conda activate adaptpose
```
Install pytorch3d following the instructions [here](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md). 

```
pip install -r requirements.txt
```

## Dataset setup
Due to licenece issues we can not share Huamn3.6m dataset. Please refer to [here](https://github.com/facebookresearch/VideoPose3D/blob/main/DATASETS.md) for instructions on downloading and processing Human3.6M. After downloaing you need to have two files for Human3.6M: ```data_3d_h36m.npz``` ```data_2d_h36m_gt.npz```. Here we provide processed data of 3DHP, 3PW, and SKi dataset:
```
source scripts/prepare_data.sh
```
## Experiments:
Downloaed the pretraind models:

```
source scripts/pretrained_models.sh
```

### 1. Cross-dataset Evaluation of Pretrained Model on 3DHP dataset

Source:Human3.6M/Target:3DHP

```
python3  run_evaluate.py --posenet_name 'videopose' --keypoints gt --evaluate  'checkpoint/adaptpose/videopose/gt/3dhp/ckpt_best_dhp_p1.pth.tar' --dataset_target 3dhp --keypoints_target 'gt' --pad 13  --pretrain_path  'checkpoint/pretrain_baseline/videopose/gt/3dhp/ckpt_best.pth.tar'
```
### 2. Cross-dataset Training of Pretrained Model on 3DHP dataset

Source:Human3.6M/Target:3DHP
```
python3 run_adaptpose.py --note poseaug --posenet_name 'videopose' --lr_p 1e-4 --checkpoint './checkpoint/adaptpose' --keypoints gt --keypoints_target gt --dataset_target '3dhp'  --pretrain_path './checkpoint/pretrain_baseline/videopose/gt/3dhp/ckpt_best.pth.tar'  --pad 13 
```

### 3. Cross-dataset Evaluation of Pretrained Model on 3DPW dataset

Source:Human3.6M/Target:3DPW
```
python3  run_evaluate.py --posenet_name 'videopose' --keypoints gt --evaluate  'checkpoint/adaptpose/videopose/gt/3dpw/ckpt_best_dhp_p1.pth.tar' --dataset_target 3dpw --keypoints_target 'gt' --pad 13  --pretrain_path  'checkpoint/pretrain_baseline/videopose/gt/3dpw/ckpt_best.pth.tar'
```

### 4. Cross-dataset Training of Pretrained Model on 3DPW dataset

Source:Human3.6M/Target:3DPW

```
python3 run_adaptpose.py --note poseaug --posenet_name 'videopose' --lr_p 1e-4 --checkpoint './checkpoint/adaptpose' --keypoints gt --keypoints_target gt --dataset_target '3dpw'  --pretrain_path './checkpoint/pretrain_baseline/videopose/gt/3dhp/ckpt_best.pth.tar'  --pad 13 
```



