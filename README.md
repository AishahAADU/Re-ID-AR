# Re-ID-AR: Improved Person Re-identification in Video via Joint Weakly Supervised Action Recognition
This is an Official Pytorch Implementation of our paper: Re-ID-AR: Improved Person Re-identification in Video via Joint Weakly Supervised Action Recognition.

[![Python 3.6](https://img.shields.io/badge/python-3.7-blue.svg)](https://www.python.org/downloads/release/python-370/) Tested using Python 3.7.x and Torch: 1.8.0.

## Architectur:
<img width="811" alt="paper2Dig" src="https://user-images.githubusercontent.com/92983150/144039860-49fbe999-fcbd-48f1-b5d2-c174190b76a9.png">

## Abstract:
"We uniquely consider the task of joint person re-identification (Re-ID) and action
recognition in video as a multi-task problem. In addition to the broader potential of joint
Re-ID and action recognition within the context of automated multi-camera surveillance,
we show that the consideration of action recognition in addition to Re-ID results in a
model that learns discriminative feature representations that both improve Re-ID per-
formance and are capable of providing viable per-view (clip-wise) action recognition.
Our approach uses a single 2D Convolutional Neural Network (CNN) architecture com-
prising a common ResNet50-IBN backbone CNN architecture, to extract frame-level fea-
tures with subsequent temporal attention for clip level feature extraction, followed by two
sub-branches:- the IDentification (sub-)Network (IDN) for person Re-ID and the Action
Recognition (sub-)Network for per-view action recognition. The IDN comprises a single
fully connected layer while the ARN comprises multiple attention blocks on a one-to-one
ratio with the number of actions to be recognised. This is subsequently trained as a joint
Re-ID and action recognition task using a combination of two task-specific, multi-loss
terms via weakly labelled actions obtained over two leading benchmark Re-ID datasets
(MARS, LPW). Our consideration of Re-ID and action recognition as a multi-task prob-
lem results in a multi-branch 2D CNN architecture that outperforms prior work in the
field (rank-1 (mAP) – MARS: 93.21%(87.23%), LPW: 79.60%) without any reliance
3D convolutions or multi-stream networks architectures as found in other contemporary
work. Our work represents the first benchmark performance for such a joint Re-ID and
action recognition video understanding task, hitherto unapproached in the literature, and
is accompanied by a new public dataset of supplementary action labels for the seminal
MARS and LPW Re-ID datasets."

[[A. Alsehaim, T.P. Breckon, In Proc. British Machine Vision Conference, BMVA, 2021]](https://breckon.org/toby/publications/papers/alsehaim21reidar.pdf)


## Requirements
```
pip install -r requirements.txt
```
## Getting Started


### Prepare dataset:
1. Download MARS dataset from [here](http://zheng-lab.cecs.anu.edu.au/Project/project_mars.html) . 
2. Download LPW dataset from [here](https://liuyu.us/dataset/lpw/index.html).

### Train and Test Re-ID_AR:
1. Download a pre-train model for Resnet backbone from [here](https://drive.google.com/file/d/13lprTFafpXORqs7XXMLYaelbtw6NxQM1/view).
2. Download action annotation from [here](https://collections.durham.ac.uk/files/r18c97kq420#.YZ-1m7unzJU). 
### 
Train and evaluate the model on MARS dataset using Three actions:
```
python3 -u  main.py --Dataset_name 'MARS'  --Net_path 'resnet50_ibn_a.pth.tar'  --action_num 3  --train_file 'train_3_action.txt' --test_file 'test_3_action.txt' --save_path 'Re-ID-AR'
```

Train and evaluate the model on MARS dataset using Five actions:
```
python3 -u  main.py --Dataset_name 'MARS'  --Net_path 'resnet50_ibn_a.pth.tar'  --action_num 5  --train_file 'train_5_action.txt' --test_file 'test_5_action.txt' --save_path 'Re-ID-AR'
```
Train and evaluate the model on MARS dataset using Eight actions:
```
python3 -u  main.py --Dataset_name 'MARS'  --Net_path 'resnet50_ibn_a.pth.tar'  --action_num 8  --train_file 'train_8_action.txt' --test_file 'test_8_action.txt' --save_path 'Re-ID-AR'
```
##
Train and evaluate the model on LPW dataset using Three actions:
```
python3 -u  main.py --Dataset_name 'LPW'  --Net_path 'resnet50_ibn_a.pth.tar'  --action_num 3  --train_file 'train_scene2_3_action.txt' --test_file 'test_gallery_3_action.txt' --test_query 'test_query_3_action.txt' --save_path 'Re-ID-AR'
```

Train and evaluate the model on LPW dataset using Eight actions:
```
python3 -u  main.py --Dataset_name 'LPW'  --Net_path 'resnet50_ibn_a.pth.tar'  --action_num 8  --train_file 'train_scene3_8_action.txt' --test_file 'test_gallery_8_action.txt' --test_query 'test_query_8_action.txt'  --save_path 'Re-ID-AR'
```
## Acknowledgement
Thanks to [Qidian213](https://github.com/Qidian213), using some implementation from his repository.

## Cite
```
 @inproceedings{alsehaim21reidar,
 author = {Alsehaim, A. and Breckon, T.P.},
 title = {Re-ID-AR: Improved Person Re-identification in Video via Joint Weakly Supervised Action Recognition},
 booktitle = {Proc. British Machine Vision Conference},
 year = {2021},
 month = {November},
 publisher = {BMVA},
 keywords = {re-id, multi-camera, person reidentification, camera-to-camera tracking, action recognition, weak labels, multi-class, cnn, deep learning},
 url = {https://breckon.org/toby/publications/papers/alsehaim21reidar.pdf},
 category = {surveillance},
}
```

