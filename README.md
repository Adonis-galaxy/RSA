# RSA: Resolving Scale Ambiguities in Monocular Depth Estimators through Language Descriptions #

Official implementation of the paper "RSA: Resolving Scale Ambiguities in Monocular Depth Estimators through Language Descriptions"

Accepted by NeurIPS 2024

Paper Link: https://arxiv.org/abs/2410.02924

Authors: Ziyao Zeng, Yangchao Wu, Hyoungseob Park, Daniel Wang, Fengyu Yang, Stefano Soatto, Dong Lao, Byung-Woo Hong, Alex Wong

<!-- ## Overview ##

### Pipeline ###

### Visualization on NYU-Depth-v2 ###

### Poster ### -->

## TODO: ##
Here we release the training code. I am still reorganizing the code (code is a little messy now). Will release the final version and the CKPT soon.

## Setup Environment ##
Create Virtual Environment:
```
cd RSA

virtualenv -p /usr/bin/python3.8 ~/venvs/rsa

vim  ~/.bash_profile
```
Insert the following line to vim:
```
alias rsa="export CUDA_HOME=/usr/local/cuda-11.1 && source ~/venvs/rsa/bin/activate"
```
Then activate it, install all packages:
```
source ~/.bash_profile

rsa

pip install -r requirements.txt


## Run training for Depth Anything on NYU-Depth-V2 / KITTI / VOID ##
Specify GPU Number in sh/train_3d_da.sh, then run by:
```
sh train_3d_da.sh
```
Before running new experiments, remember to change the model_name in train_3d_da.sh and config/arguments_train_3d_da.txt to be the same.

## Run training for MiDaS on NYU-Depth-V2 / KITTI / VOID ##
Specify GPU Number in sh/train_3d_midas.sh, then run by:
```
sh train_3d_midas.sh
```
Before running new experiments, remember to change the model_name in train_3d_midas.sh and config/arguments_train_3d_midas.txt to be the same.

## Run training for DPT on NYU-Depth-V2 / KITTI / VOID ##
Specify GPU Number in sh/train_3d_da.sh, then run by:
```
sh train_3d_da.sh
```
Before running new experiments, remember to change the model_name in train_3d_da.sh and config/arguments_train_3d_da.txt to be the same.



## Setup Datasets ##
### Prepare Datasets ###
Download [NYU-Depth-v2](https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html), [KITTI](https://www.cvlibs.net/datasets/kitti/), and [VOID](https://github.com/alexklwong/void-dataset). Or you can refer to [KBNet](https://github.com/alexklwong/calibrated-backprojection-network) to prepare datasets through the provided download scripts.

Then change the data_path and gt_path in train_3d.py and DATA_PATH_VOID in eval_void/data_utils.py

The structure of dataset should look like this:

    ├── nyu_depth_v2
    │   ├── official_splits             # path to nyu-depth-v2 data_path_eval and gt_path_eval
    │   │   ├── test
    │   │   │   ├── bathroom
    │   │   │   │   ├── rgb_00045.jpg
    │   │   │   │   ├── rgb_00046.jpg
    │   │   │   │   ├── ...
    │   │   ├── train                    # We don't use this part
    │   │   │   ├── ...
    │   ├── sync                           # path to nyu-depth-v2 data_path and gt_path
    │   │   ├── basement_0001a
    │   │   │   ├── rgb_00000.jpg
    │   │   │   ├── rgb_00001.jpg
    │   │   │   ├── ...
    └── ...

    ├── kitti_raw_data                     # path to kitti data_path and data_path_eval
    │   ├── 2011_09_26                     # name of dataset
    │   │   ├── 2011_09_26_drive_0001_sync
    │   │   │   ├── ...
    └── ...


    ├── kitti_ground_truth                 # path to kitti gt_path and gt_path_eval
    │   ├── 2011_09_26_drive_0001_sync
    │   │   ├── ...
    └── ...

    ├── XXX                               # DATA_PATH_VOID in eval_void/data_utils.py
    │   ├── void_1500                     # path to void dataset
    │   │   ├── data
    │   │   │   ├── birthplace_of_internet
    │   │   │   │   ├── ...
    │   └── ...


## Acknowledgements ##
We would like to acknowledge the use of code snippets from various open-source libraries and contributions from the online coding community, which have been invaluable in the development of this project. Specifically, we would like to thank the authors and maintainers of the following resources:

[CLIP](https://github.com/openai/CLIP)

[WorDepth](https://github.com/Adonis-galaxy/WorDepth)

[Depth Anything](https://github.com/LiheYoung/Depth-Anything)

[MiDaS](https://github.com/isl-org/MiDaS)

[DPT](https://github.com/isl-org/DPT)

[MaskDINO](https://github.com/IDEA-Research/MaskDINO)

[LLaVA](https://github.com/haotian-liu/LLaVA)

[KBNet](https://github.com/alexklwong/calibrated-backprojection-network)

[VOID-dataset] (https://github.com/alexklwong/void-dataset)