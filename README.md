# NOTE: this repo has been deprecated. Please use https://github.com/microsoft/VideoX
 
# 2D-TAN (Optimized)  

## Introduction

This is an optimized re-implementation repository for AAAI'2020 paper: [Learning 2D Temporal Localization Networks for Moment Localization with Natural Language](https://arxiv.org/abs/1912.03590). 

![](pipeline.jpg)

**We show advantages in speed and performance compared with the official implementation (https://github.com/microsoft/2D-TAN).**

## Comparison

### *Performance: Better Results*

**1. TACoS Dataset**
| Repo | Rank1@0.1 | Rank1@0.3 | Rank1@0.5 | Rank5@0.1 | Rank5@0.3 | Rank5@0.5 |
| ---- |:-------------:| :-----:|:-----:|:-----:|:-----:|:-----:|
| Official | 47.59 | 37.29 | 25.32 | 70.31 | 57.81 | 45.04 |
| **Ours** | **57.54** | **45.36** | **31.87** | **77.88** | **65.83** | **54.29** |

**2. ActivityNet Dataset**
| Repo | Rank1@0.3 | Rank1@0.5 | Rank1@0.7 | Rank5@0.3 | Rank5@0.5 | Rank5@0.7 |
| ---- |:-------------:| :-----:|:-----:|:-----:|:-----:|:-----:|
| Official | 59.45 | 44.51 | 26.54 | 85.53 | 77.13 | 61.96 |
| **Ours** | **60.00** | **45.25** | **28.62** | **85.80** | **77.25** | **62.11** |

### *Speed and Cost: Faster Training/Inference, Less Memory Cost*

**1. Speed (ActivityNet Dataset)**
| Repo | Training | Inferece | Required Training Epoches |
| ---- |:-------------:| :-----:| :-----:|
| Official | 1.98 s/batch | 0.81 s/batch | 100 |
| **Ours** | **1.50 s/batch** | **0.61 s/batch** | **5** |

**2. Memory Cost (ActivityNet Dataset)**
| Repo | Training | Inferece |
| ---- |:-------------:| :-----:|
| Official | 4*10145 MB/batch | 4*3065 MB/batch |
| **Ours** | **4*5345 MB/batch** | **4*2121 MB/batch** |

*Note: These results are measured on 4 NVIDIA Tesla V100 GPUs, with batch size 32.*

## Installation
The installation for this repository is easy. Please refer to [INSTALL.md](https://github.com/ChenJoya/2dtan/blob/master/INSTALL.md).

## Dataset
Please refer to [DATASET.md](DATASET.md) to prepare datasets.

## Quick Start
We provide scripts for simplifying training and inference. Please refer to [scripts/train.sh](scripts/train.sh), [scripts/eval.sh](scripts/eval.sh).

For example, if you want to train TACoS dataset, just modifying [scripts/train.sh](scripts/train.sh) as follows:

```bash
# find all configs in configs/
model=2dtan_128x128_pool_k5l8_tacos
# set your gpu id
gpus=0,1,2,3
# number of gpus
gpun=4
# please modify it with different value (e.g., 127.0.0.2, 29502) when you run multi 2dtan task on the same machine
master_addr=127.0.0.1
master_port=29501
...
```

Another example, if you want to evaluate on ActivityNet dataset, just modifying [scripts/eval.sh](scripts/eval.sh) as follows:

```bash
# find all configs in configs/
config_file=configs/2dtan_64x64_pool_k9l4_activitynet.yaml
# the dir of the saved weight
weight_dir=outputs/2dtan_64x64_pool_k9l4_activitynet
# select weight to evaluate
weight_file=model_1e.pth
# test batch size
batch_size=32
# set your gpu id
gpus=0,1,2,3
# number of gpus
gpun=4
# please modify it with different value (e.g., 127.0.0.2, 29502) when you run multi 2dtan task on the same machine
master_addr=127.0.0.2
master_port=29502
...
```

## Support
Please open a new issue. We would like to answer it. Please feel free to contact me: chenjoya@foxmail.com if you need my help.

## Acknowledgements
We greatly appreciate the official 2D-Tan repository https://github.com/microsoft/2D-TAN and maskrcnn-benchmark https://github.com/facebookresearch/maskrcnn-benchmark. We learned a lot from them. Moreover, please remember to cite the paper:
```
@InProceedings{2DTAN_2020_AAAI,
author = {Zhang, Songyang and Peng, Houwen and Fu, Jianlong and Luo, Jiebo},
title = {Learning 2D Temporal Adjacent Networks forMoment Localization with Natural Language},
booktitle = {AAAI},
year = {2020}
} 
```

