## Installation

### Requirements:
- CUDA >= 9.0

### Installation
```bash
# create and activate a clean conda env
conda create -n 2dtan
conda activate 2dtan 

# install the right pip and dependencies for the fresh python
conda install ipython pip

# install some dependencies
pip install yacs h5py terminaltables tqdm
# Note: you can use tsinghua mirror to speed up downloading if you are in China
# pip install yacs h5py terminaltables tqdm -i https://pypi.tuna.tsinghua.edu.cn/simple

# follow PyTorch installation in https://pytorch.org/get-started/locally/
# we give the instructions for CUDA 10.1, others are also okay
conda install pytorch torchvision cudatoolkit=10.1 torchtext -c pytorch
# Note: you can use tsinghua mirror to speed up downloading if you are in China
# conda install pytorch torchvision cudatoolkit=10.1 torchtext -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/

# clone 2dtan and enjoy it!
git clone https://github.com/ChenJoya/2dtan
