# Prepare Dataset

2D-Tan uses c3d pretrained feature for training. Therefore we have uploaded the annotations and features in Google Drive. Please download them in any path, then create soft link in this folder. We give instructions for TACoS and ActivityNet datasets.

## TACoS
```bash
# downloading tacos.zip from https://drive.google.com/file/d/1Y53pJebaZdwenbHOpj-yIURSk_cit1OI/view?usp=sharing
# unzip it. this can be done in any path
unzip tacos.zip
# create soft link in this folder
mkdir datasets
ln -s /your/path/tacos datasets/
```

## ActivityNet
```bash
# Please follow https://github.com/microsoft/2D-TAN/tree/master/data/ActivityNet to prepare ActivityNet1.3 datasets
# create soft link in this folder
mkdir datasets
ln -s /your/path/activitynet1.3 datasets/
```
