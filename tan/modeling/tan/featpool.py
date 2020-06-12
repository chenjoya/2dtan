import torch
from torch import nn

from tan.modeling import registry

@registry.FEATPOOL.register("FeatAvgPool")
class FeatAvgPool(nn.Module):
    def __init__(self, input_size, hidden_size, kernel_size, stride):
        super(FeatAvgPool, self).__init__()
        self.conv = nn.Conv1d(input_size, hidden_size, 1, 1)
        self.pool = nn.AvgPool1d(kernel_size, stride)

    def forward(self, x):
        return self.pool(self.conv(x.transpose(1, 2)).relu())
        # return self.pool(self.conv(x).relu())

@registry.FEATPOOL.register("FeatMaxPool")
class FeatMaxPool(nn.Module):
    def __init__(self, input_size, hidden_size, kernel_size, stride):
        super(FeatMaxPool, self).__init__()
        self.conv = nn.Conv1d(input_size, hidden_size, 1, 1)
        self.pool = nn.MaxPool1d(stride)

    def forward(self, x):
        return self.pool(self.conv(x.transpose(1, 2)).relu())
        # return self.pool(self.conv(x).relu())

def build_featpool(cfg):
    input_size = cfg.MODEL.TAN.FEATPOOL.INPUT_SIZE
    hidden_size = cfg.MODEL.TAN.FEATPOOL.HIDDEN_SIZE
    kernel_size = cfg.MODEL.TAN.FEATPOOL.KERNEL_SIZE
    stride = cfg.INPUT.NUM_PRE_CLIPS // cfg.MODEL.TAN.NUM_CLIPS
    return registry.FEATPOOL[
        cfg.MODEL.TAN.FEATPOOL.POOLER
    ](input_size, hidden_size, kernel_size, stride)
