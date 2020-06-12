import torch
from torch import nn

from .featpool import build_featpool
from .feat2d import build_feat2d
from .integrator import build_integrator
from .predictor import build_predictor
from .loss import build_tanloss

class TAN(nn.Module):
    def __init__(self, cfg):
        super(TAN, self).__init__()
        self.featpool = build_featpool(cfg) 
        self.feat2d = build_feat2d(cfg)
        self.integrator = build_integrator(cfg)
        self.predictor = build_predictor(cfg, self.feat2d.mask2d)
        self.tanloss = build_tanloss(cfg, self.feat2d.mask2d)
    
    def forward(self, batches, ious2d=None):
        """
        Arguments:

        Returns:

        """
        feats = self.featpool(batches.feats)
        map2d = self.feat2d(feats)
        map2d = self.integrator(batches.queries, batches.wordlens, map2d)
        scores2d = self.predictor(map2d)
        # print(self.training) 
        if self.training:
            return self.tanloss(scores2d, ious2d)
        return scores2d.sigmoid_() * self.feat2d.mask2d
