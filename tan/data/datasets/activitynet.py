import os
from os.path import join, dirname
import json
import logging

import torch

from .utils import video2feats, moment_to_iou2d, embedding

class ActivityNetDataset(torch.utils.data.Dataset):

    def __init__(self, ann_file, root, feat_file, num_pre_clips, num_clips, pre_query_size):
        super(ActivityNetDataset, self).__init__()

        with open(ann_file, 'r') as f:
            annos = json.load(f)

        self.annos = []
        logger = logging.getLogger("tan.trainer")
        logger.info("Preparing data, please wait...")
        for vid, anno in annos.items():
            duration = anno['duration']
            # Produce annotations
            for timestamp, sentence in zip(anno['timestamps'], anno['sentences']):
                if timestamp[0] < timestamp[1]:
                    moment = torch.tensor([max(timestamp[0], 0), min(timestamp[1], duration)])
                iou2d = moment_to_iou2d(moment, num_clips, duration) 
                query = embedding(sentence)
                self.annos.append(
                    {
                        'vid': vid,
                        'moment': moment,
                        'iou2d': iou2d,
                        'sentence': sentence,
                        'query': query,
                        'wordlen': query.size(0),
                        'duration': duration,
                    }
                 )

        self.feats = video2feats(feat_file, annos.keys(), num_pre_clips, dataset_name="activitynet")

    def __getitem__(self, idx):
        anno = self.annos[idx]
        vid = anno['vid']
        return self.feats[vid], anno['query'], anno['wordlen'], anno['iou2d'], idx
    
    def __len__(self):
        return len(self.annos)
    
    def get_duration(self, idx):
        return self.annos[idx]['duration']
    
    def get_sentence(self, idx):
        return self.annos[idx]['sentence']
    
    def get_moment(self, idx):
        return self.annos[idx]['moment']
    
    def get_vid(self, idx):
        return self.annos[idx]['vid']
