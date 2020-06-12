from terminaltables import AsciiTable
from tqdm import tqdm
import logging

import torch

from tan.data import datasets
from tan.data.datasets.utils import iou, score2d_to_moments_scores

def nms(moments, scores, topk, thresh):
    scores, ranks = scores.sort(descending=True)
    moments = moments[ranks]
    suppressed = ranks.zero_().bool()
    numel = suppressed.numel()
    for i in range(numel - 1):
        if suppressed[i]:
            continue
        mask = iou(moments[i+1:], moments[i]) > thresh
        suppressed[i+1:][mask] = True

    return moments[~suppressed]

def evaluate(dataset, predictions, nms_thresh, recall_metrics=(1,5), iou_metrics=(0.1,0.3,0.5,0.7)):
    """evaluate dataset using different methods based on dataset type.
    Args:
    Returns:
    """
    dataset_name = dataset.__class__.__name__
    logger = logging.getLogger("tan.inference")
    logger.info("Performing {} evaluation (Size: {}).".format(dataset_name, len(dataset)))
    
    num_recall_metrics, num_iou_metrics = len(recall_metrics), len(iou_metrics)
    table = [['Rank@{},mIoU@{}'.format(i,j) \
        for i in recall_metrics for j in iou_metrics]]
    
    recall_metrics = torch.tensor(recall_metrics)
    iou_metrics = torch.tensor(iou_metrics)
    recall_x_iou = torch.zeros(num_recall_metrics, num_iou_metrics)

    num_clips = predictions[0].shape[-1]
    for idx, score2d in tqdm(enumerate(predictions)):  
        duration = dataset.get_duration(idx)
        moment = dataset.get_moment(idx) 

        candidates, scores = score2d_to_moments_scores(score2d, num_clips, duration)
        moments = nms(candidates, scores, topk=recall_metrics[-1], thresh=nms_thresh)

        for i, r in enumerate(recall_metrics):
            mious = iou(moments[:r], dataset.get_moment(idx))
            bools = mious[:,None].expand(r, num_iou_metrics) > iou_metrics
            recall_x_iou[i] += bools.any(dim=0)

    recall_x_iou /= len(predictions)

    table.append(['{:.02f}'.format(recall_x_iou[i][j]*100) \
        for i in range(num_recall_metrics) for j in range(num_iou_metrics)])
    table = AsciiTable(table)
    for i in range(num_recall_metrics*num_iou_metrics):
        table.justify_columns[i] = 'center'
    logger.info('\n' + table.table)
