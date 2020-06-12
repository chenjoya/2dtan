import logging

import torch

from tan.utils.comm import get_world_size
from tan.utils.imports import import_file
from . import datasets as D
from .samplers import DistributedSampler
from .collate_batch import BatchCollator

def build_dataset(dataset_list, dataset_catalog, cfg, is_train=True):
    # build specific dataset
    if not isinstance(dataset_list, (list, tuple)):
        raise RuntimeError(
            "dataset_list should be a list of strings, got {}".format(
                dataset_list
            )
        )
    datasets = []
    for dataset_name in dataset_list:
        data = dataset_catalog.get(dataset_name)
        factory = getattr(D, data["factory"])
        args = data["args"]
        args["num_pre_clips"] = cfg.INPUT.NUM_PRE_CLIPS
        args["num_clips"] = cfg.MODEL.TAN.NUM_CLIPS
        args["pre_query_size"] = cfg.INPUT.PRE_QUERY_SIZE
        # make dataset from factory
        dataset = factory(**args)
        datasets.append(dataset)

    # for testing, return a list of datasets
    if not is_train:
        return datasets

    # for training, concatenate all datasets into a single one
    dataset = datasets[0]
    if len(datasets) > 1:
        dataset = D.ConcatDataset(datasets)
    return [dataset]

def make_data_sampler(dataset, shuffle, distributed):
    if distributed:
        return DistributedSampler(dataset, shuffle=shuffle)
    if shuffle:
        sampler = torch.utils.data.sampler.RandomSampler(dataset)
    else:
        sampler = torch.utils.data.sampler.SequentialSampler(dataset)
    return sampler

def make_batch_data_sampler(dataset, sampler, batch_size):
    batch_sampler = torch.utils.data.sampler.BatchSampler(
        sampler, batch_size, drop_last=False
    )
    return batch_sampler

def make_data_loader(cfg, is_train=True, is_distributed=False, is_for_period=False):
    num_gpus = get_world_size()
    if is_train:
        batch_size = cfg.SOLVER.BATCH_SIZE
        assert (
            batch_size % num_gpus == 0
        ), "SOLVER.BATCH_SIZE ({}) must be divisible by the number of GPUs ({}) used.".format(
            batch_size, num_gpus)
        batch_size_per_gpu = batch_size // num_gpus
        shuffle = True
        max_epoch = cfg.SOLVER.MAX_EPOCH
    else:
        batch_size = cfg.TEST.BATCH_SIZE
        assert (
            batch_size % num_gpus == 0
        ), "TEST.BATCH_SIZE ({}) must be divisible by the number of GPUs ({}) used.".format(
            batch_size, num_gpus)
        batch_size_per_gpu = batch_size // num_gpus
        shuffle = False if not is_distributed else True

    if batch_size_per_gpu > 1:
        logger = logging.getLogger(__name__)

    paths_catalog = import_file(
        "tan.cfg.paths_catalog", cfg.PATHS_CATALOG, True
    )
    DatasetCatalog = paths_catalog.DatasetCatalog
    dataset_list = cfg.DATASETS.TRAIN if is_train else cfg.DATASETS.TEST
    datasets = build_dataset(dataset_list, DatasetCatalog, cfg, is_train=is_train or is_for_period)

    data_loaders = []
    for dataset in datasets:
        sampler = make_data_sampler(dataset, shuffle, is_distributed)
        batch_sampler = make_batch_data_sampler(dataset, sampler, batch_size_per_gpu)
        data_loader = torch.utils.data.DataLoader(
            dataset,
            num_workers=cfg.DATALOADER.NUM_WORKERS,
            batch_sampler=batch_sampler,
            collate_fn=BatchCollator(),
        )
        data_loaders.append(data_loader)
    if is_train or is_for_period:
        # during training, a single (possibly concatenated) data_loader is returned
        assert len(data_loaders) == 1
        return data_loaders[0]
    return data_loaders
