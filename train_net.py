import argparse
import os

import torch
from torch import optim
from torch import multiprocessing
multiprocessing.set_sharing_strategy('file_system')

from tan.config import cfg
from tan.data import make_data_loader
from tan.engine.inference import inference
from tan.engine.trainer import do_train
from tan.modeling import build_model
from tan.utils.checkpoint import TanCheckpointer
from tan.utils.comm import synchronize, get_rank
from tan.utils.imports import import_file
from tan.utils.logger import setup_logger
from tan.utils.miscellaneous import mkdir, save_config

def train(cfg, local_rank, distributed):
    model = build_model(cfg)
    device = torch.device(cfg.MODEL.DEVICE)
    model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=cfg.SOLVER.LR)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg.SOLVER.MILESTONES)

    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank,
            # this should be removed if we update BatchNorm stats
            broadcast_buffers=False,
        )

    output_dir = cfg.OUTPUT_DIR
    save_to_disk = get_rank() == 0
    checkpointer = TanCheckpointer(
        cfg, model, optimizer, scheduler, output_dir, save_to_disk
    )

    if cfg.MODEL.WEIGHT != "":
        extra_checkpoint_data = checkpointer.load(f=None, use_latest=True)
    else:
        extra_checkpoint_data = checkpointer.load(f=cfg.MODEL.WEIGHT, use_latest=False)
    
    arguments = {"epoch": 1}
    arguments.update(extra_checkpoint_data)
    
    data_loader = make_data_loader(
        cfg,
        is_train=True,
        is_distributed=distributed,
    )

    test_period = cfg.SOLVER.TEST_PERIOD
    if test_period > 0:
        data_loader_val = make_data_loader(cfg, is_train=False, is_distributed=distributed, is_for_period=True)
    else:
        data_loader_val = None

    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD

    
    do_train(
        cfg,
        model,
        data_loader,
        data_loader_val,
        optimizer,
        scheduler,
        checkpointer,
        device,
        checkpoint_period,
        test_period,
        arguments,
    )

    return model

def run_test(cfg, model, distributed):
    if distributed:
        model = model.module
    torch.cuda.empty_cache()  # TODO check if it helps
    dataset_names = cfg.DATASETS.TEST
    data_loaders_val = make_data_loader(cfg, is_train=False, is_distributed=distributed)
    for dataset_name, data_loader_val in zip(dataset_names, data_loaders_val):
        inference(
            model,
            data_loader_val,
            dataset_name=dataset_name,
            nms_thresh=cfg.TEST.NMS_THRESH,
            device=cfg.MODEL.DEVICE,
        )
        synchronize()

def main():
    parser = argparse.ArgumentParser(description="Tan")
    parser.add_argument(
        "--config-file",
        default="configs/2dtan_128x128_pool_k5l8_tacos.yaml",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "--skip-test",
        dest="skip_test",
        help="Do not test the final model",
        action="store_true",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
        synchronize()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir:
        mkdir(output_dir)

    logger = setup_logger("tan", output_dir, get_rank())
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(args)

    logger.info("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    output_config_path = os.path.join(cfg.OUTPUT_DIR, 'config.yml')
    logger.info("Saving config into: {}".format(output_config_path))
    # save overloaded model config in the output directory
    save_config(cfg, output_config_path)

    model = train(cfg, args.local_rank, args.distributed)

    if not args.skip_test:
        run_test(cfg, model, args.distributed)

if __name__ == "__main__":
    main()
