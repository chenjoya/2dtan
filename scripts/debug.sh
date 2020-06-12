model=2dtan_128x128_pool_k5l8
dataset=tacos

gpus=0
gpun=1
master_addr=127.0.0.1
master_port=29501

# ------------------------ need not change -----------------------------------
config_file=configs/$model\_$dataset\.yaml
output_dir=outputs/$model\_$dataset

CUDA_VISIBLE_DEVICES=$gpus python -m torch.distributed.launch \
--nproc_per_node=$gpun --master_addr $master_addr --master_port $master_port train_net.py --config-file $config_file OUTPUT_DIR $output_dir SOLVER.BATCH_SIZE 8 \


