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

CUDA_VISIBLE_DEVICES=$gpus python -m torch.distributed.launch \
--nproc_per_node=$gpun --master_addr $master_addr --master_port $master_port \
test_net.py --config-file $config_file --ckpt $weight_file OUTPUT_DIR $model_dir TEST.BATCH_SIZE $batch_size 

