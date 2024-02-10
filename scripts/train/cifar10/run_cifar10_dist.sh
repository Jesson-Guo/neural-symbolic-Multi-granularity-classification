#!/bin/bash
python -m torch.distributed.launch --nproc_per_node=4 --master_port 29505 \
        main.py \
        --devices 1,2,4,5 \
        --epochs 50 \
        --lr 0.0005 \
        --method tot \
        --k 2 \
        --data cifar10 \
        --words /mnt/data/ztl/mycode/data/cifar-10-batches-py/words.txt \
        --config ./src/vpt/configs/files/simple/cifar10.yaml \
        --tree ./tots/cifar10-2.json \
        --loss ldam \
        --train \
        --pretrained \
        --naive \
