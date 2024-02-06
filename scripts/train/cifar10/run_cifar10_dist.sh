#!/bin/bash
python -m torch.distributed.launch --nproc_per_node=4 --master_port 29505 \
        main.py \
        --devices 0,1,2,3 \
        --epochs 100 \
        --lr 0.0005 \
        --method tot \
        --k 2 \
        --data cifar10 \
        --words /root/autodl-tmp/data/cifar-10-batches-py/words.txt \
        --config ./src/vpt/configs/files/simple/cifar10.yaml \
        --tree ./tots/no_other/cifar10.json \
        --loss ldam \
        --train \
        --pretrained \
        --naive \
