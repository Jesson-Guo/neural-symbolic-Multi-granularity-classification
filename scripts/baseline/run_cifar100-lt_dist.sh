#!/bin/bash
python -m torch.distributed.launch --nproc_per_node=2 --master_port 29501 \
        main.py \
        --devices 0,1 \
        --epochs 300 \
        --lr 0.0005 \
        --method vit \
        --k 10 \
        --data cifar100-lt \
        --words /root/autodl-tmp/data/cifar-100-python/words.txt \
        --config ./src/vpt/configs/files/simple/cifar100-lt.yaml \
        --loss ldam \
        --train \
        --naive \
        --pretrained \
