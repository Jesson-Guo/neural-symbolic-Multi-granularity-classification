#!/bin/bash
python -m torch.distributed.launch --nproc_per_node=4 --master_port 29503 \
        main.py \
        --devices 0,1,2,3 \
        --epochs 300 \
        --lr 0.0005 \
        --method tot \
        --k 10 \
        --data cifar100-lt \
        --words /root/autodl-tmp/data/cifar-100-python/words.txt \
        --config ./src/vpt/configs/files/simple/cifar100-lt.yaml \
        --tree ./tots/no_other/cifar100-temp.json \
        --loss ldam \
        --train \
        --naive \
        --pretrained \
