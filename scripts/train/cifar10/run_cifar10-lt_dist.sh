#!/bin/bash
python -m torch.distributed.launch --nproc_per_node=4 --master_port 29502 \
        main.py \
        --devices 0,1,2,3 \
        --epochs 200 \
        --lr 0.0005 \
        --method tot \
        --k 2 \
        --data cifar10-lt \
        --words /mnt/data/ztl/mycode/data/cifar-10-batches-py/words.txt \
        --config ./src/vpt/configs/files/simple/cifar10-lt.yaml \
        --tree ./tots/cifar10-2.json \
        --loss ldam \
        --train \
        --naive \
        # --resume \
        # MODEL.MODEL_NAME tot_cifar10-lt-2_cbce_best.pth
        # --pretrained \
