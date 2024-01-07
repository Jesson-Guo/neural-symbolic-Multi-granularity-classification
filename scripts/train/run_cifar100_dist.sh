#!/bin/bash
python -m torch.distributed.launch --nproc_per_node=2 --master_port 29501 \
        main.py \
        --epochs 50 \
        --lr 0.5 \
        --root /root/autodl-tmp/data \
        --method tot \
        --k 10 \
        --data cifar100 \
        --words /root/autodl-tmp/data/cifar-100-python/words.txt \
        MODEL.MODEL_NAME B_16-i21k-300ep-lr_0.001-aug_light0-wd_0.03-do_0.0-sd_0.0--cifar100-steps_10k-lr_0.001-res_224.npz
