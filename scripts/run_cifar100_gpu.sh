#!/bin/bash
python -m torch.distributed.launch --nproc_per_node=4 --master_port 29502 \
        main.py \
        --arch cifar100 \
        --resume 0 \
        --data_path ../data \
        --epochs 100 \
        --batch_size 64 \
        --num_classes 100 \
        --lr 0.001 \
        --lamb 0.001 \
        --use_cbam 0 \
        --ngpu 2 \
        --prefix RESNET18_CIFAR100 \
        --eval 1 \
        --wnids ../data/cifar-100-python/wnids.txt \
        --words ../data/cifar-100-python/words.txt \
        --hier ./structure_released.xml \
        --cap 800000 \
        --agent dqn \
        --ckpt ./checkpoints/RESNET18_CIFAR100_model_best_0.pt \
        -j 8
