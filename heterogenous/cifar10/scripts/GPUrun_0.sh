#!/bin/bash
python ../main.py \
    --init-method file://Downloads/CI/data \
    --model resnet32 \
    --eta0 0.1 \
    --momentum 0 \
    --weight-decay 5e-4 \
    --step-decay-milestones 80 120 \
    --step-decay-factor 0.1 \
    --clipping-param 1.0 \
    --clipping-option local\
    --world-size 8 \
    --rank 0 \
    --gpu-id 0 \
    --communication-interval 2 \
    --train-epochs 20 \
    --batchsize 16 \
    --dataset CIFAR10 \
    --dataroot ../data \
    --reproducible \
    --seed 0 \
    --log-folder ../logs &
python ../main.py \
    --init-method file://Downloads/CI/data \
    --model resnet32 \
    --eta0 0.1 \
    --momentum 0 \
    --weight-decay 5e-4 \
    --step-decay-milestones 80 120 \
    --step-decay-factor 0.1 \
    --clipping-param 1.0 \
    --clipping-option local\
    --world-size 8 \
    --rank 1 \
    --gpu-id 1 \
    --communication-interval 2 \
    --train-epochs 20 \
    --batchsize 16 \
    --dataset CIFAR10 \
    --dataroot ../data \
    --reproducible \
    --seed 1111 \
    --log-folder ../logs &
python ../main.py \
    --init-method file://Downloads/CI/data  \
    --model resnet32 \
    --eta0 0.1 \
    --momentum 0 \
    --weight-decay 5e-4 \
    --step-decay-milestones 80 120 \
    --step-decay-factor 0.1 \
    --clipping-param 1.0 \
    --clipping-option local\
    --world-size 8 \
    --rank 2 \
    --gpu-id 2 \
    --communication-interval 2 \
    --train-epochs 20 \
    --batchsize 16 \
    --dataset CIFAR10 \
    --dataroot ../data \
    --reproducible \
    --seed 2222 \
    --log-folder ../logs &
python ../main.py \
    --init-method file://Downloads/CI/data \
    --model resnet32 \
    --eta0 0.1 \
    --momentum 0 \
    --weight-decay 5e-4 \
    --step-decay-milestones 80 120 \
    --step-decay-factor 0.1 \
    --clipping-param 1.0 \
    --clipping-option local\
    --world-size 8 \
    --rank 3 \
    --gpu-id 3 \
    --communication-interval 2 \
    --train-epochs 20 \
    --batchsize 16 \
    --dataset CIFAR10 \
    --dataroot ../data \
    --reproducible \
    --seed 3333 \
    --log-folder ../logs
