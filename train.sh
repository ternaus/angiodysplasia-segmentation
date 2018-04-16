#!/usr/bin/env bash

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1,2,3

for i in 0 1 2 3 4
do
    python train.py --device-ids 0,1,2,3 --limit 10000 --batch-size 16 --n-epochs 10 --fold $i --model UNet16
    python train.py --device-ids 0,1,2,3 --limit 10000 --batch-size 16 --n-epochs 15 --fold $i --lr 0.00001 --model UNe16
done