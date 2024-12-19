#!/bin/bash

nvidia-smi
torchrun --nproc_per_node 3 main.py --cfg configs/ucf50/video-focalnet_small.yaml --output output/ --opts DATA.NUM_FRAMES 8
