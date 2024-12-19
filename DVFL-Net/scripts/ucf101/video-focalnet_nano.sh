#!/bin/bash

nvidia-smi
torchrun --nproc_per_node 3 main.py --cfg configs/ucf101/video-focalnet_nano.yaml --output output/ --opts DATA.NUM_FRAMES 8
