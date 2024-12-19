#!/bin/bash

nvidia-smi
torchrun --nproc_per_node 3 main.py --cfg configs/hmdb51/video-focalnet_base.yaml --output output/ --opts DATA.NUM_FRAMES 8