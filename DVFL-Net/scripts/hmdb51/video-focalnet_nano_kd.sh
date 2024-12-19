#!/bin/bash

nvidia-smi
torchrun --nproc_per_node 3 main_KD.py --cfg configs/hmdb51/video-focalnet_nano_kd.yaml --output output/ --opts DATA.NUM_FRAMES 8 KD.ALPHA 0.2 KD.BETA 0.3 KD.GAMMA 0.5 KD.TEMPERATURE 10
