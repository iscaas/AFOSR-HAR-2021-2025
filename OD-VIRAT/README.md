# OD-VIRAT: A Large-Scale Benchmark for Object Detection in Realistic Surveillance Environments

[Anonymous]
<!-- [Hayat Ullah](https://scholar.google.com.pk/citations?user=xnXPj0UAAAAJ&hl=en),  -->
<!-- [Arslan Munir](https://people.cs.ksu.edu/~amunir/) -->


[![paper](https://img.shields.io/badge/IEEE-Paper-<COLOR>.svg)]()
<hr />

> **Abstract:**
>*will add paper abstract here...*

## Table of Contents
<!--ts-->
   * [News](#rocket-News)
   * [Overview](#overview)
   * [Visualization](#visualization-first-and-last-layer-spatio-temporal-modulator)
   * [Environment Setup](#environment-setup)
   * [Dataset Preparation](#dataset-preparation)
   * [Training](#training)
   * [Evaluation](#evaluation)
   * [Citation](#citation)
   * [Acknowledgements](#acknowledgements)
<!--te-->

## Dataset Detail and  Preparation

Please follow [DATA.md](./DATA.md) for dataset details and preparation step.

## Training
<!--
To train a Video-FocalNet on a video dataset from scratch, run:

```bash
python -m torch.distributed.launch --nproc_per_node <num-of-gpus-to-use>  main.py \
--cfg <config-file> --batch-size <batch-size-per-gpu> --output <output-directory> \
--opts DATA.ROOT path/to/root DATA.TRAIN_FILE train.csv DATA.VAL_FILE val.csv
```

Alternatively, the `DATA.ROOT`, `DATA.TRAIN_FILE`, and `DATA.VAL_FILE` paths can be set directly in the config files provided in the `configs` directory. We also provide bash scripts to train Video-FocalNets on various datasets in the `scripts` directory.

Additionally, the TRAIN.PRETRAINED_PATH can be set (either in the config file or bash script) to provide a pretrained model to initialize the weights. To initialize from the ImageNet-1K weights please refer to the [FocalNets](https://github.com/microsoft/FocalNet) repository and download the [FocalNet-T-SRF](https://github.com/microsoft/FocalNet/releases/download/v1.0.0/focalnet_tiny_srf.pth), [FocalNet-S-SRF](https://github.com/microsoft/FocalNet/releases/download/v1.0.0/focalnet_small_srf.pth) or [FocalNet-B-SRF](https://github.com/microsoft/FocalNet/releases/download/v1.0.0/focalnet_base_srf.pth) to initialize Video-FocalNet-T, Video-FocalNet-S or Video-FocalNet-B respectively. Alternatively, one of the provided pretrained Video-FocalNet models can also be utilized to initialize the weights. -->

## Evaluation
<!--
To evaluate pre-trained Video-FocalNets on your dataset:

```bash
python -m torch.distributed.launch --nproc_per_node <num-of-gpus-to-use>  main.py  --eval \
--cfg <config-file> --resume <checkpoint> \
--opts DATA.NUM_FRAMES 8 DATA.BATCH_SIZE 8 TEST.NUM_CLIP 4 TEST.NUM_CROP 3 DATA.ROOT path/to/root DATA.TRAIN_FILE train.csv DATA.VAL_FILE val.csv
```

For example, to evaluate the `Video-FocalNet-B` with a single GPU on Kinetics400:

```bash
python -m torch.distributed.launch --nproc_per_node 1  main.py  --eval \
--cfg configs/kinetics400/video_focalnet_base.yaml --resume video-focalnet_base_k400.pth \
--opts DATA.NUM_FRAMES 8 DATA.BATCH_SIZE 8 TEST.NUM_CLIP 4 TEST.NUM_CROP 3 DATA.ROOT path/to/root DATA.TRAIN_FILE train.csv DATA.VAL_FILE val.csv
```

Alternatively, the `DATA.ROOT`, `DATA.TRAIN_FILE`, and `DATA.VAL_FILE` paths can be set directly in the config files provided in the `configs` directory.
According to our experience and sanity checks, there is a reasonable random variation of about +/-0.3% top-1 accuracy when testing on different machines.

Additionally, the TRAIN.PRETRAINED_PATH can be set (either in the config file or bash script) to provide a pretrained model to initialize the weights. To initialize from the ImageNet-1K weights please refer to the [FocalNets](https://github.com/microsoft/FocalNet) repository and download the [FocalNet-T-SRF](https://github.com/microsoft/FocalNet/releases/download/v1.0.0/focalnet_tiny_srf.pth), [FocalNet-S-SRF](https://github.com/microsoft/FocalNet/releases/download/v1.0.0/focalnet_small_srf.pth) or [FocalNet-B-SRF](https://github.com/microsoft/FocalNet/releases/download/v1.0.0/focalnet_base_srf.pth) to initialize Video-FocalNet-T, Video-FocalNet-S or Video-FocalNet-B respectively. Alternatively, one of the provided pretrained Video-FocalNet models can also be utilized to initialize the weights. -->

## Citation

<!-- If you find our work, this repository, or pretrained models useful, please consider giving a star :star: and citation.
```bibtex
@InProceedings{Wasim_2023_ICCV,
    author    = {Wasim, Syed Talal and Khattak, Muhammad Uzair and Naseer, Muzammal and Khan, Salman and Shah, Mubarak and Khan, Fahad Shahbaz},
    title     = {Video-FocalNets: Spatio-Temporal Focal Modulation for Video Action Recognition},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    year      = {2023},
}
``` -->

## Contact
<!-- If you have any questions, please create an issue on this repository or contact at hullah2024@fau.edu. -->

## Acknowledgements
Our code is based on [MMDetection](https://github.com/open-mmlab/mmdetection) repositoru. We thank the authors for releasing their code. If you use our code, please consider citing these works as well.
