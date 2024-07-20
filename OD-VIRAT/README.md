<p align="center">
  <img src="Figures/od_virat.png" width="1000"/>
</p> 

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
  <!-- * [News](#rocket-News) -->
   * [Overview](#overview)
   * [Visualization](#visualization-visual-illustration-of-each-scene)
   * [Environment Setup](#environment-setup)
   * [Dataset Detail and Data Preparation](#dataset-detail-and-data-preparation)
   * [Training](#training)
   * [Evaluation](#evaluation)
   * [Citation](#citation)
   * [Acknowledgements](#acknowledgements)
<!--te-->

<!-- ## News -->

## Overview

**OD-VIRAT** is an object detection benchmark developed for object detection tasks in challenging surveillance environments. This dataset is constructed from videos of the [VIRAT Ground 2.0 Dataset](https://viratdata.org/), comprising 329 surveillance video captured through stationary ground cameras mounted at significant heights (mostly atop buildings), spanning 11 distinct scenes.
<hr>
<table class="small-font-table">
  <tr>
    <th rowspan="2">Dataset</th>
    <th colspan="3"># of Scenes</th>
    <th colspan="3"># of Videos</th>
    <th colspan="3"># of Frames</th>
    <th rowspan="2">Resolution</th>
  </tr>
  <tr>
    <td><b>train</b></td>
    <td><b>validation</b></td>
    <td><b>test</b></td>
    <td><b>train</b></td>
    <td><b>validation</b></td>
    <td><b>test</b></td>
    <td><b>train</b></td>
    <td><b>validation</b></td>
    <td><b>test</b></td>
  </tr>
  <tr>
    <td><b>OD-VIRAT Large</b></td>
    <td><p align="center">10</td>
    <td><p align="center">10</td>
    <td><p align="center">8</td>
    <td><p align="center">156</td>
    <td><p align="center">52</td>
    <td><p align="center">52</td>
    <td><p align="center">377686</td>
    <td><p align="center">137971</td>
    <td><p align="center">84339</td>
    <td><p align="center">(1920 &times 1080),(1280 &times 720)</td>
  </tr>
   <tr>
    <td><b>OD-VIRAT Tiny</b></td>
    <td><p align="center">10</td>
    <td><p align="center">10</td>
    <td><p align="center">8</td>
    <td><p align="center">156</td>
    <td><p align="center">52</td>
    <td><p align="center">52</td>
    <td><p align="center">12501</td>
    <td><p align="center">4573</td>
    <td><p align="center">2786</td>
    <td><span style="font-size: smaller;">(1920 &times 1080),(1280 &times 720)</span></td>
  </tr>
</table>
<hr>

## Visualization: Visual Illustration of Each Scene 

<!-- First row of images -->
<div style="display: flex; justify-content: space-around;">
  <img src="Figures/image1.jpg" width="198px" />
  <img src="Figures/image2.jpg" width="198px" />
  <img src="Figures/image3.jpg" width="198px" />
  <img src="Figures/image4.jpg" width="198px" />
  <img src="Figures/image5.jpg" width="198px" />
</div>

<!-- Second row of images -->
<div style="display: flex; justify-content: space-around; margin-top: 50px;">
  <img src="Figures/image6.jpg" width="198px" />
  <img src="Figures/image7.jpg" width="198px" />
  <img src="Figures/image8.jpg" width="198px" />
  <img src="Figures/image9.jpg" width="198px" />
  <img src="Figures/image10.jpg" width="198px" />
</div>

<!--
<p align="center">
  <img alt="Visualization Scuba Diving" src="figs/vis/scuba_diving.png" width="900"/>
</p>

<p align="center">
  <img alt="Visualization Threading Needle" src="figs/vis/threading_needle.png" width="900"/>
</p>

<p align="center">
  <img alt="Visualization Walking the Dog" src="figs/vis/walking_the_dog.png" width="900"/>
</p>

<p align="center">
  <img alt="Visualization Water Skiing" src="figs/vis/water_skiing.png" width="900"/>
</p>
-->

## Environment Setup
Please follow [INSTALL.md](./INSTALL.md) for preparing the environement and installation of prerequisite packages.

## Dataset Detail and Data Preparation

Please follow [DATA.md](./DATA.md) for dataset details and data preparation.

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
