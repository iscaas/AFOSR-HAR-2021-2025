<p align="center">
  <img src="Figures/od_virat.png" width="1000"/>
</p> 

# OD-VIRAT: A Large-Scale Benchmark for Object Detection in Realistic Surveillance Environments
[Hayat Ullah](https://scholar.google.com.pk/citations?user=xnXPj0UAAAAJ&hl=en),
[Abbas Khan](https://scholar.google.com.pk/citations?user=k-HJxNAAAAAJ&hl=en&oi=sra),
[Arslan Munir](https://scholar.google.com.pk/citations?user=-P9waaQAAAAJ&hl=en)


[![paper](https://img.shields.io/badge/ACM-TOMM-<COLOR>.svg)]() [**Under Review**]
<hr />

> **Abstract:**
>Realistic human surveillance datasets are crucial for training and evaluating computer vision models under real-world conditions, facilitating the development of robust algorithms for human and human-interacting object detection in complex environments. These datasets need to offer diverse and challenging data to enable a comprehensive assessment of model performance and the creation of more reliable surveillance systems for public safety. To this end, we present two visual object detection benchmarks named OD-VIRAT Large and OD-VIRAT Tiny, aiming at advancing visual understanding tasks in surveillance imagery. The video sequences in both benchmarks cover 10 different scenes of human surveillance recorded from significant height and distance. The proposed benchmarks offer rich annotations of bounding boxes and categories, where OD-VIRAT Large has 8.7 million annotated instances in 599,996 images and OD-VIRAT Tiny has 288,901 annotated instances in 19,860 images. This work also focuses on benchmarking state-of-the-art object detection architectures, including RETMDET, YOLOX, RetinaNet, DETR, and Deformable-DETR on this object detection-specific variant of VIRAT dataset. To the best of our knowledge, it is the first work to examine the performance of these recently published state-of-the-art object detection architectures on realistic surveillance imagery under challenging conditions such as complex backgrounds, occluded objects, and small-scale objects. The proposed benchmarking and experimental settings will help in providing insights concerning the performance of selected object detection models and set the base for developing more efficient and robust object detection architectures.

## Table of Contents
<!--ts-->
  <!-- * [News](#rocket-News) -->
   * [Visualization](#visualization-visual-illustration-of-each-scene)
   * [Environment Setup](#environment-setup)
   * [Dataset Detail and Data Preparation](#dataset-detail-and-data-preparation)
   * [Training](#training)
   * [Evaluation](#evaluation)
   * [Citation](#citation)
   * [Acknowledgements](#acknowledgements)
<!--te-->

<!-- ## News -->

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
To train a specific model on the OD-VIRAT Tiny dataset, run the following command:
```bash
sbatch --mem=30G --time=40:00:00 --gres=gpu:1 --nodes=1 trainer.sh config
```
- ```sbatch```: Submits the job to the SLURM scheduler.
- ```--mem=30G```: Requests 30 GB of memory for the job.
- ```--time=40:00:00```: Sets a maximum job run time of 40 hours.
- ```--gres=gpu:1```: Requests 1 GPU for the job.
- ```--nodes=1```: Allocates 1 node for the job.
- ```trainer.sh config```: Runs the ```trainer.sh``` script with ```config``` as an argument.
  
**Or**

```bash
--launcher slurm --mem=30G --time=40:00:00 --gres=gpu:1 --nodes=1 trainer.sh config
```
- ```--launcher```: Submits the job to the SLURM scheduler.
- ```--mem=30G```: Requests 30 GB of memory for the job.
- ```--time=40:00:00```: Sets a maximum job run time of 40 hours.
- ```--gres=gpu:1```: Requests 1 GPU for the job.
- ```--nodes=1```: Allocates 1 node for the job.
- ```trainer.sh config```: Runs the ```trainer.sh``` script with ```config``` as an argument.

For example, to train the ```Video-FocalNet-B``` with a single GPU on ```OD-VIRAT Tiny```:

For example, to train the **Video-FocalNet-B** with a single GPU on **OD-VIRAT Tiny**:

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
If you have any questions, feel free to open an issue on this repository or reach out at hullah2024@fau.edu.

## Acknowledgements
Our code is based on [MMDetection](https://github.com/open-mmlab/mmdetection) repositoru. We thank the authors for releasing their code. If you use our code, please consider citing these works as well.
