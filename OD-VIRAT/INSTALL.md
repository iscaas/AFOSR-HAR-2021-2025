## Installation Instructions

- Clone this repo:
<!--
```bash
git clone https://github.com/TalalWasim/Video-FocalNets
cd Video-FocalNets
``` -->

- Create a conda virtual environment and activate it:

```bash
conda create -n [env_name] python=3.10 -y
conda activate [env_name]
```

- Install `PyTorch==2.3.0` and `torchvision==0.18.0` with `CUDA==12.1`:

```bash
conda install pytorch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 cudatoolkit=12.1 -c pytorch
```

- Clone `mmdetection` repo inside `OD-VIRAT` directory:
```bash
cd OD-VIRAT
git clone https://github.com/open-mmlab/mmdetection.git
```

- Install the following prerequisite libraries for  `mmdetection`.
```bash
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"
```

- Now install  `mmdetection`.
```bash
cd mmdetection
pip install -v -e .
```   

- Install other requirements:
