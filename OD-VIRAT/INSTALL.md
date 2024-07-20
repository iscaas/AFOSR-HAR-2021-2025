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

- Install the following prerequisite libraries for  `mmdetection`:
```bash
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"
```

- Now install  `mmdetection`:
```bash
cd mmdetection
pip install -v -e .
```   

- Verify the `mmdetection` installation using the following command:
  
**step1:** 
```bash
mim download mmdet --config rtmdet_tiny_8xb32-300e_coco --dest .
```
**step2:**
```bash
python demo/image_demo.py demo/demo.jpg rtmdet_tiny_8xb32-300e_coco.py --weights rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth --device cpu
```
  
- Install other requirements:
