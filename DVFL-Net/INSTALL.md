
## Installation Instructions

- Clone this repository:
```bash
git clone https://github.com/hayatkhan8660-maker/DVFL-Net.git
cd DVFL-Net
```

- Create a conda virtual environment and activate it:

```bash
conda create -n DVFL python=3.10 -y
conda activate DVFL
```

- Install `PyTorch==2.4.0` and `torchvision==0.19.0` with `CUDA==12.4`:

```bash
conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 cudatoolkit=12.4 -c pytorch
```

- Install `timm`:

```bash
pip install timm
```

- Install `Apex`:

```bash
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```
**Note**: make sure the compatibility of ``` Apex ``` with your ``` CUDA ``` version. Otherwise the code will throw errors.

- Install other prerequisite requirements:

```bash
pip install decord==0.6.0 einops==0.4.1 imgaug==0.4.0 numpy==1.22.3 pandas==1.4.2 Pillow==9.0.1 PyYAML==6.0 termcolor==2.3.0 thop yacs
```
