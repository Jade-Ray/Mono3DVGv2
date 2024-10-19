# Mono3DVGv2

## Mono3DVG-TRv2 Codebase
### 1. Enviroment and Installation

#### 1.1 Clone this project and create a conda environment: python>=3.8, our version of python == 3.12.7

#### 1.2 Install pytorch and torchvision matching your CUDA version: torch >= 1.9.0, our version of torch == 2.4.1, pytorch-cuda==11.8

#### 1.3 Install Huggingface accelerate, transformers, and diffusers: accelerate >= 1.0.0, transformers >= 4.40.0, diffusers >= 0.30.0, our version of accelerate == 1.0.0, transformers == 4.40.2, diffusers == 0.30.2

```bash
conda install -c conda-forge accelerate transformers diffusers
```

#### 1.4 Install other dependencies

```bash
pip install -U albumentations timm ninja wandb
```
