# Mono3DVGv2: Bootstrapping vision–language transformer for monocular 3D visual grounding

This repository hosts the official implementation of [Bootstrapping vision–language transformer for monocular 3D visual grounding](https://ietresearch.onlinelibrary.wiley.com/doi/full/10.1049/ipr2.13315) based on the excellent work [Mono3DVG](https://github.com/ZhanYang-nwpu/Mono3DVG).

In this work, We introduce an improved version of the Mono3DVG model, named Mono3DVG-TRv2, which integrates a large-scale vision feature extractor, a high-performance transformer architecture, and a fusion technique of tailored vision–language. We also present vision–language fusion embeddings to improve mono-depth map prediction. And this method can also begeneralised to enhance the performance of multi-modal representation on this benchmark based on one-stage methods.

![](https://s2.loli.net/2025/03/21/3woCUxLnBg8TjJr.png)

![](https://s2.loli.net/2025/03/21/cRYeirXtkFdEz2O.png)

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

#### 1.5 Download [Mono3DRefer](https://drive.google.com/drive/folders/1ICBv0SRbRIUnl_z8DVuH8lz7KQt580EI?usp=drive_link) datasets and prepare the directory structure as:
```
    │Mono3DVGV2/
    ├──Mono3DRefer/
    │   ├──images/
    │   │   ├──000000.png
    │   │   ├──...
    │   ├──calib/
    │   │   ├──000000.txt
    │   │   ├──...
    │   ├──Mono3DRefer_train_image.txt
    │   ├──Mono3DRefer_val_image.txt
    │   ├──Mono3DRefer_test_image.txt
    │   ├──Mono3DRefer.json
    │   ├──test_instanceID_split.json
    ├──configs
    │   ├──simple-bridge-swin-dab-without-adapter.yaml
    │   ├──...
    ├──lib
    │   ├──datasets/
    │   │   ├──...
    │   ├──helpers/
    │   │   ├──...
    │   ├──losses/
    │   │   ├──...
    │   ├──models/
    │   │   ├──...
    |——pretrained-models
    |   ├──roberta-base
    |   |——swin_large_patch4_window7_224
    |   |——mono3dvg
    |   |  |——checkpoint_best.pth
    │   ├──...
    ├──utils
    │   ├──...
    ├──outputs    #  save_path
    │   ├──simple-bridge-swin-dab-without-adapter-6l
    │   │   ├──...
    ├──test.py
    ├──train.py
```
You can also change the dataset path at "root_dir" in `configs/simple-bridge-swin-dab-without-adapter.yaml`.

### 1.6 Download pre-trained model and checkpoint

You must download the Pre-trained model of **RoBERTa** and **Mono3DVG-TR** from [Mono3DVG](https://github.com/ZhanYang-nwpu/Mono3DVG?tab=readme-ov-file#15-download-pre-trained-model-and-checkpoint) .

The Pre-trained model of **backbone** will auto download by `timm` [PyTorch Image Models](https://huggingface.co/timm) , or you can change the backbone version at `pretrained_backbone_path` in `configs/simple-bridge-swin-dab-without-adapter.yaml`.

You can download the checkpoint we provide to evaluate the Mono3DVG-TR model.

|Models|Links|File Path|
|:----:|:----:|:----:|
| simple-bridge-swin-dab-without-adapter-6l | [model](https://pan.quark.cn/s/96dfe88ff342) | outputs/simple-bridge-swin-dab-without-adapter-6l |

### 2. Get Strated

#### 2.1 Train

You can modify the settings of GPU, models and training in `configs/simple-bridge-swin-dab-without-adapter.yaml`
```vim
python train.py --config configs/simple-bridge-swin-dab-without-adapter.yaml
```

#### 2.2 Test
The best checkpoint will be evaluated as default. You can change it at "pretrain_model: 'checkpoint_best.pth'" in configs/simple-bridge-swin-dab-without-adapter.yaml:
```vim
python test.py --config configs/simple-bridge-swin-dab-without-adapter.yaml
```

#### 2.3 DDP Train
Our model supports [Accelerate](https://huggingface.co/docs/accelerate/quicktour), which offers a unified interface for launching and training on different distributed setups. This allows you to easily scale your PyTorch code for training and inference on distributed setups with hardware like GPUs and TPUs.

Firstly, set DDP config with a unified interface:
```vim
accelerate config
```
You can check your setup:
```vim
accelerate test
```
When the DDP setup completed, enjoying the distributed train:
```vim
accelerate launch train.py --config configs/simple-bridge-swin-dab-without-adapter.yaml
```

## Visualization

1. Qualitative results of our Mono3DVG-TRv2.
  ![](https://s2.loli.net/2025/03/21/37yRBpMA2ibn5rf.png)
2. Illustration of attention mechanisms for query relevance in the decoder’s ﬁnal layer.
  ![](https://s2.loli.net/2025/03/21/q9SPGR3x5Ky17lN.png)

## Results

1. Comparison with baselines.
  ![](https://s2.loli.net/2025/03/21/okSLWQ1d5a4Dvtj.png)
2. Results for ’near’-’medium’-’far’ subsets and ’easy’-’moderate’-’hard’ subsets.
  ![](https://s2.loli.net/2025/03/21/I5UQxV28a4nljc6.png)

## Citation
```
@article{lei2025bootstrapping,
  title={Bootstrapping vision--language transformer for monocular 3D visual grounding},
  author={Lei, Qi and Sun, Shijie and Song, Xiangyu and Song, Huansheng and Feng, Mingtao and Wu, Chengzhong},
  journal={IET Image Processing},
  volume={19},
  number={1},
  pages={e13315},
  year={2025},
  publisher={Wiley Online Library}
}
```

## Acknowlegment
Our code is based on (AAAI2024)[Mono3DVG](https://github.com/ZhanYang-nwpu/Mono3DVG), We sincerely appreciate their contributions and authors for releasing source codes and Mono3DRefer dataset. 
