This repository contains a part of supplementary Python code for our paper[(preprint link)](https://arxiv.org/abs/2105.07107) on out-of-distribution detection and robustness to distributional shift peer-reviewed and published on IEEE International Conference on Machine Learning and Applications [(IEEE ICMLA)](https://www.icmla-conference.org/) 2021. 

This method was able to achieve state-of-the-art performance on various Vision and Language Out-of-distribution detection tasks.
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/an-effective-baseline-for-robustness-to/out-of-distribution-detection-on-cifar-10)](https://paperswithcode.com/sota/out-of-distribution-detection-on-cifar-10?p=an-effective-baseline-for-robustness-to)

## Prerequisites

### Environment
The code was tested with Python 3.7+ using PyTorch v1.2+. Additional Python libraries may be required, as specified in `requirements.txt`.

### Datasets
The pre-training and evaluation code makes use of the following datasets:

Dataset | URL
-- | --
CIFAR-10/100 | `https://www.cs.toronto.edu/~kriz/cifar.html`
Tiny ImageNet | `https://tiny-imagenet.herokuapp.com/`
Textures | `https://www.robots.ox.ac.uk/~vgg/data/dtd/`
LSUN | `https://www.yf.io/p/lsun`
SVHN | `http://ufldl.stanford.edu/housenumbers/`
Places | `http://places.csail.mit.edu/`

Training with background data further requires ILSVRC'12 or Tiny Images:

Dataset | URL
-- | --
ILSVRC 2012 | `http://www.image-net.org/challenges/LSVRC/2012/`
80M Tiny Images | `https://groups.csail.mit.edu/vision/TinyImages/`


All datasets must be downloaded and prepared under paths specified in `datasets/__init__.py`. **Currently it is hardcoded to sushil's local device. **
ILSVRC'12 data should be further processed into LMDB format for faster data loading.

## Instructions

### Pre-training
Use `train.py` to pretrain models on the in-distribution datasets. Models are automatically saved under `checkpoints/` folder. Example:
```
python train.py --gpus 0 -a wrn40 -d cifar10 --epochs 100
```

### Important: For Adversarial background resampling
Use `cv_train_bg_resample.py` to perform adversarial resampling on background dataset. Resampling weights are automatically saved under `checkpoints/` folder. Example:
```
python cv_train_bg_resample.py --gpus 0 -a wrn40 -id cifar10 -od tiny_images --epochs 50 --load-path path/to/model.pth
```

### Fine-tuning with background data using Abstention
Use `train_bg.py` to finetune models using background data. Supports full background dataset, uniformly subsampled background, or resampled background using learned weights. Example:
```
python train_bg.py --gpus 0 -a wrn40 -id cifar10 -od tiny_images --epochs 50 --load-path path/to/model.pth
python train_bg.py --gpus 0 -a wrn40 -id cifar10 -od tiny_images --epochs 50 --load-path path/to/model.pth --resample random -p 0.1
python train_bg.py --gpus 0 -a wrn40 -id cifar10 -od tiny_images --epochs 50 --load-path path/to/model.pth --resample path/to/resample/weights.pth -p 0.1
```

### Evaluation: OOD detection
Use `test_ood.py` to evaluate OOD detection of trained models on one or more test sets. Example:
```
python test_ood.py --gpus 0 -a wrn40 -id cifar10 -od gaussian uniform textures lsun svhn places --out-ratio 0.2 --load-path path/to/model.pth
```

## References
1. [Background Data Resampling for Outlier-Aware Classification, Li, Yi and Vasconcelos, Nuno, CVPR 2020](https://openaccess.thecvf.com/content_CVPR_2020/html/Li_Background_Data_Resampling_for_Outlier-Aware_Classification_CVPR_2020_paper.html) | [code](https://github.com/JerryYLi/bg-resample-ood)
