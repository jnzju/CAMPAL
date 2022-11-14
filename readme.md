# Active Learning with Controllable Augmentation-induced Acquisition



This is a [PyTorch](https://pytorch.org/) implementation of our ICLR 2022 paper [CAMPAL](openreview.net/forum?id=vE93gf9kYkf).

Title: Active Learning with Controllable Augmentation-induced Acquisition

Authors: Jianan Yang, Haobo Wang, Sai Wu, Gang Chen, Junbo Zhao

```
@inproceedings{anonymous2023active,
	title={Active Learning with Controllable Augmentation Induced Acquisition},
	author={Anonymous},
	booktitle={Submitted to The Eleventh International Conference on Learning Representations },
	year={2023},
	url={https://openreview.net/forum?id=vE93gf9kYkf},
	note={under review}
}
```



## Start Running CAMPAL

### Data Preparation

The datasets included in this paper are all public datasets available from official websites or mirrors. Just put them in the directory as follows.

```
data
├── Caltech-101
│   ├── Annotations
│   │   ├── accordion
│   │   ├── ...
│   │   └── yin_yang
│   └── data
│       ├── accordion
│       ├── ...
│       └── yin_yang
├── Caltech-256
│   └── data
│       ├── 001.ak47
│       ├── ...
│       └── 257.clutter
├── cifar10
│   ├── cifar-10-batches-py
│   └── cifar-10-python.tar.gz
├── cifar100
│   ├── cifar-100-python
│   └── cifar-100-python.tar.gz
├── FashionMNIST
│   └── raw
│       ├── t10k-images-idx3-ubyte
│       ├── t10k-images-idx3-ubyte.gz
│       ├── t10k-labels-idx1-ubyte
│       ├── t10k-labels-idx1-ubyte.gz
│       ├── train-images-idx3-ubyte
│       ├── train-images-idx3-ubyte.gz
│       ├── train-labels-idx1-ubyte
│       └── train-labels-idx1-ubyte.gz
├── imagenet
│   ├── meta
│   │   ├── det_synset_words.txt
│   │   ├── imagenet.bet.pickle
│   │   ├── imagenet_mean.binaryproto
│   │   ├── synsets.txt
│   │   ├── synset_words.txt
│   │   ├── test.txt
│   │   ├── train.txt
│   │   └── val.txt
│   ├── meta.bin
│   ├── train
│   │   ├── n01440764
│   │   ├── ...
│   │   └── n15075141
│   └── val
│       ├── n01440764
│       ├── ...
│       └── n15075141
├── SVHN
│   ├── test_32x32.mat
│   └── train_32x32.mat
└── tiny-imagenet-200
    ├── test
    │   └── images
    ├── train
    │   ├── n01443537
    │   ├── ...
    │   └── n12267677
    ├── val
    │   ├── n01443537
    │   ├── ...
    │   └── n12267677
    ├── wnids.txt
    └── words.txt

deep-active-learning
├── architectures
├── datasets
├── evaluation
├── main.py
├── query_strategies
├── test.sh
├── readme.md
└── utils
```



### Start Running

We provide the following shell codes for model training in `test.sh`. Just following the format provided in `test.sh` when running our codes. We also provide a illustration for training configurations as follows.

1. Basic Active Learning setting
   - `--strategy`: The basic AL strategy to apply, including EntropySampling, LeastConfidence, MarginSampling, BALDDropout, CoreSet, and BadgeSampling
   - `--num-init-labels`: The size of initial labeled pool
   - `--n-cycle`: The number of active learning cycles
   - `--num-query`: The number of unlabeled samples to query at each cycle
   - `--subset`: Perform evaluation on a subset of unlabeled samples to reduce computational costs on large datasets
   - `--updating`: Whether to load the model parameters obtained from the last cycle
   - `--n-epoch`: The number of training epochs for each cycle

2. Setting for augmentation policies in CAMPAL
   - `--aug-lab-on`: Just a flag for configuration on augmentations performed on labeled pool, no practical use
   - `--aug-ratio-lab`: The number of single-image augmentations performed on each labeled sample
   - `--mix-ratio-lab`: The number of image-mixing augmentations performed on each labeled sample, only available for `StrengthGuidedAugment`
   - `--aug-lab-training-mode`: The type of augmentation optimization strategy to use, including `StrengthGuidedAugment`, `RandAugment`, `AutoAugment`, `TrivialAugmentWide`
   - `--aug-ulb-on`: Just a flag for configuration on augmentations performed on unlabeled pool, no practical use
   - `--aug-ratio-ulb`: The number of single-image augmentations performed on each unlabeled sample
   - `--mix-ratio-ulb`: The number of image-mixing augmentations performed on each unlabeled sample, only available for `StrengthGuidedAugment`
   - `--aug-ulb-evaluation-mode`: The type of augmentation optimization strategy to use on unlabeled pool, including `StrengthGuidedAugment`, `RandAugment`, `AutoAugment`, `TrivialAugmentWide`
   - `--aug-metric-ulb`: How to aggregate informativeness induced from augmentations, including `normal`, `max`, `min`, `sum`, `density` for score-based strategies (EntropySampling, LeastConfidence, MarginSampling, BALDDropout), and `standard`, `chamfer`, `hausdorff` for representation-based strategies.