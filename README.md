# RE' Learning and Evaluating Representations for Deep One-Class Classification

This repository contains code to reproduce results from Table 2 and Table 7 of the paper: ["Learning and Evaluating Representations for Deep One-Class Classification"](https://openreview.net/forum?id=HCSgyPUfeDj) as part of the COMP6248 UoS Reproducability Challenge.

![Paper_Figure1](https://github.com/COMP6248-Reproducability-Challenge/LRDOCC_reproduction/blob/main/Paper_Figure1.PNG | width = 80)

### Reproduction
Each folder contains scripts to generate the respective method representations and subsequently perform one class-classification with linear and RBF kernel OC-SVMs.

- ResNet18-50_Baseline_Model: reproduction of experiments on ResNet18 (random weights) and an ImageNet pre-trained ResNet50 on f-MNIST, CIFAR10 and CIFAR100.
- Denoising_Model: reproduction of experiments with a denoising autoencoder on fMNIST and CIFAR10.
- Rotation_Prediction_Model: reproduction of experiments with a rotation prediction ResNet18 network on fMNIST.
- SimCLR: reproduction of experiments with the [SimCLR](https://arxiv.org/abs/2002.05709) network on fMNIST and CIFAR10.
- Table_Means_Verification: verification of the row means of all tables in the paper.

### Requirements

    - Python=3.7
    - PyTorch=1.8
    - torchvision=0.9
    - scikit-learn=0.22

Team members:

- Niko Chazaridis (@chazarnik)
- Marios Christodoulou (@mchris7)
- Ian Simpson (@statsonthecloud)

---------------------------------------
