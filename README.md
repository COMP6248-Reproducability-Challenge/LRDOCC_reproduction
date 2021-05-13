# RE' Learning and Evaluating Representations for Deep One-Class Classification

This repository contains code to reproduce results from tables 2 and 7 of the paper: ["Learning and Evaluating Representations for Deep One-Class Classification"](https://openreview.net/forum?id=HCSgyPUfeDj) as part of the COMP6248 UoS Reproducability challenge.

### Reproduction
Each folder contains scripts to generate the respective method representations and subsequently perform one class-classification with linear and rbf kernel OC-SVMs.

- Denoising: reproduction of experiments with a denoising autoencoder on fMNIST and CIFAR10.
- Random_and_Imagenet: reproduction of experiments with a ResNet18(random weights) and a pre-trained resnet on fMNIST, CIFAR10 and CIFAR100.
- Rotation_prediction: reproduction of experiments with a rotation prediction ResNet18 network on fMNIST.
- SimCLR: reproduction of experiments with the [SimCLR](https://arxiv.org/abs/2002.05709) network on fMNIST and CIFAR10.

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
