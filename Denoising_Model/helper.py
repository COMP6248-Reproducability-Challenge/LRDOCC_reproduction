import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

import numpy as np
import sklearn

from torch import nn
from torch import optim

from urllib.request import urlopen
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import normalize
from sklearn.svm import OneClassSVM
from sklearn import metrics
from sklearn.metrics import roc_auc_score, roc_curve, auc

from tqdm.autonotebook import tqdm
from itertools import chain

def K9_dataloader(dataset="fMNIST",batch_size=16,shuffle_train_set=False):
    #Setup the preprocess function
    if dataset == "fMNIST":
        fmnist_image_dim = 784
        preprocess_input = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.view(fmnist_image_dim)) # flatten into vector
        ])

    else:
        cifar10_image_dim = 3072
        preprocess_input = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.view(cifar10_image_dim)) # flatten into vector
        ])

    #Download the data
    if dataset == "fMNIST":
        train_set = torchvision.datasets.FashionMNIST(root='./data/FashionMNIST', train=True, download=True, transform=preprocess_input)
        test_set  = torchvision.datasets.FashionMNIST(root='./data/FashionMNIST', train=False, download=True, transform=preprocess_input)

    if dataset == "CIFAR10":
        train_set = torchvision.datasets.CIFAR10(root='./data/CIFAR10', train=True, download=True, transform=preprocess_input)
        test_set  = torchvision.datasets.CIFAR10(root='./data/CIFAR10', train=False, download=True, transform=preprocess_input)

    if dataset == "CIFAR100":
        train_set = torchvision.datasets.CIFAR100(root='./data/CIFAR100',train=True, download=True, transform=preprocess_input)
        test_set  = torchvision.datasets.CIFAR100(root='./data/CIFAR100',train=False, download=True, transform=preprocess_input)

    #Setup the loaders
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=shuffle_train_set)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size)

    print()
    print(f"train_loader batches: {len(train_loader)}, test_loader batches: {len(test_loader)}, batch_size: {batch_size}")
    print(f"train_shuffle set to {shuffle_train_set}")
    print(f"train_set length: {len(train_set)}, test_set length: {len(test_set)}")
    print(f"train_loader and test_loader ready for {dataset}.")
    print()

    return train_loader, test_loader

def K9_stratified_class_sample(data_ft, data_lb, samp_per_cls=500, random_seed = False):
    print(f"Choosing {samp_per_cls} samples per class.  data_lb has {len(np.unique(data_lb))} classes.")

    idxx = np.array([],dtype="int")
    for cls in np.unique(data_lb):
        idxx = np.append(idxx, np.random.choice(np.where(data_lb == cls)[0],size=samp_per_cls))

    data_ft_sampled, data_lb_sampled = data_ft[idxx,:], data_lb[idxx]

    print("Summary of data_lb_sampled class sample size:")
    i = 1
    for cls in np.unique(data_lb):
        if i % 10 != 0:
            print(f"Cls {cls}: {np.sum(data_lb_sampled == cls)}",end= " | ")
        else:
            print(f"Cls {cls}: {np.sum(data_lb_sampled == cls)}")
        i += 1
    return data_ft_sampled, data_lb_sampled

def K9_OCSVM(X_train, y_train, X_test, y_test, kernel_type='rbf'):
    print("Starting K9 OC-SVM, kernel_type = "+kernel_type)
    print(f"X_train.shape: {X_train.shape}, y_train.shape: {y_train.shape}")
    print(f"X_test.shape: {X_test.shape}, y_test.shape: {y_test.shape}")
    print()

    train_classes, per_class_auc = np.unique(y_train), []
    for one_class in train_classes:
        # Normalise train set; set gamma (both as per the paper)
        OC_X_train = X_train[y_train==one_class,:]
        OC_X_train_normalised = normalize(OC_X_train,norm='l2',axis=1) # normalise by L2 norm on a **row** basis

    # Fit the OC-SVM
    if kernel_type == 'rbf':
        gamma = 10/(np.var(OC_X_train_normalised) * OC_X_train_normalised.shape[1])
        clf = OneClassSVM(kernel='rbf', gamma=gamma)

    else:
        clf = OneClassSVM(kernel='linear')

    clf = clf.fit(OC_X_train_normalised)

    # Use fitted model to make predictions on test
    X_test_normalised = normalize(X_test,norm='l2',axis=1) #normalise by L2 norm on a **row** basis
    y_test_pred = clf.predict(X_test_normalised)
    y_test_pred_scores = clf.score_samples(X_test_normalised)
    y_test_pred_AUC = roc_auc_score(1.*(y_test==one_class),y_test_pred_scores)

    # Save AUC
    per_class_auc = np.append(per_class_auc, y_test_pred_AUC)

    #Print out results
    y_test_tp, y_test_tn = np.sum(y_test_pred[y_test==one_class] == 1), np.sum(y_test_pred[y_test!=one_class] == -1)
    print(f"Class: {one_class}")
    print(f"AUC score: {y_test_pred_AUC: .4f}, Accuracy: {(y_test_tp+y_test_tn)/len(y_test): .4f}")
    print(f"OC_X_train shape: {OC_X_train.shape}, X_test shape: {X_test.shape}.")
    print(f"Test: n_inlier: {np.sum(y_test==one_class)} ; n_outlier: {np.sum(y_test!=one_class)}, After fitting, test TP: {y_test_tp}, test TN: {y_test_tn}.")
    print()

    print(f"Unweighted mean AUC: {np.mean(per_class_auc): .4f}")
    return per_class_auc
