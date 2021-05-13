import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18
from torch.utils.data import Dataset, DataLoader, Subset

import numpy as np
import os
import gc
import copy
import time as tm



class LRDOCCRotNet18(nn.Module):
    def __init__(self, base_model):
        super(LRDOCCRotNet18, self).__init__()
        self.base_model = base_model
        self.HEAD_DIMS = 512
        self.ROT_CLASSES = 4

        self.projection_head = nn.Sequential(
                nn.Linear(self.HEAD_DIMS, self.HEAD_DIMS),
                nn.ReLU(),
                nn.Linear(self.HEAD_DIMS, self.HEAD_DIMS),
                nn.ReLU(),
                nn.Linear(self.HEAD_DIMS, self.HEAD_DIMS),
                nn.ReLU(),
                nn.Linear(self.HEAD_DIMS, self.ROT_CLASSES)
        )
    
    def forward(self, x):
        x = self.base_model(x)
        x = self.projection_head(x)
        
        return x


from sklearn import svm
from sklearn.metrics import classification_report, roc_curve, auc, roc_auc_score
from sklearn.preprocessing import normalize

import sklearn
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import normalize
from sklearn.svm import OneClassSVM
from sklearn import metrics
from sklearn.metrics import roc_auc_score, roc_curve, auc


def K9_OCSVM(X_train, y_train, X_test, y_test, kernel_type='rbf'):

  print("Starting K9 OC-SVM, kernel_type = "+kernel_type)
  print(f"X_train.shape: {X_train.shape}, y_train.shape: {y_train.shape}")
  print(f"X_test.shape: {X_test.shape}, y_test.shape: {y_test.shape}")
  print()
  since = tm.time()
  train_classes, per_class_auc = np.unique(y_train), []
  for one_class in train_classes:
    #Normalise train set; set gamma (both as per the paper)
    OC_X_train = X_train[y_train==one_class,:]
    OC_X_train_normalised = normalize(OC_X_train,norm='l2',axis=1)   #normalise by L2 norm on a **row** basis
    
    #Fit the OC-SVM
    if kernel_type == 'rbf':
      gamma = 100/(np.var(OC_X_train_normalised) * OC_X_train_normalised.shape[1])
      clf = OneClassSVM(kernel='rbf', gamma=gamma)
    else:
      clf = OneClassSVM(kernel='linear')
    clf = clf.fit(OC_X_train_normalised)

    #Use fitted model to make predictions on test
    X_test_normalised = normalize(X_test,norm='l2',axis=1)   #normalise by L2 norm on a **row** basis
    y_test_pred = clf.predict(X_test_normalised)
    y_test_pred_scores = clf.score_samples(X_test_normalised)
    y_test_pred_AUC = roc_auc_score(1.*(y_test==one_class),y_test_pred_scores)

    #save AUC
    per_class_auc = np.append(per_class_auc, y_test_pred_AUC)

    #Print out results
    y_test_tp, y_test_tn = np.sum(y_test_pred[y_test==one_class] == 1), np.sum(y_test_pred[y_test!=one_class] == -1)
    print(f"Class: {one_class}")
    print(f"AUC score: {y_test_pred_AUC: .4f}, Accuracy: {(y_test_tp+y_test_tn)/len(y_test): .4f}")
    print(f"OC_X_train shape: {OC_X_train.shape}, X_test shape: {X_test.shape}.")
    print(f"Test: n_inlier: {np.sum(y_test==one_class)} ; n_outlier: {np.sum(y_test!=one_class)}, After fitting, test TP: {y_test_tp}, test TN: {y_test_tn}.")
    print()
  print(f"Unweighted mean AUC: {np.mean(per_class_auc): .4f}")

  time_elapsed = tm.time() - since
  print('Classification complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

  return per_class_auc