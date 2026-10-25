#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 15 19:16:08 2025
Implementation of Accessible model predicts response in hormone receptor positive HER2 negative breast cancer
receiving neoadjuvant chemotherapy (2025).
"""

import numpy as np
import pandas as pd
from sklearn.frozen import FrozenEstimator

# -- [Machine Learning classifiers]
# -- Linear model
from sklearn.linear_model import LogisticRegression
# -- ensemble
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

# -- neighbors
from sklearn.neighbors import KNeighborsClassifier
# -- svm
from sklearn import svm


# -- [model selection]
from sklearn.model_selection import StratifiedKFold

# -- [Metrics]
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import roc_auc_score


# -- [Feature selector]
from sklearn.utils import shuffle

from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split

import smote_variants as sv
import random

from data_preparation import prepare_task, scale_fold
# Hyperparameter setting
seed = 9
numfolds = 5
save_confusion_matrix = False

# set random seed
random.seed(seed)
np.random.seed(seed)


# -- [Tarea a resolver]: 'pCR', 'OS' o 'DFS'
task = 'DFS'  # 'pCR', 'OS' o 'DFS'

# load data + preparación, limpieza y transformación
# (en OS/DFS, include_post_treatment=False deja solo las variables previas al tratamiento)
samples, labels, feature_names, continuous_idx = prepare_task(task, data_dir='.', include_post_treatment=True)

# Shuffle data
samples, labels = shuffle(samples, labels, random_state=seed)

num_classes = len(np.unique(labels))

print("Data Shape:", samples.shape, labels.shape)

skf = StratifiedKFold(n_splits=numfolds)
skf.get_n_splits(samples, labels)
acc_list = np.zeros(numfolds)
f1_list = np.zeros(numfolds)
auc_list = np.zeros(numfolds)



for fold, (train_index, test_index) in enumerate(skf.split(samples, labels)):
    print(f"\n\nFold {fold+1}:")

    # [training and testing data]
    X_tr = samples[train_index]
    y_tr = labels[train_index]
    X_ts = samples[test_index]
    y_ts = labels[test_index]
    # escalado ajustado solo con el train del fold (evita fuga de información)
    X_tr, X_ts = scale_fold(X_tr, X_ts, continuous_idx)
    print("Tr:", X_tr.shape)
    print("Ts:", X_ts.shape)

    #oversampler = sv.NRSBoundary_SMOTE()
    oversampler = sv.SMOTE()
    #oversampler = sv.Assembled_SMOTE(pop=2, thres=0.5)
    X_tr, y_tr = oversampler.sample(X_tr, y_tr)

    # -- OK
    clf = LogisticRegression(random_state=seed, class_weight='balanced', penalty='elasticnet', solver='saga', C=0.5, l1_ratio=0.5)
    #clf = svm.SVC(C=0.5, kernel='rbf', gamma="scale")
    #clf = AdaBoostClassifier(n_estimators=100, random_state=seed)
    #clf = RandomForestClassifier(n_estimators=200, random_state=seed)

    # --- training phase
    X_train, X_valid, y_train, y_valid = train_test_split(X_tr, y_tr, test_size=0.20, random_state=seed)  # ~10% of total for valid

    clf.fit(X_train, y_train)
    calibrated_clf = CalibratedClassifierCV(
        estimator=FrozenEstimator(clf),
        method='sigmoid'
    )
    # Calibrate the probabilities on the validation set (crucially, data unseen during initial fit)
    calibrated_clf.fit(X_valid, y_valid)

    # --- make predictions
    y_p = calibrated_clf.predict(X_ts)

    # --- get metrics
    acc = accuracy_score(y_ts, y_p)
    f1 = f1_score(y_ts, y_p)
    auc = roc_auc_score(y_ts, y_p)

    print('Acc: %.2f \t F1: %.2f \t AUC: %.4f' % (acc * 100, f1 * 100, auc))

    acc_list[fold] = acc
    f1_list[fold] = f1
    auc_list[fold] = auc

    X_train = pd.DataFrame(X_tr)


print("\n\n[Average classification results]")
print('Machine Learning classifier \tAcc: %.2f \t F1-score: %.2f \t AUC: %.4f' % (acc_list.mean()*100, f1_list.mean()*100, auc_list.mean()))







