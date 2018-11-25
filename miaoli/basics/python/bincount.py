#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2018/9/17 21:19
# @Author  : Miao Li
# @File    : bincount.py

"""bincount
"""

import numpy as np
from sklearn import metrics

y_train_ = np.array([1,1,2,2,3,3,0,4])
y_train_pred = np.array([2,2,1,1,3,3,0,0])

ARI = metrics.adjusted_rand_score(y_train_, y_train_pred)
print("ARI", ARI)

AMI = metrics.adjusted_mutual_info_score(y_train_, y_train_pred)
print("AMI", AMI)

H, C, VM = metrics.homogeneity_completeness_v_measure(y_train_, y_train_pred)
print("H", H)
print("C", C)
print("VM", VM)

right = 0.
for i in range(3):
    print(y_train_pred == i)
    _ = np.bincount(y_train_[y_train_pred == i])
    right += _.max()

print('acc: %s' % (right / len(y_train_)))