#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2018/10/19 16:15
# @Author  : Miao Li
# @File    : evaluation.py

"""
evaluate clustering results
"""
import numpy as np
from sklearn import metrics
from sklearn.utils.linear_assignment_ import linear_assignment


def evaluate_clustering(y, y_pred):
    """
    :param y: labels of examples, eg. [1, 1, 1, 0]
    :param y_pred: labels of examples, eg. [0, 0, 1, 1]
    :return: scores
    """
    # ACC, based acc in VaDE
    y = np.array(y)
    y_pred = np.array(y_pred)
    assert y_pred.size == y.size
    D = max(y_pred.max(), y.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y[i]] += 1
    ind = linear_assignment(w.max() - w)
    ACC =  sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size

    # ARI, AMI, VM
    ARI = metrics.adjusted_rand_score(y, y_pred)

    AMI = metrics.adjusted_mutual_info_score(y, y_pred)

    # VM equals NMI
    VM = metrics.v_measure_score(y, y_pred)

    return ACC, ARI, AMI, VM


def plot_2d(points, labels, epoch, location):
    """

    :param points: 2d points, x and y
    :param labels: labels of classes
    :param epoch: epoch
    :return: None
    """
    import matplotlib.pyplot as plt
    plt.figure()
    plt.scatter(points[:, 0], points[:, 1], c=labels)
    plt.savefig(location + str(epoch % 10) + ".png")
    plt.close()


if __name__ == "__main__":
    y = [1, 1, 1, 0]
    y_pred = [0, 0, 1, 1]
    evaluate_clustering(y, y_pred)