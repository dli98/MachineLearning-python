import numpy as np
import tensorflow as tf
import math
import matplotlib.pyplot as plt
from scipy.io import loadmat


def load_dataset():
    train_dataset = loadmat('ex6data1.mat')
    X = train_dataset["X"]  # your train set features
    y = train_dataset["y"]  # your train set labels
    # cross validation set
    return X, y


def plotBoundary(clf, X):
    '''plot decision bondary'''
    x_min, x_max = X[:,0].min()*1.2, X[:,0].max()*1.1
    y_min, y_max = X[:,1].min()*1.1,X[:,1].max()*1.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500),
                         np.linspace(y_min, y_max, 500))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contour(xx, yy, Z)
