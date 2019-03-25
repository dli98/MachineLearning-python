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


from sklearn import svm

def Boundary_function(clf, X, Y):
    # get the separating hyperplane
    w = clf.coef_[0]
    print(w)
    a = -w[0] / w[1]   # 斜率
    x_min, x_max = X[:,0].min()*1.2, X[:,0].max()*1.1
    xx = np.linspace(x_min, x_max)
    yy = a * xx - (clf.intercept_[0]) / w[1]   # 点斜式方程

    # plot the parallels to the separating hyperplane that pass through the
    # support vectors
    b = clf.support_vectors_[0]
    yy_down = a * xx + (b[1] - a * b[0])
    b = clf.support_vectors_[-1]
    yy_up = a * xx + (b[1] - a * b[0])

    # plot the line, the points, and the nearest vectors to the plane
    plt.figure(figsize=(8,5))
    plt.plot(xx, yy, 'k-')
    plt.plot(xx, yy_down, 'k--')
    plt.plot(xx, yy_up, 'k--')
    # 圈出支持向量
    plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1],
                s=150, facecolors='none', edgecolors='k', linewidths=1.5)
    plt.scatter(X[:, 0], X[:, 1], c=Y.flatten(), cmap='rainbow')

    plt.axis('tight')
    plt.show()

    print(clf.decision_function(X))



if __name__ == '__main__':
    # text()
    X, Y = load_dataset()
    models = [svm.SVC(C, kernel='linear', gamma='auto') for C in [1, 100]]
    clfs = [model.fit(X, Y.ravel()) for model in models]
    title = ['SVM Decision Boundary with C = {} (Example Dataset 1'.format(C) for C in [1, 100]]
    for model, title in zip(clfs, title):
        Boundary_function(model, X, Y)
