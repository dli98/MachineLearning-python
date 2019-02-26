import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.optimize import minimize


def load_data(path):
    data = loadmat(path)
    x = data['X']
    y = data['y']
    return x, y


def plot_an_image(X, y):
    pick_one = np.random.randint(0, 5000)
    image = X[pick_one, :]
    plt.matshow(image.reshape((20, 20)))
    plt.show()
    print('this should be {}'.format(y[pick_one]))


def plot_100_image(X):
    """
    随机画100个数字
    """
    sample_idx = np.random.choice(np.arange(X.shape[0]), 100)  # 随机选100个样本
    sample_images = X[sample_idx, :]  # (100,400)

    fig, ax_array = plt.subplots(nrows=10, ncols=10, sharex=True, sharey=True, figsize=(8, 8))

    for row in range(10):
        for column in range(10):
            ax_array[row, column].matshow(sample_images[10 * row + column].reshape((20, 20)))
    plt.xticks([])
    plt.yticks([])
    plt.show()


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def regularized_cost(theta, X, y, l):
    """
    don't penalize theta_0
    args:
        X: feature matrix, (m, n+1) # 插入了x0=1
        y: target vector, (m, )
        l: lambda constant for regularization
    """
    thetaReg = theta[1:]
    first = (-y * np.log(sigmoid(X @ theta))) + (y - 1) * np.log(1 - sigmoid(X @ theta))
    reg = (thetaReg @ thetaReg) * l / (2 * len(X))
    return np.mean(first) + reg


def regularized_gradient(theta, X, y, l):
    """
    don't penalize theta_0
    args:
        l: lambda constant
    return:
        a vector of gradient
    """
    thetaReg = theta[1:]
    first = (1 / len(X)) * X.T @ (sigmoid(X @ theta) - y)
    # 这里人为插入一维0，使得对theta_0不惩罚，方便计算
    reg = np.concatenate([np.array([0]), (l / len(X)) * thetaReg])
    return first + reg


def one_vs_all(X, y, l, K):
    """generalized logistic regression
    args:
        X: feature matrix, (m, n+1) # with incercept x0=1
        y: target vector, (m, )
        l: lambda constant for regularization
        K: numbel of labels
    return: trained parameters
    """
    all_theta = np.zeros((K, X.shape[1]))  # (10, 401)

    for i in range(1, K + 1):
        theta = np.zeros(X.shape[1])  # (401,)
        y_i = np.array([1 if label == i else 0 for label in y])

        ret = minimize(fun=regularized_cost, x0=theta, args=(X, y_i, l), method='TNC',
                       jac=regularized_gradient, options={'disp': True})
        all_theta[i - 1, :] = ret.x
    print(all_theta.shape)
    return all_theta


def predict_all(X, all_theta):
    # compute the class probability for each class on each training instance
    h = sigmoid(X @ all_theta.T)  # 注意的这里的all_theta需要转置
    # create array of the index with the maximum probability
    # Returns the indices of the maximum values along an axis.
    h_argmax = np.argmax(h, axis=1)
    # because our array was zero-indexed we need to add one for the true label prediction
    h_argmax = h_argmax + 1

    return h_argmax


def load_weight(path):
    data = loadmat(path)
    return data['Theta1'], data['Theta2']


def neural_networks():
    # Neural Networks
    theta1, theta2 = load_weight('ex3weights.mat')
    print(theta1.shape, theta2.shape)
    X, y = load_data('ex3data1.mat')
    y = y.flatten()
    X = np.insert(X, 0, 1, axis=1)  # intercept
    print(X.shape)
    a1 = X
    z2 = a1 @ theta1.T
    a2 = np.insert(sigmoid(z2), 0, 1, axis=1)
    z3 = a2 @ theta2.T
    a3 = sigmoid(z3)
    y_pred = np.argmax(a3, axis=1) + 1
    accuracy = np.mean(y_pred == y)
    print('accuracy = {0}%'.format(accuracy * 100))  # accuracy = 97.52%


def main():
    x, y = load_data('ex3data1.mat')
    # plot_an_image(x, y)
    # plot_100_image(x)
    x = np.insert(x, 0, 1, axis=1)  # (5000, 401)
    y = y.flatten()  # 这里消除了一个维度，方便后面的计算 or .reshape(-1) （5000，）
    all_theta = one_vs_all(x, y, 1, 10)  # 每一行是一个分类器的一组参数
    y_pred = predict_all(x, all_theta)
    accuracy = np.mean(y_pred == y)
    print('accuracy = {0}%'.format(accuracy * 100))


if __name__ == '__main__':
    # main()
    neural_networks()
