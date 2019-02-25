import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D


def plot_data(data):
    # print(data.head())
    # print(data.describe())
    # print(data)
    data.plot(kind='scatter', x='Population', y='Profit', figsize=(8, 5))
    plt.xlabel('Population of City in 10,000s')
    plt.ylabel('Profit in $10,000s')
    plt.title('Scatter plot of training data')
    plt.show()


def computeCost(x, y, theta):
    inner = np.power(((x.dot(theta.T)) - y), 2)
    return np.mean(inner) / 2


def gradientDescent(X, y, theta, alpha, epoch):
    cost = np.zeros(epoch)  # 初始化一个ndarray，包含每次epoch的cost
    m = X.shape[0]  # 样本数量m
    for i in range(epoch):
        # 利用向量化一步求解
        temp = theta - (alpha / m) * ((X.dot(theta.T) - y).T).dot(X)  # 一个 θ 临时矩阵(1, 2)
        theta = temp
        cost[i] = computeCost(X, y, theta)
    return theta, cost


def plot_cost(theta0, theta1, cost):
    figure = plt.figure()
    axes = Axes3D(figure)
    axes.plot3D(theta0, theta1, cost)
    plt.show()


def linear_regression_one_variable():
    path = 'ex1data1.txt'
    data = pd.read_csv(path, header=None, names=['Population', 'Profit'])
    # plot_data(data)
    data.insert(0, 'Ones', 1)
    cols = data.shape[1]  # columns
    x = data.iloc[:, : -1].values
    y = data.iloc[:, cols - 1: cols].values
    theta = np.array([[0, 0]])
    # cost = computeCost(x, y, theta)

    alpha = 0.01  # learing rate
    epoch = 1000  # 迭代次数
    final_theta, cost = gradientDescent(x, y, theta, alpha, epoch)
    minCost = computeCost(x, y, final_theta)
    print(minCost)
    # plot_cost(cost[0], cost[1], cost[2])

    x = np.linspace(data.Population.min(), data.Population.max(), 100)  # 横坐标
    h = final_theta[0, 0] + (final_theta[0, 1] * x)  # 纵坐标，利润

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(x, h, 'r', label='Prediction')
    ax.scatter(data['Population'], data.Profit, label='Traning Data')
    ax.legend(loc=2)  # 2表示在左上角
    ax.set_xlabel('Population')
    ax.set_ylabel('Profit')
    ax.set_title('Predicted Profit vs. Population Size')
    plt.show()


def main():
    linear_regression_one_variable()
    # linear_regression_multiple_variables()


if __name__ == '__main__':
    main()
