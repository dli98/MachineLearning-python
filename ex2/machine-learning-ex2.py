import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt


def visualizing_data(data):
    print(data.head())
    pos = data[data.admitted.isin(['1'])]
    neg = data[data.admitted.isin(['0'])]
    plt.scatter(pos['exam1'], pos['exam2'], marker='+', c='k', label='Admitted')
    plt.scatter(neg['exam1'], neg['exam2'], marker='o', c='y', label='Not Admitted')
    # 设置横纵坐标名
    plt.xlabel('Exam 1 Score')
    plt.ylabel('Exam 2 Score')
    plt.legend(loc=1, prop={'size': 9})
    plt.show()


def visualizing_data2(data):
    print(data.head())
    pos = data[data.Accepted.isin(['1'])]
    neg = data[data.Accepted.isin(['0'])]
    plt.scatter(pos['Test 1'], pos['Test 2'], marker='+', c='k', label='y=1')
    plt.scatter(neg['Test 1'], neg['Test 2'], marker='o', c='y', label='y=0')
    # 设置横纵坐标名
    plt.xlabel('Microchip Test 1')
    plt.ylabel('Microchip Test 2')
    plt.legend(loc=1, prop={'size': 9})
    plt.show()


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def cost(theta, x, y):
    first = (-y) * np.log(sigmoid(x.dot(theta)))
    second = (1 - y) * np.log(1 - sigmoid(x.dot(theta)))
    return np.mean(first - second)


def gradient(theta, x, y):
    return (x.T @ (sigmoid(x @ theta) - y)) / len(x)


def predict(theta, x):
    probability = sigmoid(x @ theta)
    return [1 if x >= 0.5 else 0 for x in probability]


def evaluating_logistic(final_theta, x, y, data):
    predictions = predict(final_theta, x)
    correct = [1 if a == b else 0 for (a, b) in zip(predictions, y)]
    accuracy = sum(correct) / len(x)
    print(accuracy)
    x1 = np.arange(100, step=0.1)
    x2 = -(final_theta[0] + x1 * final_theta[1]) / final_theta[2]
    pos = data[data.admitted.isin(['1'])]
    neg = data[data.admitted.isin(['0'])]
    plt.scatter(pos['exam1'], pos['exam2'], marker='+', c='k', label='Admitted')
    plt.scatter(neg['exam1'], neg['exam2'], marker='o', c='y', label='Not Admitted')
    plt.plot(x1, x2)
    plt.axis([30, 100, 30, 100])
    plt.xlabel('Exam 1 Score')
    plt.ylabel('Exam 2 Score')
    plt.legend(loc=1, prop={'size': 9})
    plt.title('Decision Boundary')
    plt.show()


def feature_mapping(x1, x2, power):
    data = {}
    for i in np.arange(power + 1):
        for p in np.arange(i + 1):
            data["f{}{}".format(i - p, p)] = np.power(x1, i - p) * np.power(x2, p)
    return pd.DataFrame(data)


def costReg(theta, X, y, lam=1):
    # 不惩罚第一项
    _theta = theta[1:]
    reg = (lam / (2 * len(X))) * (_theta @ _theta)
    return cost(theta, X, y) + reg


def gradientReg(theta, X, y, l=1):
    reg = (1 / len(X)) * theta
    reg[0] = 0
    return gradient(theta, X, y) + reg


def main():
    # path = 'ex2data1.txt'
    # data = pd.read_csv(path, names=['exam1', 'exam2', 'admitted'])
    # # visualizing_data(data)
    # data.insert(0, 'Ones', 1)
    # x = data.iloc[:, :-1].values
    # y = data.iloc[:, -1].values
    # theta = np.zeros(x.shape[1])
    # print(cost(theta, x, y))
    # print(gradient(theta, x, y))
    # result = opt.fmin_tnc(func=cost, x0=theta, fprime=gradient, args=(x, y))
    # final_theta = result[0]
    # evaluating_logistic(final_theta, x, y, data)
    data2 = pd.read_csv('ex2data2.txt', names=['Test 1', 'Test 2', 'Accepted'])
    # visualizing_data2(data2)
    x1 = data2['Test 1'].values
    x2 = data2['Test 2'].values
    _data2 = feature_mapping(x1, x2, power=6)
    x = _data2.values
    y = data2['Accepted'].values
    theta = np.zeros(x.shape[1])
    print(costReg(theta, x, y, lam=1))
    result2 = opt.fmin_tnc(func=costReg, x0=theta, fprime=gradientReg, args=(x, y, 2))
    final_theta = result2[0]
    predictions = predict(final_theta, x)
    correct = [1 if a == b else 0 for (a, b) in zip(predictions, y)]
    accuracy = sum(correct) / len(correct)
    print(accuracy)


if __name__ == '__main__':
    main()
