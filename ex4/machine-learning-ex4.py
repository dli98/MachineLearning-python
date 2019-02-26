import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import scipy.optimize as opt
from sklearn.metrics import classification_report


def displayData(X):
    """
    Display Image
    """
    sample_idx = np.random.choice(np.arange(X.shape[0]), 100)  # 随机选100个样本
    sample_images = X[sample_idx, :]  # (100,400)

    fig, ax_array = plt.subplots(nrows=10, ncols=10, sharex=True, sharey=True, figsize=(8, 8))

    for row in range(10):
        for column in range(10):
            ax_array[row, column].matshow(sample_images[10 * row + column].reshape((20, 20)), cmap='gray')
    plt.xticks([])
    plt.yticks([])
    plt.axis('off')
    plt.set_cmap('gray_r')
    plt.show()


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def expand_y(y):
    result = []
    # 把y中每个类别转化为一个向量，对应的lable值在向量对应位置上置为1
    for i in y:
        y_array = np.zeros(10)
        y_array[i - 1] = 1
        result.append(y_array)
    return np.array(result)


def feed_forward(ThetaVec, X):
    '''得到每层的输入和输出'''
    theta1 = ThetaVec[:25 * 401].reshape(25, 401)
    theta2 = ThetaVec[25 * 401:].reshape(10, 26)
    # 前面已经插入过偏置单元，这里就不用插入了
    a1 = X
    z2 = a1 @ theta1.T
    a2 = np.insert(sigmoid(z2), 0, 1, axis=1)
    z3 = a2 @ theta2.T
    a3 = sigmoid(z3)

    return a1, z2, a2, z3, a3


def cost(ThetaVec, X, y):
    a1, z2, a2, z3, h = feed_forward(ThetaVec, X)
    J = 0
    for i in range(len(X)):
        first = - y[i] * np.log(h[i])
        second = (1 - y[i]) * np.log(1 - h[i])
        J = J + np.sum(first - second)
    J = J / len(X)
    '''
        # or just use verctorization
        J = -y * np.log(h) - (1 -y) * np.log(1-h)
        return J.sum() / len(X)
    '''
    return J


def regularized_cost(theta, X, y, Lambda=1.0):
    theta1 = theta[:25 * 401].reshape(25, 401)
    theta2 = theta[25 * 401:].reshape(10, 26)
    reg = np.sum(theta1[:, 1:] ** 2) + np.sum(theta2[:, 1:] ** 2)  # or use np.power(a, 2)
    return Lambda / (2 * len(X)) * reg + cost(theta, X, y)


def sigmoid_gradient(z):
    return sigmoid(z) * (1 - sigmoid(z))


def random_init_weights(size):
    return np.random.uniform(-0.12, 0.12, size)


def gradient(ThetaVec, X, y):
    '''
    unregularized gradient, notice no d1 since the input layer has no error
    return 所有参数theta的梯度，故梯度D(i)和参数theta(i)同shape，重要。
    '''
    theta2 = ThetaVec[25 * 401:].reshape(10, 26)
    a1, z2, a2, z3, h = feed_forward(ThetaVec, X)
    d3 = h - y  # (5000, 10)
    d2 = d3 @ theta2[:, 1:] * sigmoid_gradient(z2)  # (5000, 25)
    D2 = d3.T @ a2  # (10, 26)
    D1 = d2.T @ a1  # (25, 401)
    D = np.hstack((D1.ravel(), D2.ravel()))
    DVec = (1 / len(X)) * D  # (10285,)

    return DVec


def regularized_gradient(ThetaVec, X, y, l=1):
    """不惩罚偏置单元的参数"""
    a1, z2, a2, z3, h = feed_forward(ThetaVec, X)
    DVec = gradient(ThetaVec, X, y)
    D1 = DVec[0:25 * 401].reshape(25, 401)
    D2 = DVec[25 * 401:].reshape(10, 26)
    theta1 = ThetaVec[:25 * 401].reshape(25, 401)
    theta2 = ThetaVec[25 * 401:].reshape(10, 26)
    theta1[:, 0] = 0
    theta2[:, 0] = 0
    reg_D1 = D1 + (l / len(X)) * theta1
    reg_D2 = D2 + (l / len(X)) * theta2

    return np.hstack((reg_D1.ravel(), reg_D2.ravel()))


def gradient_checking(ThetaVec, X, y, e):
    def a_numeric_grad(plus, minus):
        """
        对每个参数theta_i计算数值梯度，即理论梯度。
        """
        return (regularized_cost(plus, X, y) - regularized_cost(minus, X, y)) / (e * 2)

    numeric_grad = []
    for i in range(len(ThetaVec)):
        plus = ThetaVec.copy()  # deep copy otherwise you will change the raw theta
        minus = ThetaVec.copy()
        plus[i] = plus[i] + e
        minus[i] = minus[i] - e
        grad_i = a_numeric_grad(plus, minus)
        numeric_grad.append(grad_i)

    numeric_grad = np.array(numeric_grad)
    analytic_grad = regularized_gradient(ThetaVec, X, y)
    diff = np.linalg.norm(numeric_grad - analytic_grad) / np.linalg.norm(numeric_grad + analytic_grad)

    print(
        'If your backpropagation implementation is correct,\nthe relative difference will be smaller than 10e-9 (assume epsilon=0.0001).\nRelative Difference: {}\n'.format(
            diff))


def nn_training(initial_thetaVec, X, y, Lambda=1.0):
    print(initial_thetaVec.shape)
    res = opt.minimize(fun=regularized_cost,
                       x0=initial_thetaVec,
                       args=(X, y, Lambda),
                       method='TNC',
                       jac=regularized_gradient,
                       options={'maxiter': 400}
                       )
    return res


def accuracy(theta, X, y):
    _, _, _, _, h = feed_forward(theta, X)
    y_pred = np.argmax(h, axis=1) + 1
    print(classification_report(y, y_pred))


def plot_hidden(theta):
    t1 = theta[:25 * 401].reshape(25, 401)
    t1 = t1[:, 1:]
    fig, ax_array = plt.subplots(5, 5, sharex=True, sharey=True, figsize=(6, 6))
    for r in range(5):
        for c in range(5):
            ax_array[r, c].matshow(t1[r * 5 + c].reshape(20, 20), cmap='gray_r')
            plt.xticks([])
            plt.yticks([])
    plt.show()


## =========== Part 1: Loading and Visualizing Data =============
print('Loading and Visualizing Data ...')
data = loadmat('ex4data1.mat')
X = data['X']  # 5000 x 400
y = data['y']  # 5000 x 1
# displayData(X)
# input("Program paused. Press Enter to continue...")

## ================ Part 2: Loading Parameters ================
print('Loading Saved Neural Network Parameters ...')
data = loadmat('ex4weights.mat')
Theta1 = data['Theta1']  # 25 x 401
Theta2 = data['Theta2']  # 10 x 26

# Unroll parameters
ThetaVec = np.hstack((Theta1.ravel(), Theta2.ravel()))
## ================ Part 3: Compute Cost (Feedforward) ================

y = np.squeeze(y)
y = expand_y(y)
X = np.insert(X, 0, 1, axis=1)
J = cost(ThetaVec, X, y)
print(f'Cost at parameters (loaded from ex4weights): {J} (this value should be about 0.287629)\n')

# input("Program paused. Press Enter to continue...")

## =============== Part 4: Implement Regularization ===============
J = regularized_cost(ThetaVec, X, y)
print(f'Cost at parameters (loaded from ex4weights): {J} (this value should be about 0.383770)')

## ================ Part 5: Sigmoid Gradient  ================
print('Evaluating sigmoid gradient...')

g = sigmoid_gradient(np.array([1, -0.5, 0, 0.5, 1]))
print('Sigmoid gradient evaluated at [1 -0.5 0 0.5 1]: ')
print(g)

## ================ Part 6: Initializing Pameters ================
print('Initializing Neural Network Parameters ...')
initial_theta1 = random_init_weights((25, 401))
initial_theta2 = random_init_weights((10, 26))

initial_thetaVec = np.hstack((initial_theta1.ravel(), initial_theta2.ravel()))

## =============== Part 7: Gradient checking ===============
# gradient_checking(initial_thetaVec, X, y, e=0.0001)  #这个运行很慢，谨慎运行

## =============== Part 8: Regularized Neural Networks ===============

res = nn_training(initial_thetaVec, X, y, Lambda=1.0)
print(res)

## ================= Part 9: Implement Predict =================
accuracy(res.x)

## ================= Part 10: Visualizing the hidden layer  =================

plot_hidden()
