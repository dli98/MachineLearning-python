import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt


def plotData(data):
    pos = data[data.admitted.isin(['1'])]
    neg = data[data.admitted.isin(['0'])]
    plt.scatter(pos['exam1'], pos['exam2'], marker='+', c='k', label='Admitted')
    plt.scatter(neg['exam1'], neg['exam2'], marker='o', c='y', label='Not Admitted')
    # 设置横纵坐标名
    plt.xlabel('Exam 1 Score')
    plt.ylabel('Exam 2 Score')
    plt.legend(loc=1, prop={'size': 9})
    plt.show()


def plotDecisionBoundary(theta, data):
    x1 = np.arange(100, step=0.1)
    x2 = -(theta[0] + x1 * theta[1]) / theta[2]
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


def visualizing_data2(data):
    print(data.head())

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


# ==================== Part 1: Plotting ====================
print('Plotting data with + indicating (y = 1) examples and o indicating (y = 0) examples.')
path = 'ex2data1.txt'
data = pd.read_csv(path, names=['exam1', 'exam2', 'admitted'])
plotData(data)

data.insert(0, 'Ones', 1)
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values
input("Program paused. Press Enter to continue...")

# # ============ Part 2: Compute Cost and Gradient ============
initial_theta = np.zeros(X.shape[1])
J = cost(initial_theta, X, y)
print('Cost at initial theta (zeros): %f' % J)
grad = gradient(initial_theta, X, y)
print('Gradient at initial theta (zeros): ' + str(grad))

input("Program paused. Press Enter to continue...")

# ============= Part 3: Optimizing using scipy  =============
result = opt.fmin_tnc(func=cost, x0=initial_theta, fprime=gradient, args=(X, y))
theta = result[0]
# Plot Boundary
plotDecisionBoundary(theta, data)
plt.show()
input("Program paused. Press Enter to continue...")

#  ============== Part 4: Predict and Accuracies ==============
prob = sigmoid(np.array([1, 45, 85]).dot(theta))
print('For a student with scores 45 and 85, we predict an admission probability of %f' % prob)

# Compute accuracy on our training set
p = predict(theta, X)
correct = [1 if a == b else 0 for (a, b) in zip(p, y)]
accuracy = sum(correct) / len(correct)
print('Train Accuracy: %f' % accuracy)

input("Program paused. Press Enter to continue...")

# =========== Part 5: Regularized Logistic Regression ============

data2 = pd.read_csv('ex2data2.txt', names=['Test 1', 'Test 2', 'Accepted'])

x1 = data2['Test 1'].values
x2 = data2['Test 2'].values
_data2 = feature_mapping(x1, x2, power=6)
X = _data2.values
y = data2['Accepted'].values
initial_theta = np.zeros(X.shape[1])

J = costReg(initial_theta, X, y, lam=1)
print('Cost at initial theta (zeros): %f' % J)
result2 = opt.fmin_tnc(func=costReg, x0=initial_theta, fprime=gradientReg, args=(X, y, 2))
theta = result2[0]
predictions = predict(theta, X)
correct = [1 if a == b else 0 for (a, b) in zip(predictions, y)]
accuracy = sum(correct) / len(correct)
print(accuracy)

input("Program paused. Press Enter to continue...")


# ============= Part 3: Optional Exercises =============

# Plot Boundary
def plotBoundary(theta, data2, Lambda):
    plt.title(r'$\lambda$ = ' + str(Lambda))
    x = np.linspace(-1, 1.5, 250)
    xx, yy = np.meshgrid(x, x)
    z = feature_mapping(xx.ravel(), yy.ravel(), power=6).values
    z = z @ theta
    z = z.reshape(xx.shape)
    pos = data2[data2.Accepted.isin(['1'])]
    neg = data2[data2.Accepted.isin(['0'])]
    plt.scatter(pos['Test 1'], pos['Test 2'], marker='+', c='k', label='y=1')
    plt.scatter(neg['Test 1'], neg['Test 2'], marker='o', c='y', label='y=0')
    plt.contour(xx, yy, z, 0)
    plt.ylim(-.8, 1.2)
    plt.show()


for Lambda in np.arange(0.0, 10.1, 1.0):
    result2 = opt.fmin_tnc(func=costReg, x0=initial_theta, fprime=gradientReg, args=(X, y, Lambda))
    theta = result2[0]
    print('lambda = ' + str(Lambda))
    print('theta:', ["%0.4f" % i for i in theta])
    plotBoundary(theta, data2, Lambda)
