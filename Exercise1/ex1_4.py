# 用户：夜卜小魔王

# 多变量梯度下降
# 由于数据数量级相差较大 进行归一化 缩短梯度下降的用时

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def compute_cost(X, Y, Theta):
    inner = np.power((X*Theta.T-Y), 2)
    return np.sum(inner/(2*len(X)))


def gradient_descent(X, Y, Theta, alpha=0.01, iterations=1000):  # 梯度下降
    temp = np.matrix(np.zeros(Theta.shape))  # temp -> (1, 3)
    parameters = int(Theta.ravel().shape[1])  # 参数个数 parameters -> 3
    cost_ = np.zeros(iterations)  # cost_ -> (1, 1000)
    for i in range(iterations):
        error = X*Theta.T-Y
        for j in range(parameters):
            term = np.multiply(error, X[:, j])
            temp[0, j] = Theta[0, j] - (alpha/len(X))*np.sum(term)
        Theta = temp  # 更新参数
        cost_[i] = compute_cost(X, Y, Theta)
    return Theta, cost_


path = "ex1data2.csv"
data = pd.read_csv(path, header=None, names=['Size', 'Bedrooms', 'Price'])
data = (data-data.mean())/data.std()  # 归一化 减去均值再除以方差
data.insert(0, 'Ones', 1)

cols = data.shape[1]
x = data.iloc[:, :cols-1]
y = data.iloc[:, cols-1:]

x = np.matrix(x.values)
y = np.matrix(y.values)

theta = np.matrix(np.array([0, 0, 0]))

g, cost = gradient_descent(x, y, theta)
print(g)
print(cost[-1])

_, ax = plt.subplots(figsize=(12, 8))
ax.plot(np.arange(1000), cost, 'r')
ax.set_xlabel("Iterations")
ax.set_ylabel("Cost")
ax.set_title("Error vs. Training Epoch")
plt.show()
