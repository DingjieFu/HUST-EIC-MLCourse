# 用户：夜卜小魔王
# 正则化 梯度下降
# 正则化的好处: 较好地避免了过拟合

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def compute_cost(X, Y, Theta):
    inner = np.power((X*Theta-Y), 2)
    return np.sum(inner)/(2*len(X))


def gradient_descent(X, Y, Theta, alpha=0.01, iterations=1000, gama=100):  # 使用正则化 一般不对theta(0)进行正则化
    temp = np.matrix(np.zeros(Theta.shape))
    parameters = int(Theta.ravel().shape[1])
    cost_ = np.zeros(iterations)
    for i in range(iterations):
        error = X*Theta-Y

        for j in range(parameters):
            if j == 0:  # 对theta(0)不正则化
                term = np.multiply(error, X[:, j])
                temp[j, 0] = Theta[j, 0] - (alpha / len(X)) * np.sum(term)
            else:
                term = np.multiply(error, X[:, j])
                temp[j, 0] = (1-alpha*gama/len(X))*Theta[j, 0] - (alpha / len(X)) * np.sum(term)
        Theta = temp  # 更新参数
        cost_[i] = compute_cost(X, Y, Theta)
    return Theta, cost_


def plot_prediction(X, Y, dataset):  # 画出预测函数
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(X, Y, 'r', label="Prediction")
    ax.scatter(dataset.Population, dataset.Profit, label="Training Data")
    ax.legend(loc=4)  # 显示标签位置
    ax.set_xlabel('Population')
    ax.set_ylabel('Profit')
    ax.set_title("predicted Profit vs. Population Size")
    plt.show()


path = "ex1data1.csv"
data = pd.read_csv(path, header=None, names=["Population", "Profit"])
data.insert(0, "Ones", 1)

cols = data.shape[1]
x = data.iloc[:, :cols-1]
y = data.iloc[:, cols-1:]
x, y = np.matrix(x), np.matrix(y)
theta = np.matrix(np.zeros((2, 1)))


print(compute_cost(x, y, theta))

final_theta, cost = gradient_descent(x, y, theta)
print(final_theta)

x = np.linspace(data.Population.min(), data.Population.max(), 100)  # 抽取100个数据
f = final_theta[0, 0] + (final_theta[1, 0]*x)

plot_prediction(x, f, data)

