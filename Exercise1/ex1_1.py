# 用户：夜卜小魔王
# 时间：2021/10/4 19:22

# 单变量回归 采用梯度下降方式

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def show_info(dataset):
    print(dataset.head())  # 预览数据
    print(dataset.describe())  # 对数据的详细描述
    dataset.plot(kind="scatter", x="Population", y="Profit", figsize=(12, 8))  # 数据散点图
    plt.show()


def compute_cost(X, Y, Theta):  # 计算代价函数
    inner = np.power((X*Theta.T-Y), 2)
    return np.sum(inner)/(2*len(X))


def gradient_descent(X, Y, Theta, alpha=0.001, iterations=1000):  # 梯度下降
    temp = np.matrix(np.zeros(Theta.shape))  # temp -> (1, 2)
    parameters = int(Theta.ravel().shape[1])  # 参数个数 parameters -> 2
    cost_ = np.zeros(iterations)  # cost_ -> (1, 1000)
    for i in range(iterations):
        error = X*Theta.T-Y
        for j in range(parameters):
            term = np.multiply(error, X[:, j])
            temp[0, j] = Theta[0, j] - (alpha/len(X))*np.sum(term)
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


def plot_cost(iterations, cost_):  # 画出代价函数
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(np.arange(iterations), cost_, "r")
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Cost')
    ax.set_title("Error vs. Training Epoch")
    plt.show()


path = "ex1data1.csv"
data = pd.read_csv(path, header=None, names=["Population", "Profit"])
data.insert(0, "Ones", 1)  # 将bias并入自变量,在数据中插入一列,名称是Once,数值全是1

# 得到输入x 输出y
cols = data.shape[1]
x = data.iloc[:, :cols-1]
y = data.iloc[:, cols-1:]

# 将数据转化成矩阵
x = np.matrix(x.values)  # x -> (97, 2)
y = np.matrix(y.values)  # y -> (97, 1)
theta = np.matrix(np.array([0, 0]))  # theta -> (1, 2)

g, cost = gradient_descent(x, y, theta)

x = np.linspace(data.Population.min(), data.Population.max(), 100)  # 抽取100个数据
f = g[0, 0] + (g[0, 1]*x)

plot_prediction(x, f, data)
