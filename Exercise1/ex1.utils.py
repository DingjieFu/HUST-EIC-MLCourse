# 用户：夜卜小魔王
# Exercise1 所需要的工具(函数)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class Plotfigure(object):  # 绘图类
    def __init__(self, data):
        self.data = data  # 数据部分
        self.x = None
        self.y = None

    def plot_dataset(self):  # 对数据集进行绘制 散点图
        self.data.plot(kind="scatter", x="Population", y="Profit", figsize=(12, 8))  # 数据散点图
        plt.show()

    def plot_line(self):  # 绘制出拟合的曲线
        _, ax = plt.subplots(figsize=(12, 8))
        ax.plot(self.x, self.y, 'r', label="Prediction")
        ax.scatter(self.data.Population, self.data.Profit, label="Training Data")
        ax.legend(loc=4)
        ax.set_xlabel("Population"), ax.set_ylabel("Profit"), ax.set_title("Predict Profit vs. Population")
        plt.show()

    def plot_cost(self):  # 绘制代价函数
        _, ax = plt.subplots(figsize=(12, 8))
        ax.plot(self.x, self.y, 'r')
        ax.set_xlabel("Iterations"), ax.set_ylabel("Error"), ax.set_title("Error vs. Iterations")
        plt.show()


class Dataprocessing(object):  # 得到数据并对数据进行预处理
    def __init__(self, path):
        self.path = path  # csv文件路径
        self.data = None  # data文件 默认为None
        self.data_names = []  # data文件的类名称 默认为空列表
        self.x = None  # 数据的输入
        self.y = None  # 数据的输出 对ex1而言只有一个输出
        self.theta = None  # 待拟合参数 其个数与输入变量个数相同 -> 注意要先在insert以后再求
        self.x_matrix = None
        self.y_matrix = None

    def load_data(self):  # 从path加载数据
        self.data = pd.read_csv(self.path, header=None, names=self.data_names)

    def insert_ones(self):  # 将theta(0)也并入变量中 即它的项恒为1 故插入一列1
        self.data.insert(0, "Ones", 1)

    def get_feature(self):  # 从数据中分离出输入和输出 ex1中为单输出 故取最后一列为y
        self.x = self.data.iloc[:, :-1]
        self.y = self.data.iloc[:, -1:]

    def get_theta(self):  # 定义初始theta变量数
        self.theta = np.matrix(np.zeros(self.x.shape[1]))  # 初始化全为0

    def transform_matrix(self):  # 将数据变为矩阵
        self.x_matrix = np.matrix(self.x)
        self.y_matrix = np.matrix(self.y)


class Linearregression(object):  # 线性回归
    def __init__(self, X, Y, Theta):
        self.x = X  # 输入  为矩阵类型
        self.y = Y  # 输出
        self.theta = Theta  # 待拟合参数
        self.cost = None  # 损失值
        self.alpha = 0.01  # learning rate
        self.iteration = 1000  # 学习轮次数

    def compute_cost(self):  # 线性回归的代价函数
        inner = np.power((self.x * self.theta.T - self.y), 2)
        return np.sum(inner) / (2 * len(self.x))

    def gradient_descent(self):  # 梯度下降
        temp = np.matrix(np.zeros(self.theta.shape))
        self.cost = np.matrix(np.zeros(self.iteration))  # 记录每一轮梯度下降后的cost
        for i in range(self.iteration):
            error = self.x * self.theta.T - self.y
            for j in range(self.x.shape[1]):
                term = np.multiply(error, self.x[:, j])
                temp[0, j] = self.theta[0, j] - (self.alpha / len(self.x)) * np.sum(term)
            self.theta = temp
            self.cost[0, i] = self.compute_cost()


if __name__ == '__main__':
    ex1 = Dataprocessing("ex1data1.csv")
    ex1.data_names = ["Population", "Profit"]
    ex1.load_data()
    ex1.insert_ones()
    ex1.get_feature()
    ex1.get_theta()
    print(ex1.data.head())
    print(ex1.x.shape, ex1.y.shape)
    print(ex1.theta.shape)
    ex1.transform_matrix()

    linear = Linearregression(ex1.x_matrix, ex1.y_matrix, ex1.theta)
    linear.compute_cost()
    linear.gradient_descent()
    print(linear.theta)
    # print(linear.cost)

    plot = Plotfigure(ex1.data)

    # plot.x = np.linspace(ex1.data.Population.min(), ex1.data.Population.max(), 100)  # 抽取100个数据
    # plot.y = linear.theta[0, 0]+linear.theta[0, 1]*plot.x
    #
    # plot.plot_line()

    # plot.x = np.arange(1000)
    # plot.y = linear.cost.T
    # plot.plot_cost()
