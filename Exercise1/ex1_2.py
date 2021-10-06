# 用户：夜卜小魔王

# 单变量回归 采用正规方程
# 由于矩阵求逆的复杂度为0(n^3) 当数量少时用正规方程快于梯度下降 且只适合线性模型

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def compute_cost(X, Y, Theta):
    inner = np.power((X*Theta-Y), 2)
    return np.sum(inner/(2*len(X)))


def plot_prediction(X, Y, dataset):  # 画出预测函数
    _, ax = plt.subplots(figsize=(12, 8))
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

# 将csv文件中的数据分为输入与输出
# iloc() -> 根据索引行号选取   loc() -> 根据索引名称选取
x = data.loc[:, ["Ones", "Population"]]  # 输入
y = data.loc[:, ['Profit']]  # 输出

# 转化为矩阵格式
x = np.matrix(x)
y = np.matrix(y)
theta = np.linalg.inv(x.T * x) * x.T * y   # 通过正规方程直接计算出最好的theta
print(theta)
print(compute_cost(x, y, theta))
plot_prediction(x, x*theta, data)  # 虽然对于数据部分拟合得更好 但是产生了错误值
