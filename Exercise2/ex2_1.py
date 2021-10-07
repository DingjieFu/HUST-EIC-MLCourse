# 用户：夜卜小魔王
# 逻辑回归
# 通过两次考试判断能否通过

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.optimize as opt
from sklearn.metrics import classification_report  # 评价报告 查看性能


def plot_scatter(dataset, option=False):  # 画出数据的散点图
    if option:
        sns.set(context='notebook', style='darkgrid', palette=sns.color_palette("RdBu", 2))
        sns.lmplot(x='exam1', y='exam2', hue='admitted', data=dataset, height=6,
                   fit_reg=False,  # 是否拟合曲线
                   scatter_kws={'s': 50}
                   )
        plt.show()
    else:
        return


def get_feature(dataset):  # 获取特征
    """
    dataset.insert(0, "Ones", 1)
    return dataset.iloc[:, :-1], dataset.iloc[:, -1:]
    """
    ones = pd.DataFrame({"Ones": np.ones(len(dataset))})  # 构建m行1列的元素
    dataset = pd.concat([ones, dataset], axis=1)  # 合并数据
    return dataset.iloc[:, :-1].values, dataset.iloc[:, -1:].values  # 返回ndarray


def sigmoid(z):  # sigmoid函数
    return 1 / (1 + np.exp(-z))


def plot_sigmoid():  # 画出sigmoid函数图像
    _, ax = plt.subplots(figsize=(10, 8))
    nums = np.arange(-10, 10, step=0.01)
    ax.plot(nums, sigmoid(nums), 'r')
    ax.set_ylim((-0.1, 1.1))
    ax.set_xlabel("z", fontsize=18)
    ax.set_ylabel("g(z)", fontsize=18)
    ax.set_title("Sigmoid Function", fontsize=18)
    plt.show()


def compute_cost(Theta, X, Y):  # 代价函数
    Theta, X, Y = np.matrix(Theta), np.matrix(X), np.matrix(Y)
    A = np.multiply(-Y, np.log(sigmoid(X * Theta.T)))
    B = np.multiply(1 - Y, np.log(1 - sigmoid(X * Theta.T)))
    return np.sum(A - B) / len(X)


def gradient_descent(Theta, X, Y):  # 梯度下降
    Theta, X, Y = np.matrix(Theta), np.matrix(X), np.matrix(Y)
    parameters = int(Theta.ravel().shape[1])
    grad = np.zeros(parameters)
    error = sigmoid(X * Theta.T) - Y
    for i in range(parameters):
        term = np.multiply(error, X[:, i])
        grad[i] = np.sum(term) / len(X)
    return grad


def predict(X, Theta):  # 验证
    Theta, X = np.matrix(Theta), np.matrix(X)
    return (sigmoid(X*Theta.T) >= 0.5).astype(int)


def plot_boundary(X, Y, dataset):  # 画决策边界
    sns.set(context='notebook', style='ticks', font_scale=1.5)
    sns.lmplot(x='exam1', y='exam2', hue='admitted', data=data, height=6,
               fit_reg=False,  # 是否拟合曲线
               scatter_kws={'s': 50}  # 散点大小
               )
    plt.plot(x, y, 'g')
    plt.xlim(0, 130)
    plt.ylim(0, 130)
    plt.title("Decision Boundary")
    plt.show()


path = "ex2data1.csv"
data = pd.read_csv(path, header=None, names=['exam1', 'exam2', 'admitted'])

# plot_scatter(data, True)

x, y = get_feature(data)  # ndarray格式
theta = np.zeros(3)
# print(x.shape, y.shape, theta.shape)
# plot_sigmoid()

# print(compute_cost(theta, x, y))
# print(gradient_descent(theta, x, y))

res = opt.minimize(fun=compute_cost, x0=theta, args=(x, y), method='Newton-CG', jac=gradient_descent)  # 拟合参数
# print(res)
final_theta = res.x
y_pre = predict(x, final_theta)
print(classification_report(y_pre, y))

coe = -(res.x/res.x[2])
x = np.arange(130, step=0.1)
y = coe[0]+coe[1]*x
plot_boundary(x, y, data)
