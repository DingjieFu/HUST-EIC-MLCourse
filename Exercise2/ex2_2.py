# 用户：夜卜小魔王

# 正则化

import numpy as np
import pandas as pd
import seaborn as sns
import scipy.optimize as opt
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report


def plot_scatter(dataset):  # 画出数据的散点图
    positive = dataset[dataset['Accepted'] == 1]  # 找到Accepted列为1的那些数据
    negative = dataset[dataset['Accepted'] == 0]
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.scatter(positive['Test1'], positive['Test2'], s=50, c='b', marker='o', label='Accepted')
    ax.scatter(negative['Test1'], negative['Test2'], s=50, c='r', marker='x', label='Rejected')
    ax.legend()
    ax.set_xlabel('Test1 Score')
    ax.set_ylabel('Test2 Score')
    plt.show()


def feature_mapping(X, Y, power, as_ndarray=False):  # 特征映射 -> 将低维映射到高伟
    data_ = {"f{}{}".format(i - p, p): np.power(X, i - p) * np.power(Y, p)
             for i in np.arange(power + 1)
             for p in np.arange(i + 1)
             }
    if as_ndarray:
        return pd.DataFrame(data_).values
    else:
        return pd.DataFrame(data_)


def sigmoid(z):  # sigmoid函数
    return 1 / (1 + np.exp(-z))


def compute_cost(Theta, X, Y):  # 代价函数
    Theta, X, Y = np.matrix(Theta), np.matrix(X), np.matrix(Y)
    A = np.multiply(-Y, np.log(sigmoid(X * Theta.T)))
    B = np.multiply(1 - Y, np.log(1 - sigmoid(X * Theta.T)))
    return np.sum(A - B) / len(X)


def regularized_cost(Theta, X, Y, L=1):  # 正则化损失函数
    theta_1_to_n = Theta[1:]
    regularized_term = (L / (2 * len(X))) * np.power(theta_1_to_n, 2).sum()
    return compute_cost(Theta, X, Y) + regularized_term


def gradient_descent(Theta, X, Y):  # 梯度下降
    Theta, X, Y = np.matrix(Theta), np.matrix(X), np.matrix(Y)
    parameters = int(Theta.ravel().shape[1])
    grad = np.zeros(parameters)
    error = sigmoid(X * Theta.T) - Y
    for i in range(parameters):
        term = np.multiply(error, X[:, i])
        grad[i] = np.sum(term) / len(X)
    return grad


def regularized_gradient(Theta, X, Y, L=1):  # 正则化梯度
    theta_1_to_n = Theta[1:]
    regularized_theta = (L / len(X)) * theta_1_to_n
    regularized_term = np.concatenate([np.array([0]), regularized_theta])
    return gradient_descent(Theta, X, Y) + regularized_term


def predict(X, Theta):  # 验证
    Theta, X = np.matrix(Theta), np.matrix(X)
    return (sigmoid(X * Theta.T) >= 0.5).astype(int)


def draw_boundary(power, L):  # 可视化
    density = 1000
    threshold = 2 * 10 ** -3
    final_theta = feature_mapped_logistic_regression(power, L)
    X, Y = find_decision_boundary(density, power, final_theta, threshold)
    df = pd.read_csv('ex2data2.csv', names=['test1', 'test2', 'accepted'])
    sns.lmplot(x='test1', y='test2', hue='accepted', data=df, height=6, fit_reg=False, scatter_kws={"s": 100})
    plt.scatter(X, Y, c='r', s=10)
    plt.title('Decision boundary')
    plt.show()


def feature_mapped_logistic_regression(power, L):
    df = pd.read_csv('ex2data2.csv', names=['test1', 'test2', 'accepted'])
    X1 = np.array(df.test1)
    X2 = np.array(df.test2)
    Y = df.iloc[:, -1:]

    X = feature_mapping(X1, X2, power, as_ndarray=True)
    Theta = np.zeros(X.shape[1])

    Res = opt.minimize(fun=regularized_cost,
                       x0=Theta,
                       args=(X, Y, L),
                       method='TNC',
                       jac=regularized_gradient)
    final_theta = Res.x
    return final_theta


def find_decision_boundary(density, power, Theta, threshold):
    t1 = np.linspace(-1, 1.5, density)  # 1000个样本
    t2 = np.linspace(-1, 1.5, density)
    cordinates = [(X, Y) for X in t1 for Y in t2]
    x_cord, y_cord = zip(*cordinates)
    mapped_cord = feature_mapping(x_cord, y_cord, power)  # this is a dataframe
    inner_product = mapped_cord.values @ Theta
    decision = mapped_cord[np.abs(inner_product) < threshold]
    return decision.f10, decision.f01


path = "ex2data2.csv"
data = pd.read_csv(path, header=None, names=['Test1', 'Test2', 'Accepted'])

# print(data.describe())
# print(data.head())
# plot_scatter(data)

x1 = np.array(data.Test1)
x2 = np.array(data.Test2)
data = feature_mapping(x1, x2, power=6)
# print(data.shape)
# print(data.head())

theta = np.zeros(data.shape[1])
x = feature_mapping(x1, x2, power=6, as_ndarray=True)
y = data.iloc[:, -1:].values
# print(x.shape)
# print(y.shape)
# print(theta.shape)
print(regularized_cost(theta, x, y))
print(regularized_gradient(theta, x, y))

res = opt.minimize(fun=regularized_cost, x0=theta, args=(x, y), method='Newton-CG', jac=regularized_gradient)
print(res)

draw_boundary(power=6, L=1)
# draw_boundary(power=6, L=0)
# draw_boundary(power=6, L=100)
