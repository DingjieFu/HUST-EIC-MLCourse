# 用户：夜卜小魔王

# 多变量线性回归 正规方程

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def compute_cost(X, Y, Theta):
    inner = np.power((x*Theta-Y), 2)
    return np.sum(inner/(2*len(X)))


path = "ex1data2.csv"
data = pd.read_csv(path, header=None, names=["Size", "Bedrooms", "Price"])
data = (data-data.mean())/data.std()
data.insert(0, "Ones", 1)

cols = data.shape[1]
x = data.iloc[:, :cols-1]
y = data.iloc[:, cols-1:]

x = np.matrix(x.values)
y = np.matrix(y.values)
theta = np.linalg.inv(x.T*x)*x.T*y

# print(theta)
# print(compute_cost(x, y, theta))

X, Y = x[:, 1], x[:, 2]
X, Y = np.meshgrid(X, Y)
temp = (x*theta).T
Z = (x*theta).T
for i in range(46):
    Z = np.vstack((Z, temp))
print(Z.shape)


fig = plt.figure(figsize=(8, 8))
ax = Axes3D(fig, auto_add_to_figure=False)  # 三维绘图
fig.add_axes(ax)
ax.scatter3D(data.Size, data.Bedrooms, data.Price, label="Training Data")
ax.plot_surface(X, Y, Z, cmap="rainbow")
ax.set_xlabel("Size")
ax.set_ylabel("Bedrooms")
ax.set_zlabel("Price")
ax.set_title("prediction vs. Training Data")
plt.show()
