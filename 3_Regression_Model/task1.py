import csv
import numpy as np
import scipy.optimize as opt


def dataLoader(trainPath, testPath):  # 加载数据集
    trainData = []
    testData = []
    with open(trainPath) as f:
        for line in f.readlines():
            # 先去除\n 再去除逗号 最后转为浮点数
            linelist = [float(parse) for parse in line.strip().split(",")]
            trainData.append(linelist)
    with open(testPath) as f:
        for line in f.readlines():
            linelist = [float(parse) for parse in line.strip().split(",")]
            testData.append(linelist)
    return trainData, testData


def regressionTrain(X, Y):
    # 初始化参数 先全定为0
    theta = np.zeros(len(X[0]))
    res = opt.minimize(fun=compute_cost, x0=theta, args=(
        X, Y), method='Newton-CG', jac=gradient_descent)  # 拟合参数
    # 最终的参数
    final_theta = res.x
    return final_theta


def sigmoid(z):  # sigmoid函数
    return 1 / (1 + np.exp(-z))


def compute_cost(Theta, X, Y):  # 代价函数
    # 全部转化为矩阵
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


def predict(X, Theta):  # 预测结果
    Theta, X = np.matrix(Theta), np.matrix(X)
    return (sigmoid(X*Theta.T) >= 0.5).astype(int)


def generateCsv(test_data, test_pre):  # 生成csv文件
    with open("result.csv", mode="w", newline="") as f:
        data = csv.writer(f)
        for i in range(len(test_data)):  # 逐行写入
            data.writerow(test_data[i]+test_pre[i])


if __name__ == '__main__':
    train_path = "PythonDeveloper/Machine_Learning/3_Regression_Model/task1/train.txt"
    test_path = "PythonDeveloper/Machine_Learning/3_Regression_Model/task1/test.txt"
    train_data, test_data = dataLoader(train_path, test_path)
    # 训练集输入参数
    train_x = [data[:-1] for data in train_data]
    # 训练集输出
    train_y = [data[-1] for data in train_data]
    theta = regressionTrain(np.array(train_x), np.array(train_y))
    test_pre = predict(np.array(test_data), theta)
    generateCsv(test_data, np.matrix.tolist(test_pre))
