import numpy as np
import scipy.optimize as opt


# 加载数据
def dataLoader(trainPath, testPath):
    trainData = []
    testData = []
    with open(trainPath) as f:
        for line in f.readlines():
            # 先去除\n 再去除\t 最后转为浮点数
            linelist = [float(parse) for parse in line.strip().split("\t")]
            trainData.append(linelist)
    with open(testPath) as f:
        for line in f.readlines():
            linelist = [float(parse) for parse in line.strip().split("\t")]
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


def calcAccuracy(pre_y, true_y):
    count = 0
    for i in range(len(pre_y)):
        if pre_y[i] == true_y[i]:
            count += 1
    return float(count)/len(pre_y)


if __name__ == '__main__':
    trainpath = "PythonDeveloper/Machine_Learning/4_Neural_Network/task1/horseColicTraining.txt"
    testpath = "PythonDeveloper/Machine_Learning/4_Neural_Network/task1/horseColicTest.txt"
    train_data, test_data = dataLoader(trainpath, testpath)
    # 训练集输入参数
    train_x = [data[:-1] for data in train_data]
    # 训练集输出
    train_y = [data[-1] for data in train_data]
    # 测试集输入
    test_x = [data[:-1] for data in test_data]
    # 训练集输出
    test_y = [data[-1] for data in test_data]
    # 预测数据
    theta = regressionTrain(np.array(train_x), np.array(train_y))
    test_pre = predict(np.array(test_x), theta)
    print("这次的准确度为：", calcAccuracy(test_pre, test_y))
