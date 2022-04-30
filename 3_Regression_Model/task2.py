import numpy as np
import scipy.optimize as opt


def bagOfWords2VecMN(inputSet, vocabList=10000):
    """
    获得文档向量，向量中的数值代表词汇表中的某个单词在一篇文档中的出现次数
    并且我们根据题目已给出的条件 假设共有10000个单词(实际上只有9999个)
    """
    # index 为0-9999
    # 这里给每位加上1 修正一下
    returnVec = [0]*vocabList
    # 对输入数据的每一项 构建出一个矢量
    for word in inputSet:
        # word 为出现位置 需要减一
        returnVec[word-1] += 1
    return returnVec


def dataLoader(dataPath, sub):
    """
    加载数据 并且将原始数据转为词袋矢量
    """
    data = []
    with open(dataPath+sub) as f:
        lines = f.readlines()
        for line in lines:
            # 去除空白 用''分割
            line = line.strip().split(' ')
            # str转为int
            wordVec = [int(word) for word in line]
            # 构建词袋矢量
            data.append(bagOfWords2VecMN(wordVec))
    return data


# 数据准备
def dataPreparation(trainPath, testPath):
    trainData = dataLoader(trainPath, "/train_data.txt")
    trainLabel = []
    testData = dataLoader(testPath, "/test_data.txt")
    with open(trainPath+"/train_labels.txt") as f:
        lines = f.readlines()
        for line in lines:
            # 去除空白 用''分割
            line = line.strip().split(' ')
            # str转为int
            trainLabel.append([int(word) for word in line][0])
    return trainData, trainLabel, testData


def regressionTrain(X, Y):
    # 初始化参数 先全定为0
    theta = np.zeros(len(X[0]))
    res = opt.minimize(fun=compute_cost, x0=theta, args=(
        X, Y), method='TNC', jac=gradient_descent)  # 拟合参数
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


def generateTxt(test_data, Theta):  # 生成csv文件
    with open("test_labels.txt", mode="w") as f:
        for item in test_data:
            pre_y = predict(np.array(item), Theta)
            f.write(str(int(pre_y))+"\n")


if __name__ == '__main__':
    train_path = "PythonDeveloper/Machine_Learning/3_Regression_Model/task2/train"
    test_path = "PythonDeveloper/Machine_Learning/3_Regression_Model/task2/test"
    traindata, trainlabel, testdata = dataPreparation(train_path, test_path)
    # print(len(traindata[0]))
    theta = regressionTrain(
        np.array(traindata), np.array(trainlabel))
    # print(len(theta))
    generateTxt(np.array(testdata), theta)
