
import numpy as np


def bagOfWords2VecMN(inputSet, vocabList=10000):
    """
    获得文档向量，向量中的数值代表词汇表中的某个单词在一篇文档中的出现次数
    并且我们根据题目已给出的条件 假设共有10000个单词(实际上只有9999个)
    """
    # index 为0-9999
    # 这里给每位加上1 修正一下
    returnVec = np.ones(vocabList)
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


def BayesTrain(trainData, classLabel):
    # 训练样本数
    trainLen = len(trainData)
    numWords = len(trainData[0])
    # 这里是求标签中1出现的概率 即正面评论的概率
    pClass = sum(classLabel) / float(trainLen)
    # 避免有为0的项影响 拉普拉斯修正 分子的修正已经在前面体现
    p0Num = np.zeros(numWords)
    p1Num = np.zeros(numWords)
    p0All = 2.0
    p1All = 2.0
    for i in range(trainLen):
        if classLabel[i] == 1:
            p1Num += trainData[i]
            p1All += sum(trainData[i])
        else:
            p0Num += trainData[i]
            p0All += sum(trainData[i])
    p1 = np.log(p1Num / p1All)
    p0 = np.log(p0Num / p0All)
    return p1, p0, pClass


def BayesTest(testData, p1, p0, pClass):
    with open("test_labels.txt", mode="w") as f:
        for item in testData:
            pre_y = BayesClassify(item, p1, p0, pClass)
            f.write(str(pre_y)+"\n")


# 朴素贝叶斯分类函数
def BayesClassify(testVec, p1, p0, pClass):
    p1 = sum(testVec*p1)+np.log(pClass)
    p0 = sum(testVec*p0)+np.log(1-pClass)
    if p1 > p0:
        return 1
    else:
        return 0


if __name__ == '__main__':
    train_path = "PythonDeveloper/Machine_Learning/2_Bayes_Model/task3/train"
    test_path = "PythonDeveloper/Machine_Learning/2_Bayes_Model/task3/test"
    traindata, trainlabel, testdata = dataPreparation(train_path, test_path)
    p1, p0, pclass = BayesTrain(traindata, trainlabel)
    BayesTest(testdata, p1, p0, pclass)
