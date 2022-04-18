
import re
import numpy as np
from random import shuffle
from itertools import chain


def textParse(bigString):
    """
    接受一个大字符串并将其解析为字符串列表。该函数去掉少于两个字符的字符串，并将所有字符串转换为小写。
    """
    listOfTokens = re.split(r'\W', bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]


def creatVocabList(dataset):
    """
    创建一个包含在所有文档中出现的不重复的词的列表。
    """
    # 合并为一维list
    vocabList = list(chain.from_iterable(dataset))
    # set去重
    vocabSet = set(vocabList)
    return list(vocabSet)


def bagOfWords2VecMN(vocabList, inputSet):
    """
    获得文档向量，向量中的数值代表词汇表中的某个单词在一篇文档中的出现次数
    """
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec


def loadData(hPath, sPath):
    """
    数据加载 得到不重复的单词列表 整合[单词,标签]
    规定ham邮件为1 spam邮件为0
    """
    docList = []
    word_label = []
    for i in range(1, 26):
        with open(hPath+'/{}.txt'.format(str(i)), encoding="ISO-8859-1") as f:
            content = textParse(f.read())
            docList.append(content)
            word_label.append([content, 1])
    for i in range(1, 26):
        with open(sPath+'/{}.txt'.format(str(i)), encoding="ISO-8859-1") as f:
            content = textParse(f.read())
            docList.append(content)
            word_label.append([content, 0])
    vocabList = creatVocabList(docList)
    for i in range(0, len(word_label)):
        wordVec = bagOfWords2VecMN(vocabList, word_label[i][0])
        if i < 25:
            word_label[i] = [wordVec, 1]
        else:
            word_label[i] = [wordVec, 0]
    return vocabList, word_label


def BayesTrain(trainData):
    numTrainDocs = len(trainData)
    numWords = len(trainData[0][0])
    classLabel = []
    for item in trainData:
        classLabel.append(item[1])
    # 设ham标签为1 所以这里实际上是求p(ham)
    pham = sum(classLabel) / float(numTrainDocs)
    # 避免有为0的项影响
    p0Num = np.ones(numWords)
    p1Num = np.ones(numWords)
    p0All = 2.0
    p1All = 2.0
    for i in range(numTrainDocs):
        if classLabel[i] == 1:
            p1Num += trainData[i][0]
            p1All += sum(trainData[i][0])
        else:
            p0Num += trainData[i][0]
            p0All += sum(trainData[i][0])
    p0 = np.log(p0Num / p0All)
    p1 = np.log(p1Num / p1All)
    return p0, p1, pham


def BayesTest(wordLabelList):
    """
    测试函数 包含划分数据集 采用交叉验证法
    """
    numEpoch = 10
    for i in range(numEpoch):
        count = 0
        for i in range(10):
            # 每次开始前都打乱数据集
            shuffle(wordLabelList)
            # 训练集 每49个一组
            trainData = wordLabelList[0:len(wordLabelList)-1]
            # 测试集 单独一组
            testData = wordLabelList[-1]
            p1, p0, pClass = BayesTrain(trainData)
            # 真实值
            _y = testData[1]
            # 预测值
            pre_y = BayesClassify(testData[0], p1, p0, pClass)
            # print(_y, pre_y)
            if pre_y == _y:
                count += 1
        print("平均预测正确率为", count/10)


# 朴素贝叶斯分类函数
def BayesClassify(vocabVec, p1, p0, pClass):
    """
    vocabVec为待分类的单词向量
    p1、p0分别为标签1、标签0的邮件中单词出现的概率矢量
    pClass 为标签1邮件出现概率
    """
    p1 = sum(vocabVec*p1)+np.log(pClass)
    p0 = sum(vocabVec*p0)+np.log(1-pClass)
    if p1 > p0:
        return 1
    else:
        return 0


if __name__ == '__main__':
    ham_path = 'PythonDeveloper/Machine_Learning/2_Bayes_Model/task1/ham'
    spam_path = 'PythonDeveloper/Machine_Learning/2_Bayes_Model/task1/spam'
    doclist, wordLabel = loadData(ham_path, spam_path)
    BayesTrain(wordLabel)
    # BayesTest(wordLabel)
