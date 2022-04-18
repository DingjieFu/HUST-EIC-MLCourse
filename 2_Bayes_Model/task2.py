import numpy as np
from struct import unpack
import matplotlib.pyplot as plt
from PIL import Image
from collections import Counter
import math

# 配置文件
config = {
    # 训练集文件
    'train_images_idx3_ubyte_file_path':
    'PythonDeveloper/Machine_Learning/2_Bayes_Model/task2/MNIST/train-images.idx3-ubyte',
    # 训练集标签文件
    'train_labels_idx1_ubyte_file_path':
    'PythonDeveloper/Machine_Learning/2_Bayes_Model/task2/MNIST/train-labels.idx1-ubyte',
    # 测试集文件
    'test_images_idx3_ubyte_file_path':
    'PythonDeveloper/Machine_Learning/2_Bayes_Model/task2/MNIST/t10k-images.idx3-ubyte',
    # 测试集标签文件
    'test_labels_idx1_ubyte_file_path':
    'PythonDeveloper/Machine_Learning/2_Bayes_Model/task2/MNIST/t10k-labels.idx1-ubyte',
    # 特征提取阙值
    'binarization_limit_value': 0.14,
    # 特征提取后的边长
    'side_length': 28
}


def decode_idx3_ubyte(path):
    '''
    解析idx3-ubyte文件,即解析MNIST图像文件
    '''
    '''
    也可不解压，直接打开.gz文件。path是.gz文件的路径
    import gzip
    with gzip.open(path, 'rb') as f:
    '''
    print('loading %s' % path)
    with open(path, 'rb') as f:
        # 前16位为附加数据，每4位为一个整数，分别为幻数，图片数量，每张图片像素行数，列数。
        magic, num, rows, cols = unpack('>4I', f.read(16))
        print('magic:%d num:%d rows:%d cols:%d' % (magic, num, rows, cols))
        mnistImage = np.fromfile(f, dtype=np.uint8).reshape(num, rows, cols)
    print('done')
    return mnistImage


def decode_idx1_ubyte(path):
    '''
    解析idx1-ubyte文件,即解析MNIST标签文件
    '''
    print('loading %s' % path)
    with open(path, 'rb') as f:
        # 前8位为附加数据，每4位为一个整数，分别为幻数，标签数量。
        magic, num = unpack('>2I', f.read(8))
        print('magic:%d num:%d' % (magic, num))
        mnistLabel = np.fromfile(f, dtype=np.uint8)
    print('done')
    return mnistLabel


def normalizeImage(image):
    '''
    将图像的像素值正规化为0.0 ~ 1.0
    '''
    res = image.astype(np.float32) / 255.0
    return res


def load_train_images(path=config['train_images_idx3_ubyte_file_path']):
    return normalizeImage(decode_idx3_ubyte(path))


def load_train_labels(path=config['train_labels_idx1_ubyte_file_path']):
    return decode_idx1_ubyte(path)


def load_test_images(path=config['test_images_idx3_ubyte_file_path']):
    return normalizeImage(decode_idx3_ubyte(path))


def load_test_labels(path=config['test_labels_idx1_ubyte_file_path']):
    return decode_idx1_ubyte(path)


def oneImagesFeatureExtraction(image):
    '''
    对单张图片进行特征提取
    '''
    res = np.empty((config['side_length'], config['side_length']))
    num = 28//config['side_length']
    for i in range(0, config['side_length']):
        for j in range(0, config['side_length']):
            # tempMean = (image[2*i:2*(i+1),2*j:2*(j+1)] != 0).sum()/(2 * 2)
            tempMean = image[num*i:num*(i+1), num*j:num*(j+1)].mean()
            if tempMean > config['binarization_limit_value']:
                res[i, j] = 1
            else:
                res[i, j] = 0
    return res


def featureExtraction(images):
    res = np.empty((images.shape[0], config['side_length'],
                    config['side_length']), dtype=np.float32)
    for i in range(images.shape[0]):
        res[i] = oneImagesFeatureExtraction(images[i])
    return res


def dataLoader():
    train_images = load_train_images()
    train_labels = load_train_labels()
    test_images = load_test_images()
    test_labels = load_test_labels()
    train_images = featureExtraction(train_images)
    test_images = featureExtraction(test_images)
    return train_images, train_labels, test_images, test_labels


def BayesTrain(train_images, train_labels):
    # 图片总数
    imageNum = train_images.shape[0]
    # 计数每个标签出现的次数 并整合成字典
    labelNum = Counter(train_labels)
    pLabel = []
    for i in range(10):
        # 每个标签出现概率
        pLabel.append(labelNum[i]/imageNum)
    # 数据的维度 即(num, 14, 14) num为图片数
    datashape = train_images.shape
    # 将(14, 14)转为一维14x14个特征
    train_images.resize((datashape[0], datashape[1]*datashape[2]))
    # 对每一个标签计算概率
    pNum = np.empty((10, train_images.shape[1]))
    pVec = np.empty((10, train_images.shape[1]))
    for i in range(10):
        pNum[i] = train_images[np.where(train_labels == i)].sum(axis=0)
        pVec[i] = (pNum[i] + 1) / (labelNum[i] + 2)
    return pLabel, pVec


def bayesClassifier(test_x, pLabel, pVec):
    '''
    使用贝叶斯分类器进行分类(极大似然估计)
    '''
    oldShape = test_x.shape
    test_x.resize(oldShape[0]*oldShape[1])
    classP = np.empty(10)
    for j in range(10):
        temp = sum([math.log(1-pVec[j][x]) if test_x[x] ==
                    0 else math.log(pVec[j][x]) for x in range(test_x.shape[0])])
        classP[j] = np.array(math.log(pLabel[j]) + temp)
        classP[j] = np.array(temp)
    test_x.resize(oldShape)
    return np.argmax(classP)


def modelEvaluation(test_x, test_y, prioriP, posteriorP):
    '''
    对贝叶斯分类器的模型进行评估
    '''
    bayesClassifierRes = np.empty(test_x.shape[0])
    for i in range(test_x.shape[0]):
        bayesClassifierRes[i] = bayesClassifier(test_x[i], prioriP, posteriorP)
    return bayesClassifierRes, (bayesClassifierRes == test_y).sum() / test_y.shape[0]


if __name__ == '__main__':
    trainx = load_train_images()
    # trainx, trainy, testx, testy = dataLoader()
    # plabel, pvec = BayesTrain(trainx, trainy)
    # preVec, acc = modelEvaluation(testx, testy, plabel, pvec)
    # print("正确率为：", acc)
    x = oneImagesFeatureExtraction(trainx[0])
    plt.subplot(211), plt.imshow(x, cmap="gray")
    plt.subplot(212), plt.imshow(trainx[0], cmap="gray")
    plt.show()
