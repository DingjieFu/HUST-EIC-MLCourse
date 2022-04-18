import numpy as np
from struct import unpack
import matplotlib.pyplot as plt
from PIL import Image
from collections import Counter
import math

# 配置文件
config = {
    # 训练集文件
    'train_images_idx3_ubyte_file_path': 'data/train-images.idx3-ubyte',
    # 训练集标签文件
    'train_labels_idx1_ubyte_file_path': 'data/train-labels.idx1-ubyte',

    # 测试集文件
    'test_images_idx3_ubyte_file_path': 'data/t10k-images.idx3-ubyte',
    # 测试集标签文件
    'test_labels_idx1_ubyte_file_path': 'data/t10k-labels.idx1-ubyte',

    # 特征提取阙值
    'binarization_limit_value': 0.14,

    # 特征提取后的边长
    'side_length': 14
}


def decode_idx3_ubyte(path):
    '''
    解析idx3-ubyte文件，即解析MNIST图像文件
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
    解析idx1-ubyte文件，即解析MNIST标签文件
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
