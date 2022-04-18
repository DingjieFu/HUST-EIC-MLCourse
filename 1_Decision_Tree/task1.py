#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File:task1.py
@Author:夜卜小魔王
@Date:2022/03/18 18:42:35
'''


import csv
import numpy as np
import pandas as pd


# 结点对象
class Node(object):
    def __init__(self, col, name_list):
        self.col = col
        self.key = name_list[col]  # 决策目标直接通过col列数转换
        self.children = {}  # 定义为当前结点的子节点

    def __str__(self):
        return f"Node Decision={self.key}"  # 打印内容


# 叶子对象 对应只有眼镜类型三种可能 即 no lenses/soft/hard
class Leaf(object):
    def __init__(self, y, type_dir):
        self.key = val2Key(y, type_dir)

    def __str__(self):
        return f"Leaf Type={self.key}"


# 该函数实现数字到标签的映射
def val2Key(value, data_dir):
    for k, v in data_dir.items():
        if v == value:
            return k


# 打印树
def print_tree(node, pre='', sub=''):
    print(pre, node, sub)
    pre += "-" * 5
    if isinstance(node, Leaf):  # 当前node是树叶时直接return
        return
    for i in node.children:
        sub = 'lastValue=' + str(i)
        print_tree(node.children[i], pre, sub)


# 添加子节点 入参为数据集，父结点， 以及标签
def create_children(_x, parent_node, name_list, type_dir):
    # 遍历父结点col列所有的取值 父结点的col为其最大熵增列数
    # 看父结点col列数据有几种取值
    for var in np.unique(_x[:, parent_node.col]):
        # 按照父结点col列数据相同分出数据子集
        sub_x = _x[_x[:, parent_node.col] == var]
        # 看子集中的眼镜类型有几种取值
        var_list = np.unique(sub_x[:, -1])
        # 数值->标签
        myKey = val2Key(var, type_dir[parent_node.col])
        # 如果子集中type相同 则是叶子结点
        if len(var_list) == 1:
            # 定义叶子
            new_leaf = Leaf(var_list[0], type_dir[-1])
            # 将该叶子结点加入
            parent_node.children[myKey] = new_leaf
            continue  # 此子集已经判断为叶子 跳入下一个子集
        # 如果子集中type不相同 再找信息增益最大的列
        sub_gain_col = get_large_gain(sub_x)
        # 添加分支节点到父节点上
        child_node = Node(col=sub_gain_col, name_list=name_list)
        parent_node.children[myKey] = child_node
        # 对子结点再次加入儿子结点
        create_children(sub_x, child_node, name_list, type_dir)


# 将标签变为离散数字值 并且返回一些标签数值对应关系
def label_encoding(data_path):
    """
    输入为原始数据路径  输出为离散化的数据路径
    """
    csv_path = data_path.split(".")[0]+".csv"
    # 将原始文件转化为csv文件，并且用逗号分割
    # with open 会自动close
    with open(file=csv_path, mode="w", encoding="UTF-8") as f:
        f1 = open(data_path)
        datalines = f1.readlines()
        for dataline in datalines:
            dataline = dataline.replace("\t", ",")
            f.write(dataline)
        f1.close()
    # 标签->离散值 并且生成了映射字典
    # # 类名
    name_list = ["Age", "Sight", "Astigmatic", "Crying", "Type"]
    # 读取csv文件
    df = pd.read_csv(csv_path, header=None, names=name_list)
    dataDir_list = []  # 映射字典组成的列表
    item_list = []
    for i in name_list:  # 对每一列
        data_dir = {}  # 标签与离散值的映射字典
        now_rank = 0  # 从0开始排序
        for item in df[i]:  # 对每一列的每一个元素
            if item in data_dir:  # 如果字典中已经存在
                # 已经在字典里的直接取他的键值
                item = data_dir[item]
            else:  # 不在字典里要加入字典
                data_dir.update({item: now_rank})  # key-value
                item = now_rank
                now_rank += 1  # rank + 1
            item_list.append(item)
        dataDir_list.append(data_dir)  # 嵌套字典
    # 本来是按列加入 所以进行了转置
    item_list = np.array(item_list).reshape(
        (-1, len(df.iloc[:, 0]))).T
    with open(csv_path, mode="w", newline="") as f:
        data = csv.writer(f)
        for i in range(len(item_list)):  # 逐行写入
            data.writerow(item_list[i])
    return csv_path, name_list[:-1], dataDir_list


# 加载数据集 这里不再对数据集进行划分
def load_dataset(csv_path):
    """
    输入为文件路径   输出为nd.array
    """
    # 特别注意要加上header=None 否则会少一行
    df = pd.read_csv(csv_path, header=None)
    col = len(df.iloc[0])  # 列数 5
    row = len(df.iloc[:, 0])  # 行数 24
    _x = np.empty((row, col), dtype=int)
    for i in range(row):
        _x[i] = df.iloc[i]
    return _x  # _x.shape = (24,5)


# 计算熵值
def calInforEntropy(_x):
    """
    对应输入为原始数据时 对应于初始熵值,即没有划分时的熵值
    """
    # 原始数据最后一列 对应佩戴眼镜类型
    _y = _x[:, -1]
    entropy = 0  # 初始化熵值为0
    # 统计每个数值出现的次数，以list形式返回
    count_list = np.bincount(_y)
    for count in count_list:
        if count == 0:
            continue
        else:
            _p = count / len(_y)
            # 概率 p = 出现次数/总数

        entropy -= _p*np.log2(_p)  # 熵计算公式
    # print(_y)
    # print(count_list)
    # print(len(_y))
    # print(entropy)
    return entropy


# 计算信息增益，即ID3决策树
def calInforGain(_x):
    col_gain_list = []  # 每一列的信息增益的列表
    for i in range(len(_x[0][:-1])):  # 对每一列而言
        col_entropy = 0  # 某一列的熵值 初始化
        # 创建一个不重复的集合
        for item in set(_x[:, i]):
            # 将某一列中值相同的分开 得到子集
            x_itemSame = _x[_x[:, i] == item]
            # 计算该子集出现的概率
            p_child = len(x_itemSame)/len(_x)
            # 计算子集的熵值
            entrop_child = calInforEntropy(x_itemSame)
            # 子集的熵乘以子集出现概率，所有相加，得到某一列的熵值
            col_entropy += p_child*entrop_child
        # 某一列的信息增益 熵值降低 说明信息变多
        inforGain = calInforEntropy(_x) - col_entropy
        col_gain_list.append(inforGain)
        # print(col_entropy)
        # print(inforGain)
        # break
    # print(col_gain_list)
    return col_gain_list


# 找到信息增益最大的列
def get_large_gain(_x):
    gain_list = calInforGain(_x)
    largest_gain = 0  # 最大信息增益值
    largest_gain_col = 0  # 最大信息增益的列
    for i in range(len(gain_list)):
        if gain_list[i] > largest_gain:
            largest_gain = gain_list[i]
            largest_gain_col = i
    # print(largest_gain_col)
    return largest_gain_col


if __name__ == '__main__':
    path = "PythonDeveloper/Machine_Learning/1_Decision_Tree/Task1/lenses.txt"
    csvpath, nameList, dataDir = label_encoding(path)
    # print(nameList)  # 四项待决策内容 一项最终类型
    # print(dataDir)  # 所有nameList中的小型映射字典
    x = load_dataset(csvpath)
    # calInforEntropy(x)
    # print(calInforGain(x))
    # get_large_gain(x)
    # 根结点 传入最大熵增列数 label
    root = Node(get_large_gain(x), nameList)
    create_children(x, root, nameList, dataDir)
    print_tree(root)
