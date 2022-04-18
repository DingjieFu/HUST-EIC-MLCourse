
import graphviz
import numpy as np
import pandas as pd
from sklearn import tree


def load_dataset(train_path, test_path):
    # 选择ISO-8859-1编码方式
    train_df = pd.read_csv(train_path, encoding="ISO-8859-1")
    test_df = pd.read_csv(test_path, encoding="ISO-8859-1")
    # df.fillna() 缺失值补特殊值-100
    train_data = train_df.fillna(-100)
    test_data = test_df.fillna(-100)
    # f_i=[〖BSSID〗_1: 1,〖BSSID〗_2: 0,…]
    train_df = train_data[['finLabel', 'BSSIDLabel', 'RoomLabel']]
    test_df = test_data[['finLabel', 'BSSIDLabel', 'RoomLabel']]
    return train_df, test_df


# 数据处理 采用BSSID集合的并集作为特征
def data_transform(data_df, bssid_labels):
    # 通过set去重，得到所有的BSSIDLabel种类的key集合
    # bssid_labels = set(data_df['BSSIDLabel'])
    bssid_len = len(bssid_labels)  # 274
    # df.group() 将数据按照finLabel相同分组
    data_groupy = data_df.groupby('finLabel')
    # print(data_bssid.describe())
    _f = []  # 指纹列表
    f_roomids = []  # 指纹房间号
    # f_i为finLabel的值 f_value为对应f_i相同的一组dataframe
    for f_i, f_data in data_groupy:
        # 在f_i相同时 将它的所有BSSIDLabel取值放入一个ndarray
        f_i_bssid_labels = np.array(f_data['BSSIDLabel'])
        # 建立一个以BSSIDLabel类别总数为大小的全零矩阵 (274,1)
        bssid_count = np.zeros((bssid_len, 1), dtype=int)
        # 对BSSIDLabel全部类别都有的列表查找
        for bssid_label in bssid_labels:
            # 如果以finLabel划分后的BSSIDLabel中有此BSSIDLabel
            if bssid_label in f_i_bssid_labels:
                # 相应位置置1 即构造f_i=[〖BSSID〗_1:1,〖BSSID〗_2:0,…]
                bssid_count[list(bssid_labels).index(bssid_label)] = 1
        # 将所有指纹组合到一起 (x,274,1)
        _f.append(bssid_count)
        # 房间号整合为一个ndarray 同f_i下roomid相同
        roomid = np.array(f_data['RoomLabel'])
        # 对每一个bssid_count 记录其所在的房间号 (x,1)
        f_roomids.append(roomid[0])
    # print(data_classes)
    # print(data_classes.count(1))
    return np.array(_f), np.array(f_roomids)


def decisionTree(traindf, testdf):
    # 通过set去重，得到所有的BSSIDLabel种类的key集合
    bssidLabels = set(traindf['BSSIDLabel'])
    trainData, trainData_classes = data_transform(traindf, bssidLabels)
    testData, testData_classes = data_transform(testdf, bssidLabels)
    # print(trainData)
    # print(trainData_classes.shape)
    # print(testDatacls)
    # print(testData_classes.shape)
    clf = tree.DecisionTreeClassifier()
    # 注意这里需要使测试集的f_i数量与训练集一样
    clf.fit(trainData.reshape((len(trainData), -1)), trainData_classes)
    pre_y = clf.predict(testData.reshape((len(testData), -1)))
    print("real:", testData_classes)
    print("pre:", pre_y)
    sub_y = pre_y - testData_classes
    acc = list(sub_y).count(0)/len(sub_y)
    print(f"准确率={acc}")

    # dot_data = tree.export_graphviz(clf, out_file=None, feature_names=[list(bssidLabels)],
    #                                 class_names=["1", "2", "3", "4"], filled=True, rounded=True)
    # graph = graphviz.Source(dot_data)
    # graph.render("BSSID")


if __name__ == '__main__':
    trainPath = 'PythonDeveloper/Machine_Learning/1_Decision_Tree/Task2/TrainDT.csv'
    testPath = 'PythonDeveloper/Machine_Learning/1_Decision_Tree/Task2/TestDT.csv'
    trainDf, testDf = load_dataset(trainPath, testPath)
    decisionTree(trainDf, testDf)
