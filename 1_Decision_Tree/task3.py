
import csv
import graphviz
import numpy as np
import pandas as pd
from sklearn import tree


# 加载数据 切分出自己感兴趣的部分
def load_data(data_path, my_features):
    df = pd.read_csv(data_path)
    # Age列中的缺失值用Age中位数进行填充
    # 缺失值处理，采用中位数填充
    print(df.describe())
    df['Age'].fillna(int(df['Age'].mean()), inplace=True)
    # print(df.info())
    # 划分出自己感兴趣的特征
    _x = df[my_features]
    # 注意测试集没有这个属性 所以进行区分
    _y = None
    if "Survived" in list(df.columns):
        _y = df[['Survived']]
    return _x, _y


# 数据处理
def data_transform(data_path, my_features, is_add=False):
    _x, _y = load_data(data_path, my_features)
    # 输入的标签
    x_cols = list(_x.columns)
    if "Pclass" in x_cols:
        _x['Pclass'].replace({1: 0, 2: 0, 3: 1}, inplace=True)
    # 将Sex属性替换成离散值
    if "Sex" in x_cols:
        _x['Sex'].replace({"male": 1, "female": 0}, inplace=True)
    # 将年龄分段为离散值
    if "Age" in x_cols:
        # Age_dir = {"child": 0, "teenager": 1, "adult": 2, "elder": 3}
        for i, item in enumerate(_x['Age']):
            if item < 20:
                _x.loc[i, 'Age'] = 0
            elif item >= 20 and item < 29:
                _x.loc[i, 'Age'] = 1
            elif item >= 29 and item < 38:
                _x.loc[i, 'Age'] = 2
            else:
                _x.loc[i, 'Age'] = 3
    # 将dataframe格式转换为ndarray
    # 是否将 SibSp和Parch相加为一个属性
    if is_add:
        _x["Relatives"] = _x["SibSp"] + _x["Parch"]
        _x.drop(["SibSp", "Parch"], axis=1, inplace=True)
    if ~is_add:
        key_dir = {0: 0, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1}
        _x["SibSp"].replace(key_dir, inplace=True)
        _x["Parch"].replace(key_dir, inplace=True)
        _x["Relatives"] = _x["SibSp"] + _x["Parch"]
        _x.drop(["SibSp", "Parch"], axis=1, inplace=True)
    if _y is None:
        return _x.values, None
    print(_x)
    return _x.values, _y.values


def decisionTree(train_path, test_path, my_features):
    train_x, train_y = data_transform(train_path, my_features)
    test_x, _ = data_transform(test_path, my_features)
    print(train_x)
    # print(train_y.shape)
    clf = tree.DecisionTreeClassifier()
    clf.fit(train_x, train_y.flatten())
    pre_y = clf.predict(test_x)
    # dot_data = tree.export_graphviz(clf, out_file=None, feature_names=['Pclass', 'Sex', 'Age', 'Relatives'], class_names=[
    #                                 "0", "1"], filled=True, rounded=True)
    # graph = graphviz.Source(dot_data)
    # graph.render("survive")
    return pre_y


# 生成submission文件
def makeCsv(test_path, pre):
    df = pd.read_csv(test_path)
    pid = df['PassengerId']
    pre = pd.DataFrame(pre, columns=["Survived"])
    result = pd.concat([pid, pre], axis=1)
    result.to_csv("gender_submission_NEW.csv", sep=',', index=None)


if __name__ == '__main__':
    trainPath = "PythonDeveloper/Machine_Learning/1_Decision_Tree/Task3/train.csv"
    testPath = "PythonDeveloper/Machine_Learning/1_Decision_Tree/Task3/test.csv"
    features = ['Pclass', 'Name', 'Sex', 'Age', 'SibSp',
                'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']
    myFeatures = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch']
    # load_data(trainPath, myFeatures)
    # data_transform(trainPath, myFeatures)
    preY = decisionTree(trainPath, testPath, myFeatures)
    makeCsv(testPath, preY)
