import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.manifold import MDS
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean


def load_dataset(path1, path2):
    # 选择ISO-8859-1编码方式
    df1 = pd.read_csv(path1, encoding="ISO-8859-1")
    df2 = pd.read_csv(path2, encoding="ISO-8859-1")
    data_df1 = df1[['finLabel', 'BSSIDLabel', 'RSSLabel']]
    data_df2 = df2[['finLabel', 'BSSIDLabel', 'RSSLabel']]
    return data_df1, data_df2


# 数据处理 采用BSSID集合的并集作为特征
def data_transform(data_df):
    # 通过set去重，得到所有的BSSIDLabel种类的key集合
    bssid_labels = set(data_df['BSSIDLabel'])
    # df1有266种, df2有271种
    bssid_len = len(bssid_labels)
    # df.group() 将数据按照finLabel相同分组
    data_groupy = data_df.groupby('finLabel')
    _f = []  # 指纹列表
    # f_i为finLabel的值 f_data为对应f_i相同的一组dataframe
    for f_i, f_data in data_groupy:
        # 在f_i相同时 将它的所有BSSIDLabel取值放入一个ndarray
        f_i_bssid_labels = np.array(f_data['BSSIDLabel'])
        # 建立一个以BSSIDLabel类别总数为大小的全零矩阵
        bssid_count = np.zeros((bssid_len, 1), dtype=int)
        # 对BSSIDLabel全部类别都有的列表查找
        for bssid_label in bssid_labels:
            # 如果以finLabel划分后的BSSIDLabel中有此BSSIDLabel
            if bssid_label in f_i_bssid_labels:
                # 相应位置置为RSSLabel值
                rss_label = f_data[f_data.BSSIDLabel ==
                                   bssid_label]['RSSLabel']
                bssid_count[list(bssid_labels).index(bssid_label)] = rss_label
        # 将所有指纹组合到一起 (x,len,1)
        _f.append(bssid_count)
    return np.array(_f)


def k_means(_X):
    # 归一化
    xmax, xmin = np.max(_X), np.min(_X)
    _X = (_X - xmin) / (xmax - xmin)
    kmeans = KMeans(n_clusters=5)
    kmeans.fit(_X)
    # 预测
    y_kmeans = kmeans.predict(_X)
    # DB指数
    print("DBI:", DaviesBouldin(_X, y_kmeans))
    # 降维可视化
    plot_MDS(_X, y_kmeans)


def DaviesBouldin(X, labels):
    n_cluster = len(np.bincount(labels))
    cluster_k = [X[labels == k] for k in range(n_cluster)]
    centroids = [np.mean(k, axis=0) for k in cluster_k]
    S = [np.mean([euclidean(p, centroids[i]) for p in k])
         for i, k in enumerate(cluster_k)]
    Ri = []
    for i in range(n_cluster):
        Rij = []
        # 计算Rij
        for j in range(n_cluster):
            if j != i:
                r = (S[i] + S[j]) / euclidean(centroids[i], centroids[j])
                Rij.append(r)
         # 求Ri
        Ri.append(max(Rij))
    # 求dbi
    dbi = np.mean(Ri)
    return dbi


def plot_MDS(*data):
    X, y = data
    mds = MDS(n_components=2)
    X_r = mds.fit_transform(X)  # 原始数据集转换到二维
    # 绘制二维图形
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    colors = ((1, 0, 0), (0, 1, 0), (0, 0, 1), (0.5, 0.5, 0), (0, 0.5, 0.5), (0.5, 0, 0.5),
              (0.4, 0.6, 0), (0.6, 0.4, 0), (0, 0.6, 0.4), (0.5, 0.3, 0.2),)  # 颜色集合，不同标记的样本染不同的颜色
    for label, color in zip(np.unique(y), colors):
        position = y == label
        ax.scatter(X_r[position, 0], X_r[position, 1],
                   label="target= %d" % label, color=color)
    ax.set_xlabel("X[0]")
    ax.set_ylabel("X[1]")
    ax.legend(loc="best")
    ax.set_title("MDS")
    plt.show()


if __name__ == '__main__':
    path1 = 'PythonDeveloper/Machine_Learning/6_Clustering/task2/DataSetKMeans1.csv'
    path2 = 'PythonDeveloper/Machine_Learning/6_Clustering/task2/DataSetKMeans2.csv'
    dataDF1, dataDF2 = load_dataset(path1, path2)
    f_ndarray = data_transform(dataDF2)
    k_means(f_ndarray[:, :, 0])
