import numpy as np
import time
from sklearn.cluster import KMeans
from sklearn import datasets

# K-Means算法
def K_Means(centers,data):  # centers：初始质心（14 X 27）；data：全部用户数据（6040 X 27）
    # 设置簇的个数
    cluster = 14
    # 调用KMeans
    kmeans = KMeans(init=centers, n_clusters=cluster, n_init=1)  # init:质心初始方式；n_clusters:聚类个数；n_init：选取多少次质心（因为质心是随机选取的，所以需要选取多次质心，以获得最佳质心的结果）
    # 计算簇质心并给每个样本预测类别
    kmeans.fit(data)
    # 聚类结果
    label = kmeans.labels_
    # 返回结果集
    results = []
    # 遍历每个聚类，划分出每个聚类中的数据
    for i in range(cluster):
        # 返回对应聚类的索引
        result_index = np.where(label == i)
        # 获取索引对应的数据
        result_data = data[result_index]
        # 把结果放入resluts中
        results.append(result_data)
    # kmeans.cluster_centers_:每个簇的质心
    return np.array(results), kmeans.cluster_centers_

# K-Means算法
def K_Means2(centers,data):  # centers：初始质心（14 X 27）；data：全部用户数据（6040 X 27）
    # 设置簇的个数
    cluster = 14
    # 调用KMeans
    kmeans = KMeans(init=centers, n_clusters=cluster, n_init=1)  # init:质心初始方式；n_clusters:聚类个数；n_init：选取多少次质心（因为质心是随机选取的，所以需要选取多次质心，以获得最佳质心的结果）
    # 计算簇质心并给每个样本预测类别
    kmeans.fit(data[:,0:27])
    # 聚类结果
    label = kmeans.labels_

    # 返回结果集
    results = []
    # 遍历每个聚类，划分出每个聚类中的数据
    for i in range(cluster):
        # 返回对应聚类的索引
        result_index = np.where(label == i)
        # 获取索引对应的数据
        result_data = data[result_index]
        # 把结果放入resluts中
        results.append(result_data)
    # kmeans.cluster_centers_:每个簇的质心
    return np.array(results), kmeans.cluster_centers_