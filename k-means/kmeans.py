# -*- coding: utf-8 -*-
"""
# @Time    : 2018/12/18 上午 10:45
# @Author  : tanliqing
# @Email   : tanliqing2010@163.com
# @File    : kmeans.py
# @Software: PyCharm
# csdn        https://blog.csdn.net/tanliqing2010
"""


from sklearn import datasets
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from itertools import cycle, islice
from sklearn.cluster import KMeans, MiniBatchKMeans

def load_data(n_sample=1500):
    noisy_circles = datasets.make_circles(n_sample, noise=0.05, factor=0.5)
    
    noisy_moons = datasets.make_moons(n_sample, noise=0.05)
    
    no_structure = np.random.rand(n_sample, 2), np.ones((1, n_sample), dtype=np.int32).tolist()[0]
    
    X, y = datasets.make_blobs(n_sample, random_state=170)
    
    transformation = [[0.6, -0.6], [-0.4, 0.8]]
    X_aniso = np.dot(X, transformation)
    aniso = (X_aniso, y)
    
    varied = datasets.make_blobs(n_sample, cluster_std=[1.0, 2.5, 0.5], random_state=170)
    
    blobs = datasets.make_blobs(n_sample, random_state=8)
    
    data_sets = [noisy_circles, noisy_moons, no_structure, aniso, varied, blobs]
    cluster_nums = [2, 2, 3, 3, 3, 3]
    data_mats = []
    for i in range(len(data_sets)):
        X, y = data_sets[i]
        X = StandardScaler().fit_transform(X)
        X_mat = np.mat(X)
        y_mat = np.mat(y)
        data_mats.append((X_mat, y_mat))
        
    plt.figure()
    # plt.subplots_adjust(left=.02, right=.98, bottom=.001, top=.96, wspace=.05, hspace=.01)
    for i in range(len(data_sets)):
        X, y = data_sets[i]
        X = StandardScaler().fit_transform(X)
        color = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                            '#f781bf', '#a65628', '#984ea3',
                                            '#999999', '#e41a1c', '#dede00']), int(max(y) + 1))))
        plt.subplot(len(data_sets), 1, i+1)
        if i == 0:
            plt.title("self_built data set")
        plt.scatter(X[:, 0], X[:, 1], c=color[y], s=10)
        plt.xlim(-2.5, 2.5)
        plt.ylim(-2.5, 2.5)
    plt.show()
    
    return data_mats, cluster_nums


def computer_distance(vecA, vecB):
    # 计算欧式距离
    return np.sqrt(np.sum(np.power(vecA - vecB, 2)))


def rand_init_center(data_mat, k):
    m, n = data_mat.shape
    # 中心点
    center_pt = np.mat(np.zeros((k, n)))
    for i in range(k):
        index = int(np.random.rand() * m)
        center_pt[i, :] = data_mat[index , :]
    return center_pt

    
def standard_kmeans(data_mat, k):
    """
    1. 初始化k个中心点
    2. 计算每个样本到中心点的距离，最近的就划分为哪个族
    3. 更新中心点
    4. 判断中心点是否有变化
    5. 没有变化，结束
    6. 有变化，回到步骤2
    :param data_mat:
    :param k:
    :return:
    """
    m, n = data_mat.shape
    center_pt = rand_init_center(data_mat, k)
    # 存储每个点到属于哪个族以及距离
    cluster_assment = np.mat(np.zeros((m,2)))
    change = True
    while change:
        change = False
        # 遍历所有样本
        for i in range(m):
            min_dis_index = -1
            min_dis = np.inf
            # 遍历所有的中心点
            for j in range(k):
                dis = computer_distance(center_pt[j, :], data_mat[i, :])
                if dis < min_dis:
                    min_dis = dis
                    min_dis_index = j
            if cluster_assment[i,0] != min_dis_index:
                change = True
            # 更新最近的中心点以及距离
            cluster_assment[i,0] = min_dis_index
            cluster_assment[i,1] = min_dis
        # 如果中心没有变化，那么就可以退出了
        if not change:
            break
        # 更新中心
        for i in range(k):
            # 提取i族的点
            index = cluster_assment[:, 0].A == i
            # nonzero返回一个元组，[0] 是行号，[1]是列号
            index = np.nonzero(index)
            pts = data_mat[index[0], :]
            if pts.shape[0] != 0:
                center_pt[i] = np.mean(pts, axis=0)
    
    return center_pt, cluster_assment


def bi_kmeans(data_mat, k):
    """
    1. 初始化第一个质心
    2. 计算到质心的距离
    3. 找到k个质心
        1. 对每一个族进行尝试 2分族
        2. 计算分组 sse总和
        3. 取一个最小的sse进行真正划分
    :param data_mat:
    :param k:
    :return:
    """
    m, n = data_mat.shape
    center0 = np.mean(data_mat, axis=0, dtype=np.float32)
    cluster_dis = np.mat(np.zeros((m, 2)), dtype=np.float32)
    centter_list = [center0]
    
            

if __name__ == '__main__':
    data_mats, cluster_num = load_data()
    
    for i in range(len(data_mats)):
        data_mat = data_mats[i][0]
        center_ps, cluster_assment = standard_kmeans(data_mat,cluster_num[0])
        # print cluster_assment, center_ps
        y_pred = np.array(cluster_assment[:,0].T, dtype=np.int32)[0]
        color = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                            '#f781bf', '#a65628', '#984ea3',
                                            '#999999', '#e41a1c', '#dede00']), int(max(y_pred) + 1))))
        plt.subplots(1, 1)
        plt.scatter(data_mat[:, 0].T.A[0], data_mat[:, 1].T.A[0], c=color[y_pred], s=10)
    plt.show()