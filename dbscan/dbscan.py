# -*- coding: utf-8 -*-
"""
# @Time    : 2018/12/19 上午 10:52
# @Author  : tanliqing
# @Email   : tanliqing2010@163.com
# @File    : dbscan.py   Density-Based Spatial Clustering of Applications with Noise
# @Software: PyCharm
# csdn        https://blog.csdn.net/tanliqing2010
"""

"""
算法步骤：
    1. 计算每个样本的e领域
    2. 判断e领域的数量大于minPt为核心对象
    3. 密度可达为一个簇
"""

import numpy as np


def computer_distance(vec1, vec2):
    return np.sqrt(np.sum(np.power(vec1 - vec2, 2)))


def computer_e_domian(dataset, e):
    m, n = dataset.shape
    e_domain = dict()
    for i in range(len(dataset)):
        di = dataset[i]
        for j in range(len(dataset)):
            if i == j:
                continue
            dj = dataset[j]
            dis = computer_distance(di, dj)
            if i not in e_domain.keys():
                e_domain[i] = list()
            if dis <= e:
                e_domain[i].append(j)
    return e_domain


def kernel_object(e_domain, min_point):
    ret = []
    for i in range(len(e_domain)):
        domain = e_domain[i]
        if len(domain) >= min_point:
            ret.append(i)
    return ret


def dbscan(dataset, e, min_pt):
    e_domain = computer_e_domian(dataset, e)
    kernel = kernel_object(e_domain, min_pt)
    
    cluster = dict()
    k = 0
    run_list = []
    pop = []

    # 循环每一个核心对象
    for ii in kernel:
        # 判断核心对象是否已经运行过
        if ii in run_list:
            continue
        # 作为起始点
        pop.append(ii)
        
        while len(pop) != 0:
            i = pop.pop()
            pti = dataset[i]
            if k not in cluster.keys():
                cluster[k] = []
            if i in run_list:
                continue
            cluster[k].append(pti)
            run_list.append(i)
            # run_list.append(i)
            for j in kernel:
                if i == j:
                    continue
                ptj = dataset[j]
                if j in run_list:
                    continue
                dis = computer_distance(pti, ptj)
                if dis <= e:
                    pop.append(j)
        k += 1
    return cluster


if __name__ == '__main__':
    from sklearn.datasets import load_iris
    import matplotlib.pyplot as plt
    iris = load_iris()
    dataset = iris.data[:,(2,1)]
    cluster = dbscan(dataset, 1, 30)
    print len(cluster)
    plt.scatter(dataset[:,0], dataset[:,1],marker='o')

    d= np.array(cluster[0])[:,0].tolist()
    for i in range(len(cluster)):
        plt.scatter(np.array(cluster[i])[:,0].tolist(),np.array(cluster[i])[:,1].tolist(), marker='x')
    plt.show()
    
    from sklearn.cluster import DBSCAN
    de = DBSCAN()
    de.fit(dataset)
    print de.labels_