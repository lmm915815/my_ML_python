# -*- coding: utf-8 -*-
"""
# @Time    : 2018/12/18 下午 4:04
# @Author  : tanliqing
# @Email   : tanliqing2010@163.com
# @File    : lda.py
# @Software: PyCharm
# csdn        https://blog.csdn.net/tanliqing2010
"""

import numpy as np


def center(data_mat, target):
    clf_list = set(target)
    data = []
    
    for clf in clf_list:
        data.append(data_mat[target == clf])
    
    center0 = []
    for d in data:
        center0.append(np.mean(d, axis=0))
    
    return data, center0
    
    
def computer_sw_mat(data, center0, n):
    ret_mat = np.mat(np.zeros((n, n)))
    for i in range(len(data)):
        xi_center = center0[i]
        xi_center = np.mat(xi_center).T
        clf_data = data[i]
        for d in clf_data:
            d = np.mat(d).T
            
            ret_mat += (d - xi_center) * (d - xi_center).T
    return ret_mat



def computer_sb_mat(center0, n):
    ret_mat = np.mat(np.zeros((n, n)))
    k = len(center0)
    for i in range(k):
        
        for j in range(i+1, k):
            ci = np.mat(center0[i]).T
            cj = np.mat(center0[j]).T
            ret_mat += (ci - cj) * (ci - cj).T
    return ret_mat


def lda(dataset, target, k):
    data, center0 = center(dataset, target)
    # print center0
    m, n = dataset.shape
    sw_mat = computer_sw_mat(data, center0, n)
    # print sw_mat
    sb_mat = computer_sb_mat(center0, n)
    # print sb_mat
    eig_val, eig_vec = np.linalg.eig(sw_mat.I * sb_mat)
    print eig_val, eig_vec
    
    decom_mat = eig_vec[:k]
    new_data = dataset * decom_mat.T
    return new_data

    
if __name__ == '__main__':
    from sklearn.datasets import load_iris
    iris = load_iris()
    dataset = iris.data
    target = iris.target
    new_data = lda(dataset, target, 2)
    # print new_data
    
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA1
    de = LDA1(solver='eigen', n_components=2)
    de.fit(dataset, target)
    newd = de.transform(dataset)
    print de.explained_variance_ratio_
    # print newd
    import matplotlib.pyplot as plt
    plt.subplot(311)
    plt.scatter(dataset[:,2].tolist(),dataset[:,1].tolist(), marker='p')
    plt.subplot(312)
    plt.scatter(new_data[:, 0].tolist(), new_data[:, 1].tolist(), marker='o')
    plt.subplot(313)
    plt.scatter(newd[:, 0].tolist(), newd[:, 1].tolist(), marker='x')
    plt.show()
    
    
    