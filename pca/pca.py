# -*- coding: utf-8 -*-
"""
# @Time    : 2018/12/17 上午 11:58
# @Author  : tanliqing
# @Email   : tanliqing2010@163.com
# @File    : pca.py
# @Software: PyCharm
# csdn        https://blog.csdn.net/tanliqing2010
"""


import numpy as np
def pca(dataMat, k):
    # 求每一列的均值
    mean = np.mean(dataMat, axis=0)
    
    xMat = dataMat - mean
    # 协方差矩阵
    covMat = np.dot(xMat.T, xMat)
    # 特征值特征向量
    e_vals, e_vecs = np.linalg.eig(covMat)
    sort_index = np.argsort(e_vals)
    print e_vals, e_vecs
    print sort_index
    
    m , n = dataMat.shape
    if k >= n:
        return np.dot(xMat, e_vecs)
    so = sort_index[n-k:]
    w_vecs = e_vecs[:, so]
    return np.dot(xMat, w_vecs)


if __name__ == '__main__':
    data = np.array([[2.5,2.4],
                    [0.5,0.7],
                    [2.2,2.9],
                    [1.9,2.2],
                    [3.1,3.0],
                    [2.3, 2.7],
                    [2, 1.6],
                    [1, 1.1],
                    [1.5, 1.6],
                    [1.1, 0.9] ])
    ret = pca(data, 1)
    print ret
    from sklearn.decomposition import PCA as pca1
    com = pca1(1)
    ret2 = com.fit(data)
    print 'dddddddddddddddddddd'
    print com.explained_variance_ratio_
    print com.explained_variance_
    print com.transform(data)
    '''
    [[-0.82797019]
 [ 1.77758033]
 [-0.99219749]
 [-0.27421042]
 [-1.67580142]
 [-0.9129491 ]
 [ 0.09910944]
 [ 1.14457216]
 [ 0.43804614]
 [ 1.22382056]]
    '''