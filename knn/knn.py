# -*- coding: utf-8 -*-
"""
# @Time    : 2018/12/16 下午 2:11
# @Author  : tanliqing
# @Email   : tanliqing2010@163.com
# @File    : knn.py
# @Software: PyCharm
# csdn        https://blog.csdn.net/tanliqing2010
"""

import numpy as np
def createDataSet():
    dataSet = np.array([[1.0, 0.9], [1.0, 1.0], [0.1, 0.2], [0.0, 0.1]])
    target = ['A', 'A', 'B', 'B']
    return dataSet, target


def knn_clf(dataSet, target, pred, k):
    # 计算预测到每个样本的距离
    diff = dataSet - pred
    distance = np.sum(diff ** 2, axis=1) ** 0.5
    # 对距离排序
    sortedindex = np.argsort(distance)
    
    # 统计最近k个的类别，投票决定什么类别
    clfcount = {}
    for i in range(k):
        lable = target[sortedindex[i]]
        if  lable not in clfcount.keys():
            clfcount[lable] = 0
        else:
            clfcount[lable] += 1
    # 字典组成元组，对第二个元素排序，逆序输出
    sortdict = sorted(clfcount.items(), key=lambda item: item[1], reverse=True)
    return sortdict[0][0]
    
    
if __name__ == '__main__':
    dataset, target = createDataSet()
    value = knn_clf(dataset, target, np.array([1.2, 1.0]), 3)
    print value
    
    