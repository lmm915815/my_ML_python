# -*- coding: utf-8 -*-
"""
# @Time    : 2018/12/16 下午 2:35
# @Author  : tanliqing
# @Email   : tanliqing2010@163.com
# @File    : kd_tree.py
# @Software: PyCharm
# csdn        https://blog.csdn.net/tanliqing2010
"""

import numpy as np


def calculate_variance(data):
    mean = sum(data) / float(len(data))
    ret = sum([ (d - mean) ** 2 for d in data]) ** 0.5
    return ret
  
    
def get_min_variance(dataset):
    # 计算方差
    numFeat = len(dataset[0])
    max_variance = 0
    index = -1
    for i in range(numFeat):
        subdata = [fea[i] for fea in dataset]
        variance = calculate_variance((subdata))
        if variance > max_variance:
            max_variance = variance
            index = i
    return max_variance, index


def calc_distance(vec1, vec2):
    ret = [v1 - v2 for v1, v2 in zip(vec1, vec2)]
    return sum(map(lambda x:x * x ,ret)) ** 0.5


def get_mid_value(data):
    # 获取中值
    data_index = np.argsort(data)
    return data[data_index[len(data) / 2]]


def spilt_data(dataset, index):
    data = [fea[index] for fea in dataset]
    mid_value = get_mid_value(data)
    subLeft = []
    subRight = []
    for fea in dataset:
        if fea[index] < mid_value:
            subLeft.append(fea)
        elif fea[index] > mid_value:
            subRight.append(fea)
    return subLeft, subRight, mid_value
    
    
def create_kd_tree(dataSet, min_leaf_size = 1, max_depth = -1):
    
    if max_depth == 0:
        return dataSet
    if len(dataSet) <= 2 * min_leaf_size:
        return dataSet
    tree = {}
    min_variance, index = get_min_variance(dataSet)
    subleft, subRight, mid_value = spilt_data(dataSet, index)
    tree['spIndex'] = index
    tree['spValue'] = mid_value
    tree['left'] = create_kd_tree(subleft, min_leaf_size, max_depth - 1)
    tree['right'] = create_kd_tree(subRight, min_leaf_size, max_depth -1)
    
    return tree


def get_search_path(tree, pred, all_path = []):
    spIndex = tree['spIndex']
    spValue = tree['spValue']
    
    
    if pred[spIndex] <= spValue:
        if isinstance(tree['left'], dict):
            all_path.append(tree['left'])
            get_search_path(tree['left'],pred, all_path)
    else:
        if isinstance((tree['right']), dict):
            all_path.append(tree['right'])
            get_search_path(tree['right'], pred, all_path)
    
    return all_path
 
 
def calc_best_near(tree, search_path):
    
    while search_path:
        end = search_path.pop()
        sp = end['spValue']
        
        
       
        



def predict(tree, pred, k):
    pass


if __name__ == '__main__':
    
    from sklearn.datasets import load_iris
    iris = load_iris()
    dataset = iris.data.tolist()
    target = iris.target.tolist()
    tree = create_kd_tree(dataset)
    search_path = []
    get_search_path(tree, [1.2, 1.0,1.,10],search_path)
    