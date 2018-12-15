# -*- coding: utf-8 -*-
"""
Created on Thu Dec 13 17:17:00 2018

@author:    tanliqing2010@163.com
csdn        https://blog.csdn.net/tanliqing2010
"""


import math
import numpy as np
import cart_regression as cart
class GbdtBin(object):
    """
    针对 0 ， 1 分类的情况
    """
    def __init__(self, n_trees, depth, leaf_size):
        self.n_trees = n_trees
        self.depth = depth
        self.leaf_size = leaf_size
        self.all_trees = []
        self.fValue =None
        self.losts = []
        pass

    def init_f_value(self, target):
        '''
        初始化f0的值
        :param target:
        :return:
        '''
        z = sum(target) / sum([1 - y for y in target])
        self.fValue = [math.log(z) for i in target]
        
        return self
    
    @staticmethod
    def sigmod(z):
        return 1.0 / (1.0 + (float)(math.exp(-z)))
    
    def computer_residual(self, target):
        return [y - self.sigmod(z) for y, z in zip(target, self.fValue)]
    
    @staticmethod
    def spilt_data(data_set, spilt_index, spilt_value):
        index_left = []
        index_right = []
        for i in range(len(data_set)):
            fea = data_set[i]
            if fea[spilt_index] <= spilt_value:
                index_left.append(i)
            else:
                index_right.append(i)
        return index_left, index_right
    
    
    def update_f_value(self, value, index):
        for i in index:
            self.fValue[i] += value
    
    def calculate_leaf_value(self, target, index):
        """
        1. 计算叶子节点值
        2. 更新f值
        :param target:
        :param index: 叶子节点索引
        :return: 叶子节点值
        """
        # 更新叶子节点值
        y = [target[i] for i in index]
        fn = [self.fValue[i] for i in index]
        p = [self.sigmod(f) for f in fn]
        residual = sum(yi - pi for yi, pi in zip(y, p))
        prob = sum([pi * (1 - pi) for pi in p])
        # 更新f值
        value = residual / prob
        self.update_f_value(value, index)
        return value
        
    def update_leaf_value(self, tree, data_set, target):
        spilt_index = tree['spIndex']
        spilt_value = tree['spValue']
        index_left, index_right = self.spilt_data(data_set, spilt_index, spilt_value)
        data_left = [data_set[i] for i in index_left]
        data_right = [data_set[i] for i in index_right]
        if isinstance(tree['left'], dict):
            self.update_leaf_value(tree['left'], data_left, target)
        else:
            tree['left'] = self.calculate_leaf_value(target, index_left)
        
        if isinstance(tree['right'], dict):
            self.update_leaf_value(tree['right'], data_right, target)
        else:
            tree['right'] = self.calculate_leaf_value(target, index_right)
    
    def compute_lost(self, target):
        p = [self.sigmod(z) for z in self.fValue]
        lost = 0.0
        for i in range(len(target)):
            lost -= target[i] * math.log(p[i]) + (1 - target[i]) * math.log(1 - p[i])
        return lost
    
    def fit(self, data_set, target):
        self.init_f_value(target)
        for i in range(self.n_trees):
            new_target = self.computer_residual(target)
            tree = cart.cartReg().buildTree(data_set, new_target, self.depth, self.leaf_size)
            self.update_leaf_value(tree, data_set, target)
            self.all_trees.append(tree)
            lost = self.compute_lost(target)
            self.losts.append(lost)
     
 
    def predict_prob(self, predict_vector):
        fn = 0.0
        for tree in self.all_trees:
            fn += cart.cartReg().predict0(predict_vector, tree)
        return self.sigmod(fn)
    
    def predict(self, predict_vector, thr=0.5):
        prob = self.predict_prob(predict_vector)
        if prob > thr:
            return 1
        return 0

def loadData():
    from sklearn.datasets import load_iris
    iris = load_iris()
    dataSet = iris.data[:100].tolist()
    target = iris.target[:100].tolist()
    return dataSet, target


if __name__ == '__main__':
    
    data_set, target = loadData()
    clf = GbdtBin(10, 3, 1)
    clf.fit(data_set, target)
    value = clf.predict_prob([5.1, 3.5, 1.4, 0.2]) # 0
    print value, clf.losts