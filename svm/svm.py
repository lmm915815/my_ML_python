# -*- coding: utf-8 -*-
"""
# @Time    : 2018/12/15 下午 12:20
# @Author  : tanliqing
# @Email   : tanliqing2010@163.com
# @File    : svm.py
# @Software: PyCharm
# csdn        https://blog.csdn.net/tanliqing2010
"""


class Svm(object):
    
    def __init__(self, k_type, c = 1):
        self.type = k_type
        self.C = c
        self.a_list = None
        self.b_list = None
        
    def smo_init(self, target):
        self.a_list = [0 for i in target]
        self.b_list = [(0, 0) for i in target]
    
    def kernel_i_j(self, i, j):
    
    def calculate_error(self, i, j):
    
    def calculate_new_b(self, i, j):
    
    def smo(self):
        # 初始化a ， b
        # 2. 求出a2
        # 3. 求出a1
        # 4. 计算b值，Ei
        # 5. 判断是否满足kkt条件
        pass
    
    def svm(self):
        # 选择核函数 惩罚系数C
        # 2. 计算a
        # 3. 求出支持向量S，即满足 0 < a < C 计算对应的bs ， 求均值 sum(bs) / s
        pass
    
    def fit(self, data_set, target):
        pass