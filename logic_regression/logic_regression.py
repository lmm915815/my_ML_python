# -*- coding: utf-8 -*-
"""
Created on Mon Dec  3 13:09:18 2018

@author: Administrator
"""

import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split

def loadData():
    iris = load_iris()
    data = iris.data
    '''
   0-50 为0分类
   51-100 为1分类
   100-150为2分类
   '''
    target = iris.target[:100].reshape((-1,1))
    X = data[:100,[0,1]]
   
    return X , target

def draw(X, target):
    label0 = np.where(target == 0)
    plt.scatter(X[label0 ,0] , X[label0 ,1] , marker='x' , color='b' , label = '0' ,s=15)
    label1 = np.where(target == 1)
    plt.scatter(X[label1 , 0] , X[label1 , 1] , marker='o' , color='r' , label = '1' , s=15)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title('total_view')
    return plt
    


def sigmod(x):
    return 1/(1 + np.exp(-x))
    
def compute_loss_theta(data , target , theta):
    m,n = data.shape
    h = sigmod(np.dot(data , theta))
    # 这里是位置相乘，不是矩阵乘法
    cost1 = -1 * np.sum(target * np.log(h) + (1-target) * np.log(1 - h)) / m
    dW = np.dot(data.T, (h - target)) / m
    
    return dW, cost1

def train(X,y,alpha =0.01,repeat=5000):
    lost = []
    theta = np.ones(X.shape[1]).reshape((-1, 1))
    for i in range(0 , repeat):
        dW , cost = compute_loss_theta(X , y , theta)
        # 梯度下降法
        theta = theta - alpha * dW
        lost.append(cost)
        if i %100 == 0:
            print 'i=%d , cost=%f' %(i,cost)
    
    return lost , theta

def drawLost(lost):
    plt.plot(lost)
    plt.xlabel('iters')    
    plt.ylabel('lost value')
    plt.show()
    
def drawBound(data, target , theta):
    pass
    
def predict(x_test ,  theta):
    return x_test.dot(theta)

if __name__ == '__main__':
    X , y = loadData()
    draw(X , y).show()
    draw(X , y)
    X = np.hstack((np.ones((X.shape[0] , 1)) ,X))
    X_train , X_test , y_train , y_test = train_test_split(X, y ,train_size= 0.7)
    lost , theta = train(X_train , y_train )
    
    x1  = np.arange(4,7.5, 0.5)
    x2 = -(theta[0]+ theta[1] * x1)/ theta[2]
    
    plt.plot(x1,x2)
    plt.show()
    
    y_pred = predict(X_test , theta)
    
    