# -*- coding: utf-8 -*-
"""
Created on Sun Dec  2 14:58:36 2018

@author: Administrator
"""

from sklearn import datasets
import sklearn
import numpy as np

import matplotlib.pylab as plt

'''
CRIM：城镇人均犯罪率。
ZN：住宅用地超过 25000 sq.ft. 的比例。
INDUS：城镇非零售商用土地的比例。
CHAS：查理斯河空变量（如果边界是河流，则为1；否则为0）。
NOX：一氧化氮浓度。
RM：住宅平均房间数。
AGE：1940 年之前建成的自用房屋比例。
DIS：到波士顿五个中心区域的加权距离。
RAD：辐射性公路的接近指数。
TAX：每 10000 美元的全值财产税率。
PTRATIO：城镇师生比例。
B：1000（Bk-0.63）^ 2，其中 Bk 指代城镇中黑人的比例。
LSTAT：人口中地位低下者的比例。
target:
MEDV：自住房的平均房价，以千美元计。
'''

def loadData():  
    boston = datasets.load_boston()
    data = boston.data
    
    data = np.column_stack( (np.ones(data.shape[0]) , data) )
    target = boston.target
    return data,target

def autoNom(data):
    
    num_fearture = data.shape[1]
    print num_fearture
    for i in range(0 , num_fearture):
        max0 = data[:,i].max()
        min0 = data[:,i].min()
        if min0 == max0 :
            continue
        for j in range(0 , len(data[:,i])):
            data[j,i] = (data[j , i] - min0)/(float)(max0 - min0)
    
    return data

def spiltData(data , target):
    X_train , X_test , y_train , y_test = sklearn.model_selection.train_test_split(
            data , target , test_size =0.3 ,random_state=0)
    return X_train , X_test , y_train , y_test


def linear_regression_mat(data,target):
    X = np.mat(data)
    X1 = X.T.dot(X)
    if np.linalg.det(X1) == 0:
        theta = np.linalg.solve(X.T , X).dot( X.T).dot( target)
        print 'dddd'
        return theta
    theta = np.linalg.inv(X1).dot(X.T).dot( target)
    #theta = np.linalg.inv(X1) * (X.T) *  target
    return theta

def predict(X_test , theta):
    return X_test.dot(theta.T)

def score(y_test , y_pre):
    y = y_test - y_pre
    y = y ** 2
    y = y.sum() / y_test.shape[0]
    y = y ** 0.5
    return y
    
    
    

if __name__ =='__main__':
    X , y = loadData()
    X = autoNom(X)
    X_train , X_test , y_train , y_test = spiltData(X , y)
    theta = linear_regression_mat(X_train,y_train)
    
    y_predict = predict(X_test , theta)
    
    s = score(y_test.reshape((-1,1)) , np.array(y_predict).reshape(1,-1)[0])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    ax.scatter(X[:,1] , y )
   
    
    plt.show()    
    
    #score()
    
