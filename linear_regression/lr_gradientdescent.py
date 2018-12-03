# -*- coding: utf-8 -*-
"""
Created on Mon Dec  3 11:52:31 2018

@author: Administrator
"""


import numpy as np
import matplotlib.pyplot as plt
def batchGradientDescent(X, y , theta , alpha , repeat):
    m = len(X)
    cost = []
    for i in range(0 , repeat):
        hyp = np.dot(X , theta)
        loss = hyp - y
        grad = np.dot(X.T ,loss) / m
        theta = theta - alpha * grad
        cost1 = 0.5 * m * np.sum(np.square(np.dot(X , theta.T) - y))
        cost.append(cost1)
    return theta , cost

if __name__ == '__main__':

    x = np.array([[1,2],[2,1],[3,2.5],[4,3],
              [5,4],[6,5],[7,2.7],[8,4.5],
              [9,2]])
    
    m, n = np.shape(x)
    x_data = np.ones((m,n))
    x_data[:,1:] = x[:,:-1]
    y_data = x[:,-1]
    
    print x_data.shape
    print y_data.shape
    m, n = np.shape(x_data)
    theta = np.ones(n)
    result , cost = batchGradientDescent(x_data,y_data,theta,0.01,1000)
    newy = np.dot(x_data,result)
    fig, ax = plt.subplots()
    ax.plot(x[:,0],newy, 'k--')
    ax.plot(x[:,0],x[:,1], 'ro')
    plt.show()
    print "final: ",result
    print cost
