# -*- coding: utf-8 -*-
"""
Created on Sun Dec  9 18:28:03 2018

@author:    tanliqing2010@163.com
csdn        https://blog.csdn.net/tanliqing2010
"""
import cart_regression as cart

import copy
def calaResidual(yReal , yPre):
    return yReal - yPre

def mapData(dataSet , preList):
    for i in range(len(dataSet)):
        dataSet[i][-1] = calaResidual(dataSet[i][-1] , preList[i])
    
    return dataSet


def gbdt(dataSet , nTree , depth , leafSize):
    allTree = []
    
    if nTree == 0:
        return None
    
    for i in range(nTree):                                                   
        dt = cart.cartReg(depth ,leafSize)
       # print dataSet[0][-1]
        dt.fit(dataSet)
        
        preList = dt.predict(dataSet)
        
        mapData(dataSet , preList)
        allTree.append(dt)

    return allTree

def predict(allTree , preList):
    value = 0
    for tree in allTree:
        value += tree.predict(preList)
    return value

if __name__ == '__main__':
    dataSet = cart.loadData()
    newData = copy.deepcopy(dataSet)
    allTrees = gbdt(newData , 20, 3, 1)
    
    print predict(allTrees , [0.0686, 0.0, 2.89, 0.0, 33.2])
    
    #print allTrees
    