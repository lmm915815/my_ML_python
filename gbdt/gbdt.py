# -*- coding: utf-8 -*-
"""
Created on Sun Dec  9 18:28:03 2018

@author:    tanliqing2010@163.com
csdn        https://blog.csdn.net/tanliqing2010
"""
import cart_regression as cart


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
        dt.fit(dataSet)
        
        preList = dt.predict(dataSet)
        
        mapData(dataSet , preList)
        allTree.append(dt.tree)

    return allTree

if __name__ == '__main__':
    dataSet = cart.loadData()
    allTrees = gbdt(dataSet[:] , 2, 2, 1)
    print allTrees
    