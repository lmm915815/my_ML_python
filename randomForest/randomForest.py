# -*- coding: utf-8 -*-
"""
Created on Sat Dec  8 14:15:07 2018

@author:    tanliqing2010@163.com
csdn        https://blog.csdn.net/tanliqing2010
"""
import random as random
import numpy as np

import buildClassifyTree as bt

def getSubDataSet(dataSet ,feaNames , numFeat, ratio = 1.0):
    '''
    para: 
        dataSet 数据集
        ratio   抽样比例
    return:
        抽样之后的数据集
    '''

    subData = []
    length = round(len(dataSet) * ratio)
    while len(subData) < length:
        index = random.randrange(len(dataSet))
        subData.append(dataSet[index])
    lengthCol = len(dataSet[0]) 
    # 随机抽取特征列索引
    indexSet = set()
    if numFeat is not None:
        if numFeat <= lengthCol  - 1:
            
            while len(indexSet) != numFeat:
                indexSet.add(random.randrange(lengthCol-1))
    subFeaNames = []
    dataMat = np.mat(dataSet)
    feaMat = np.mat(feaNames)
    m , n = dataMat.shape
    index = list(indexSet)
    index = sorted(index)
    
    subFeaNames = feaMat[ : , index].tolist()
    
    index.append(-1)
    retMat = dataMat[: , index]
    subData = retMat.tolist()
    # TODO oob数据
    
    return subData ,subFeaNames[0] 

def createRandomForest(dataSet ,feaNames , numTree ,numFeat =None, ratio =1 , depth =-1, leafSize =1):
    '''
    '''
    subDataSets = []
    subFeaNames = []
    for i in range(numTree):
        subData ,subFeaNames= getSubDataSet(dataSet ,feaNames ,numFeat ,  ratio)
        subDataSets.append(subData)
    
    allTree = []
    for subData in subDataSets:
        myTree = bt.buildTree(subData ,subFeaNames ,None ,  depth , leafSize)
        allTree.append(myTree)
        
    return allTree

if __name__ == '__main__' :
    dataSet , feaNames = bt.createDataSet1()
    allTree = createRandomForest(dataSet , feaNames ,3  , 3 )
    
    
    
    