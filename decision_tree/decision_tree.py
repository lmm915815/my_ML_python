# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 11:36:00 2018

@author: Administrator
"""

import numpy as np

def createDataSet1(): 
    # 创造示例数据 
    dataSet = [[0, 0, 0, 0, 'no'], #数据集 
               [0, 0, 0, 1, 'no'], 
               [0, 1, 0, 1, 'yes'], 
               [0, 1, 1, 0, 'yes'], 
               [0, 0, 0, 0, 'no'], 
               [1, 0, 0, 0, 'no'],
               [1, 0, 0, 1, 'no'], 
               [1, 1, 1, 1, 'yes'], 
               [1, 0, 1, 2, 'yes'], 
               [1, 0, 1, 2, 'yes'], 
               [2, 0, 1, 2, 'yes'], 
               [2, 0, 1, 1, 'yes'], 
               [2, 1, 0, 1, 'yes'], 
               [2, 1, 0, 2, 'yes'], 
               [2, 0, 0, 0, 'no']] 
    labels = ['age', 'hasWork', 'hasHouse', 'xindai'] #特征标签 
    return dataSet, labels #返回数据集和分类属性




def calaShannonEnt(dataSet , index = -1):
    #m , n = dataSet.shape
    m = len(dataSet)

    labelCounts = {}
    # 提取每个分类数量
    for line in dataSet:
        fea = line[index]
        if fea not in labelCounts.keys():
            labelCounts[fea] = 0
        labelCounts[fea] += 1
    shannonEnt = 0
    for key in labelCounts:
        prob = float(labelCounts[key]) / m
        shannonEnt -= prob * np.log(prob)
    
    return shannonEnt

def spiltDataSet(dataSet  , axis , value ):
    '''
    para: 
        dataSet:        特征 + 标签数据
        axis:           根据那个特征进行分割数据
        value:          根据那个值进行分割数据
    return:
        指定特征指定值的数据集
    '''
    retData = []
    for fea in dataSet:
        if fea[axis] == value:
            feaVec = fea[:axis]
            feaVec.extend(fea[axis+1:])
            retData.append(feaVec)
    
    return retData

def chooseBestFeatureToSplit(dataSet , imp = 0 ):
    '''
    原理：
        外循环每一个特征，内循环每一个特征属性
    para: 
        数据集 np.array
    return: 
        特征索引
    '''
    m = len(dataSet) 

    n = len(dataSet[0]) -1
    if n == 1:
        return 0
    baseEnt = calaShannonEnt(dataSet)
    bestFea = -1
    bestGain = 0
    bestGainRate = 0
    # 循环所有的特征
    for i in range(n):
        # 所有行，第i列数据
        FeaList = [ fea[i] for fea in dataSet ]
        uniqueVars = set(FeaList)
        newEnt = 0
        for value in uniqueVars:
            subData = spiltDataSet(dataSet , i , value)
            prob = len(subData) / float(m)
            newEnt += prob * calaShannonEnt(subData)
        infoGain = baseEnt - newEnt
        if imp == 0:  # 计算信息增益  ID3算法
            if infoGain > bestGain:
                bestFea = i
                bestGain = infoGain
        else:   # 这里是计算的信息增益比 c4.5算法
            #这里计算的是某个特征条件下的熵 ， 而不是集合类别的熵
            
            iv = calaShannonEnt(dataSet , i)
            if iv == 0:
                continue
            infoGainRate = infoGain / iv
            if bestGainRate < infoGainRate:
                bestFea = i
                bestGainRate = infoGainRate
    return bestFea

            
def majorityCnt(classList):
    '''
    叶子节点判断属于什么类别  投票的方式
    '''
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClass = sorted(classCount.items() ,reverse = True )
    return sortedClass[0][0]

def createTree(dataSet , labels , imp = 0):
    '''
    思路： 
        1. 选择最佳特征创建根节点 
        2. 遍历该特征所有属性，迭代创建决策树
    
    '''
    # 获取所有的数据类别列表
    classList = [lab[-1] for lab in dataSet]
    # 递归出口
    # 如果第一个类别就是全部
    if (classList.count(classList[0]) == len(classList)):
        return classList[0]
    # 所有特征全部使用完了，但是有可能还是没有完全分开，那么就投票决定了
    if len(dataSet[0]) == 1 :
        return majorityCnt(classList)

    bestFea = chooseBestFeatureToSplit(dataSet , imp)
    print '------' , bestFea
    bestFeaLabel = labels[bestFea]
    myTree = {bestFeaLabel :{}}
    print 'diyici' , myTree
    del(labels[bestFea])
    # 
    featValues = [fea[bestFea] for fea in dataSet]
    uniqueVals = set(featValues)
    
    for value in uniqueVals:
        subLabels = labels[:]
        myTree[bestFeaLabel][value] = createTree(spiltDataSet(
                dataSet , bestFea , value) , subLabels)
        print 'dierci :----' , myTree
    return myTree

def predict(myTree , labels, preLabel):
    # 获取第一个根节点
    firstFea = myTree.keys()[0]
    # 获取孩子节点
    secondDict = myTree[firstFea]
    # 获取根节点在label中的索引
    featIndex = labels.index(firstFea)
    # 遍历根节点所有的边
    for key in secondDict.keys():
        # 判断边是与给定的相等，走向对应的分支
        if preLabel[featIndex] == key:
            if isinstance(secondDict[key] , dict):
                classLabel = predict(secondDict[key] , labels , preLabel)
            else:
                classLabel = secondDict[key]
    return classLabel

if __name__ == '__main__':
    
    dataSet , labels = createDataSet1()

    myTree = createTree((dataSet) , labels[:] , 0) 
    print  myTree
    pred = predict(myTree , labels , [1,1 ,0,1])
    print pred
    
        
    
    
    
    
    
    
    
    