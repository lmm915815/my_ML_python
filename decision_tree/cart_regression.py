# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 12:18:01 2018

@author:    tanliqing2010@163.com
csdn        https://blog.csdn.net/tanliqing2010
"""


from copy import  deepcopy
def loadData(fileName):
    dataSet = []
    
    fd = open(fileName)
    for line in fd.readlines():
        curr = line.strip().split('\t')
        theLine = map(float , curr)  # 把所有的字段都映射成浮点型数据
        dataSet.append(theLine)
    
    return dataSet

def calaError(dataSet  , mean0 = -1):
    
    numCount = len(dataSet)
    target = [t[-1] for t in dataSet]
    if mean0 == -1:
        mean = sum(target) / numCount
    else:
        # 这里主要是在后剪枝的时候，需要使用
        mean = mean0
    
    err = sum([ (t-mean)**2  for t in target ]) 
    return err

def meanValue(dataset):
    target = [t[-1] for t in dataset]
    return sum(target) / float(len(target))

def spiltDataSet(dataSet , col , value):
    retSetLeft = []
    retSetRight= []
    for fea in dataSet:
        if fea[col] <= value:
            retSetLeft.append(fea)
        else:
            retSetRight.append(fea)
    return retSetLeft , retSetRight


def chooseBestFeatureToSpilt(dataSet ,ops):
    numFeat = len(dataSet[0]) - 1
    bestErr = 999999
    bestIndex = -1
    bestValue = 0
    minLeftCount = ops[1]
    minErr = ops[0]
    baseErr = calaError(dataSet)
    
    # 数据不可分，限制叶子节点中样本数量
    if len(dataSet) <= 2 *  minLeftCount:
        return None,meanValue(dataSet),baseErr
    
    for i in range(numFeat):
        colList = [fea[i] for fea in dataSet]
        feaList = set(colList)
        for feaValue in feaList:
            subLeft ,subRight = spiltDataSet(dataSet , i , feaValue)
            if len(subLeft)<minLeftCount or len(subRight)<minLeftCount:
                continue
            newErr = calaError(subLeft) + calaError(subRight)
            if newErr < bestErr:
                bestErr = newErr
                bestIndex = i
                bestValue = feaValue
    # 当数据集不可分，或者误差小于指定的阈值的情况下
    if abs(baseErr - bestErr) < minErr :
        return None,meanValue(dataSet),baseErr
    
    return bestIndex , bestValue , bestErr

def createTree(dataSet, ops=(1, 4)):
    
    index , value ,err= chooseBestFeatureToSpilt(dataSet , ops)
    
    if index is None:
        return value
    myTree = {}
    myTree['spIndex'] = index 
    myTree['spValue'] = value
    
    subLeft , subRight = spiltDataSet(dataSet , index , value)
    myTree['left'] = createTree(subLeft , ops)
    myTree['right'] = createTree(subRight , ops)
    
    return myTree

def isLeaf(tree):
    return not isinstance(tree , dict)

def isTree(tree):
    return not isLeaf(tree)


def predict(myTree , preVec):
    # 出口
    if isLeaf(myTree):
        return myTree
    
    index = myTree['spIndex']
    value = myTree['spValue']
    preValue = preVec[index]
    if preValue <= value:
        return predict(myTree['left'] , preVec)
    else:
        return predict(myTree['right'] , preVec)

def getMean(tree):
    if isTree(tree['left']) :
        tree['left'] = getMean(tree['left'])
    if isTree(tree['right']):
        tree['right'] = getMean(tree['right'])
    # 如果是叶子节点   递归出口   
    return (tree['left'] + tree['right']) / 2.0


def prune(tree , testData):
    '''
    1. 把测试数据分配到回归树的叶子节点
    2. 计算合并叶子节点前后 误差的变化
    3. 如果没有分配到数据，那么就直接把子树合并起来
    '''
    
    if len(testData) == 0:
        return getMean(tree)
    
    # 根据回归树来划分我们的测试数据
    if isTree(tree['left']) or isTree(tree['right']):
        dataLeft , dataRight = spiltDataSet(testData , tree['spIndex'] , tree['spValue'])
    
    if isTree(tree['left']):
        tree['left'] = prune(tree['left'] , dataLeft)
    if isTree(tree['right']):
        tree['right'] = prune(tree['right'] , dataRight)
    
    if isLeaf(tree['left']) and isLeaf(tree['right']):
        dataLeft , dataRight = spiltDataSet(testData , tree['spIndex'] , tree['spValue'])
        #sumErr = calaError(dataLeft) + calaError(dataRight)
        
        #sumErrMer = calaError(testData)
        #value = meanValue(testData)
        ## 均值就是叶子节点的值 ， 不需要自己在计算的
        sumErr = calaError(dataLeft , tree['left']) + calaError(dataRight , tree['right'])
        value = tree['left'] + (tree['right'])/float(2)
        sumErrMer = calaError(testData , value)
        if sumErr > sumErrMer:
            return value
        else:
            return tree
    else:
        return tree
    
        
    

if __name__ == '__main__':
    dataSet = loadData('test.txt')
    myTree = createTree(dataSet , ops=(1,10))
    pre = predict(myTree, [0.56])
    
    pTree = prune(deepcopy(myTree) , dataSet)
    pre1 = predict(pTree, [0.56])
    
    
