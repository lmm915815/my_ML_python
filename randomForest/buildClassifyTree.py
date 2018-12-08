# -*- coding: utf-8 -*-
"""
Created on Sat Dec  8 15:10:28 2018

@author:    tanliqing2010@163.com
csdn        https://blog.csdn.net/tanliqing2010
"""
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

def spiltData(dataSet , spIndex , spValue , isContinue = False):
    '''
    '''
    rLeft = []
    rRight = []
    for feat in dataSet:
        # 离散值处理方式
        if not isContinue:
            if feat[spIndex] == spValue:
                ret = feat[:spIndex]
                ret.extend(feat[spIndex+1:])
                rLeft.append(ret)
            else:
                ret = feat[:spIndex]
                ret.extend(feat[spIndex+1:])
                rRight.append(ret)
        # 连续值处理方式
        else:
            if feat[spIndex] <= spValue:
                rLeft.append(feat)
            else:
                rRight.append(feat) 
    return rLeft , rRight

def calaGini(dataSet):
    featList = [ fea[-1] for fea in dataSet]
    featDict = {}
    for feat in featList:
        if feat not in featDict.keys():    # 判断是否在集合中 in、 not in 
            featDict[feat] = 0
        featDict[feat] += 1
    gini = 0.0
    num = len(dataSet)
    for feat in featDict:
        prob = featDict[feat] / float(num)
        gini += prob * (1 - prob)
    return gini
        

def findBestSplit(dataSet ,isContinueList):
    '''
    para: 
        dataSet     数据集
        
    return:
        spIndex   最佳特征值索引
        spValue     最佳特征值
        minGini     gini指数
    '''
    numFeat = len(dataSet[0]) - 1
    minGini = 9999
    numData = float(len(dataSet))
    spIndex = -1
    spValue = 0
    
    # 遍历所有的特征
    for i in range(numFeat):
        featValues = [fea[i] for fea in dataSet]
        # 遍历所有的特征值
        for value in featValues:
            subLeft, subRight = spiltData(dataSet , i , value , isContinueList[i])
            gini = len(subLeft)/numData * calaGini(subLeft) + len(subRight) / numData * calaGini(subRight)
            if gini < minGini:
                minGini = gini
                spIndex = i 
                spValue = value
    return spIndex , spValue , minGini
    
def majorityCnt(dataSet):
    '''
    投票决定类别
    '''
    featList = [ fea[-1] for fea in dataSet ]
    feaDict = {}
    for fea in featList:
        if fea not in feaDict:
            feaDict[fea] = 0
        feaDict[fea] += 1
    
    # 这里是吧dict转换为元组列表，对第二个元素进行排序，返回元组列表
    labels = sorted(feaDict.items() ,key=lambda item:item[1] , reverse=True)
    return labels[0][0]
    
def buildTree(dataSet  ,feaNames , isContinueList = None , depth = -1  , leafSize = 1):
    '''
    para: 
        dataSet     数据集
        depth       树的深度 默认是不限制树的深度
        leaf_size   叶子节点最少数量样本
    return:
        返回树结构
    '''
    myTree = {}
    # 定义递归出口信息
    if depth == 0 :             # 树的深度达到
        return majorityCnt(dataSet)
    if len(dataSet) < 2 * leafSize:       # 叶子节点样本需要保证
        return majorityCnt(dataSet)
    # 所有的特征都已经使用完毕，但是还没有分出类别，投票决定了
    if len(dataSet[0]) == 1 :
        return majorityCnt(dataSet)
    # 数据集都是一个类别的了
    labels = [fea[-1] for fea in dataSet]
    if labels.count(labels[0]) == len(labels) :
        return majorityCnt(dataSet)
    if isContinueList is None:
        isContinueList = [False for i in range(len(dataSet[0]))]
    spIndex , spValue , minGini = findBestSplit(dataSet , isContinueList)
    feaName = feaNames[spIndex]
    myTree['spIndex'] = spIndex
    myTree['spValue'] = spValue
    myTree['spFeatName'] = feaName
    subLeft  , subRight = spiltData(dataSet , spIndex , spValue , isContinueList[spIndex])
    myTree['left']  = buildTree(subLeft  ,feaNames ,isContinueList, depth-1 , leafSize)
    myTree['right'] = buildTree(subRight ,feaNames ,isContinueList, depth-1 , leafSize)
    
    return myTree
    
if __name__ == '__main__' :
    dataSet , feaNames = createDataSet1()
    isContinueList = [True , True , True ,True,True]
    myTree = buildTree(dataSet ,feaNames , isContinueList)
    print myTree
    
    
    
    