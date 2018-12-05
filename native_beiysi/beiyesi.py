# -*- coding: utf-8 -*-
"""
Created on Mon Dec  3 16:20:32 2018

@author: Administrator
"""

import numpy as np

def loadData():
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]    #1 is abusive, 0 not
    return postingList,classVec


def createWorldVec(dataSet):
    '''
    # 根据文本生成词向量
    para:
        dataSet:    所有的文本
    
    return: 
        文本去重的词向量
    
    '''
    vec  = set()
    for doc in dataSet:
        vec = vec | set(doc)
    return list(vec)


def setOfWords2Vec(vec , data):
    '''
    文本转化为词向量
    para:   
        vec:    词向量
        data:   需要转化的文本
    return: 
        retVec:     转化的词向量
    '''
    retVec = [0] * len(vec)
    for word in data:
        if word in vec:
            retVec[vec.index(word)] +=1
        else:
            print 'word is not in word vectory'
    return retVec


def trainNB0(trainMat , trainCategory):
    '''
     计算文本每个词出现概率
    para:
        trainMat:   文本词向量
        trainCategory:  分类向量
    return：
        p1Vec:      分类1的词向量概率
        p0Vec:      分类0的词向量概率
        pAbusive:   分类1的先验概率
    '''
    m,n = trainMat.shape
    # 计算正类的概率
    pAbusive = np.sum(trainCategory) / float(m)
    p0Num = np.ones(n)
    p1Num = np.ones(n)
    p0Denom = 2.0
    p1Denom = 2.0
    
    # 循环所有的文本向量
    for i in range(m):
        if trainCategory[i] == 1:
            # 对应位置相加
            p1Num = p1Num + trainMat[i]
            # 累计所有为1的值
            p1Denom = p1Denom + np.sum(trainCategory[i])
        else:
            p0Num += trainMat[i]
            p0Denom += np.sum(trainCategory[i])
    # 采用log形式，连乘就可以转化为累加的形式，后续可以减少小概率连乘导致等于0
    p1Vec = np.log(p1Num / p1Denom)
    p0Vec = np.log(p0Num / p0Denom)
    
    return p1Vec , p0Vec , pAbusive
    
# 
def classifyNB(vec2Classify , p0Vec , p1Vec , pClass1):
    '''
    para:
        vec2Classify:我们需要检测的词向量
        p0Vec:      负类词向量的概率
        p1Vec:      正类词向量的概率
        pClass1:    正负的概率
    
    return: 1 正类
            0 负类
    '''
    p1 = sum(vec2Classify * p1Vec) + np.log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + np.log(1-pClass1)
    if p1>p0:
        return 1
    else:
        return 0



if __name__ == '__main__':
    data , target = loadData()
    worldList = createWorldVec(data)
    
    trainMat = []
    for line in data:
        trainMat.append(setOfWords2Vec(worldList , line))
        
    p1Vec ,p0Vec ,p1Class = trainNB0(np.array(trainMat) , np.array(target) )
    
    testDoc = ['love' , 'my' , 'dalmation']
    testVec = np.array(setOfWords2Vec(worldList , testDoc))
    ret = classifyNB(testVec , p0Vec , p1Vec ,p1Class)
    print testDoc , 'classify as ' ,ret
    
