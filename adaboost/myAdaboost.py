# -*- coding: utf-8 -*-
"""
Created on Fri Dec  7 10:02:36 2018

@author:    tanliqing2010@163.com
csdn        https://blog.csdn.net/tanliqing2010
"""

import numpy as np

def loadSimData(): 
    '''
    输入：无
    功能：提供一个两个特征的数据集
    输出：带有标签的数据集
    ''' 
    datMat = np.matrix([[1. ,2.1],[2. , 1.1],[1.3 ,1.],[1. ,1.],[2. ,1.]]) 
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0] 
    return datMat, classLabels



def classifyData(dataMat , index , spValue , thresholdIneq):
    '''
    默认是1 分类，满足条件就是-1 分类
    指定特征列，特征值
    返回分类
    '''
    retMat = np.ones((dataMat.shape[0] , 1))
    if thresholdIneq == 'lt' :
        retMat[dataMat[:,index] <= spValue] = -1
    else:
        retMat[dataMat[:,index] > spValue] = -1
    return retMat
    

def findWeakClassify(dataSet , classLabels , weigth):
    '''
    para:
        dataSet         数据集
        classLabel
        weight
    return:
        bestClassEst    弱分类器对样本的分类结果
        bestStump       分类器 一个字典， 包括 最佳特征以及特征值，符号
        minErr          误差率
    '''
    dataMat = np.mat(dataSet)
    labelMat = np.mat(classLabels)
    m ,n = dataMat.shape
    stepNum = 10
    minErr = np.inf
    bestStump = {}
    bestClassEst = np.mat(np.zeros((m,1)))
    # 遍历所有的特征列
    for i in range(n):
        feaMin = dataMat[:,i].min()
        feaMax = dataMat[:,i].max()
        stepSize = (feaMax - feaMin) / stepNum
        # 循环所有可能的分割点
        for j in range(-1 , stepNum +1 ):
            # 循环所有分割符号
            for thresholdIneq in ['lt' , 'gt']:
                spValue = feaMin + j * stepSize
                predClass = classifyData(dataMat , i , spValue , thresholdIneq)
                errArray = np.ones((m ,1))
                # 赋值为-0就是正确分类的样本， 为1就是错误分类的样本
                errArray[predClass == labelMat.T] = 0
                # 计算分类错误率
                weightErr = weigth.T.dot(errArray)  # == weight * errArray.T
                if weightErr < minErr :
                    minErr = weightErr
                    bestStump['spIndex'] = i
                    bestStump['spValue'] = spValue
                    bestStump['thrsholdIneq'] = thresholdIneq
                    bestClassEst = predClass.copy()
    return bestClassEst , bestStump , minErr

def calaAlpha(minErr):
    return float(0.5 * np.log((1 - minErr) / max(minErr , 1e-16)))

def updateWeight(bestClass  , labels , weigth ,alpha):
    '''
    para: 
        beasClass       弱分类器对样本的分类结果
        labels          真实样本的分类
        weigth          样本的权重
        alpha           弱分类器的权重
    return:
        更新后的样本权重
    '''
    # 假设弱分类器把数据分为 -1 类，但是真实类是  1 直接相乘---ok
    expAll = np.multiply(bestClass , -1 * alpha * labels.T)
    # 对应样本的权重  *  exp(+- alpha)
    w = np.multiply(weigth , np.exp(expAll))
    # 归一化
    return w / w.sum()

    

def train(dataArray , classLabels , iters = 40):
    weakClasss = []
    m , n = np.shape(dataArray)
    weight = np.mat( np.ones((m,1)) ) / float(m) 
    labelMat = np.mat(classLabels)
    aggClassEst = np.mat(np.zeros((m,1)))
    # 迭代多少次
    for i in range(iters):
        # 计算弱分类器
        bestClassEst , bestStump , minErr = findWeakClassify(dataArray , classLabels , weight)
        # 计算弱分类器的权值
        alpha = calaAlpha(minErr)
        bestStump['alpha'] = alpha
        weakClasss.append(bestStump)
        
        # 更新样本的权值
        weight = updateWeight(bestClassEst , labelMat , weight , alpha)
        
        # 计算错误率
        aggClassEst += alpha * bestClassEst
        # np.sign(aggClassEst) != labelMat.T  这个就是错误分类的样本索引
        aggErr = np.multiply(np.sign(aggClassEst) != labelMat.T , np.ones((m,1)))
        
        aggErrRate = aggErr.sum() / float(m)
        if aggErrRate == 0.0 :
            break
    return weakClasss

def predict(weakClass , preMat):
    mat = np.mat(preMat)
    m,n = mat.shape
    aggClass = np.mat(np.zeros((m,1)))
    for i in range(len(weakClass)):
        index = weakClass[i]['spIndex']
        value = weakClass[i]['spValue']
        flag = weakClass[i]['thrsholdIneq']
        alpha = weakClass[i]['alpha']
        preClass = classifyData(mat , index, value , flag)
        
        aggClass += alpha * preClass
    return np.sign(aggClass)
    

if __name__ == '__main__' :
    dataSet , labels = loadSimData()
    weakClass = train(dataSet , labels , 9)
    
    preClass = predict(weakClass , np.mat([ [1,2] ,[2,2] ]))
        
                    
                    
                    
                    
                    
                    
                    