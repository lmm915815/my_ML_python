# -*- coding: utf-8 -*-
"""
Created on Sun Dec  9 15:12:14 2018

@author:    tanliqing2010@163.com
csdn        https://blog.csdn.net/tanliqing2010
"""

class cartReg(object):
    def __init__(self  , depth =None, leafSize=1):
        if depth is None:
            depth = -1
        self.depth = depth
        self.leafSize = leafSize
        self.tree = None
        
        
    @staticmethod
    def spiltData( dataSet ,target, spIndex , spValue):
        retLeft = []
        retRight = []
        tarLeft = []
        tarRight = []
        for i in range(len(dataSet)):
            fea = dataSet[i]
            if fea[spIndex] <= spValue:
                retLeft.append(fea)
                tarLeft.append(target[i])
            else:
                tarRight.append(target[i])
                retRight.append(fea)
        return retLeft , retRight , tarLeft , tarRight
    
    
    def calaError(self, targets):
        
        if len(targets) == 0:
            return 0.0
        meanValue = sum(targets) / float(len(targets))
        err = 0.0
        for t in targets:
            err += (t - meanValue) ** 2
        return err
    
    def majortyValue(self, targets):
        
        meanValue = sum(targets) / float(len(targets))
        return meanValue
    
    def findBestFeatureToSpilt(self, dataSet , target):
        numFeat = len(dataSet[0]) - 1
        errMin = 99999
        bestIndex = -1
        bestValue = 0
        # 循环所有的特征
        for i in range(numFeat):
            # TODO 这里是回归，特征值可能特别多，所以可以按照一定的步长来计算
            featValueList = set([fea[i] for fea in dataSet])
            for value in featValueList:
                errSum  = 0.0
                subLeft, subRight , subTarLeft , subTarRight = self.spiltData(dataSet ,target, i , value)
                errSum  = self.calaError(subTarLeft)
                errSum += self.calaError(subTarRight)
                if errSum < errMin:
                    errMin = errSum
                    bestIndex = i
                    bestValue = value
        return bestIndex , bestValue , errMin
    
    
    def buildTree(self, dataSet , target , depth , leafSize):
        # 出口信息
        if depth == 0:
            return self.majortyValue(target)
        if len(dataSet) < 2 * leafSize:
            return self.majortyValue(target)
        if target.count(target[0]) == len(target):
            return self.majortyValue(target)
        myTree = {}
        spIndex , spValue , errMin = self.findBestFeatureToSpilt(dataSet ,target)
        subLeft , subRight ,subTarLeft , subTarRight  = self.spiltData(dataSet , target ,spIndex , spValue)
        if len(subLeft) < leafSize or len(subRight) < leafSize:
            return self.majortyValue(target)
        
        myTree['spIndex'] = spIndex
        myTree['spValue'] = spValue
        myTree['left'] = self.buildTree(subLeft ,subTarLeft, depth -1 , leafSize)
        myTree['right'] = self.buildTree(subRight ,subTarRight , depth -1 , leafSize)
        
        return myTree
    
    def fit(self , dataSet , target):
        self.tree = self.buildTree(dataSet  ,target, self.depth , self.leafSize)
        return self
    
    
    def predict0(self ,preVec , tree):
        if not isinstance(tree ,dict):
            return tree
        
        spIndex = tree['spIndex']
        spValue = tree['spValue']
        value = preVec[spIndex]
        
            
        if value <= spValue:
            if isinstance(tree['left'], dict):
                return self.predict0(preVec , tree['left'])
            else:
                return tree['left']
        else:
            if isinstance(tree['right'] , dict):
                return self.predict0(preVec, tree['right'])
            else:
                return tree['right']
        
    def predict(self , preList):
        retList = []
        if isinstance( preList[0] , list):
            for preVec in preList:
                value = self.predict0(preVec , self.tree)
                retList.append(value)
        else:
            return self.predict0(preList , self.tree)
        return retList

def loadData():
    from sklearn.datasets import load_boston
    # import numpy as np
    
    boston = load_boston()
    dataSet = boston.data[:100 , (0,1,2,3)].tolist()
    target = boston.target[:100]
    return dataSet, target

     
if __name__ == '__main__' :

    
    dataSet , target = loadData()
    
    dt = cartReg(5,1)
    dt.fit(dataSet , target)
    print dt.predict([0.00632, 18.0, 2.31, 0.0, 4.98 ])#, 24
    
        