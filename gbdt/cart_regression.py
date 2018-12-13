# -*- coding: utf-8 -*-
"""
Created on Sun Dec  9 15:12:14 2018

@author:    tanliqing2010@163.com
csdn        https://blog.csdn.net/tanliqing2010
"""

class cartReg(object):
    def __init__(self  , depth , leafSize):
        if depth is None:
            depth = -1
        self.depth = depth
        self.leafSize = leafSize
        self.tree = None
        
    def setTree(self , tree):
        self.tree = tree
        
    def spiltData(self , dataSet , spIndex , spValue):
        retLeft = []
        retRight = []
        for fea in dataSet:
            if fea[spIndex] <= spValue:
                retLeft.append(fea)
            else:
                retRight.append(fea)
        return retLeft , retRight
    
    def calaError(self , dataSet):
        targets = [t[-1] for t in dataSet]
        if len(targets) == 0:
            return 0.0
        meanValue = sum(targets) / float(len(targets))
        err = 0.0
        for t in targets:
            err += (t - meanValue) ** 2
        return err
    
    def majortyValue(self , dataSet):
        targets = [t[-1] for t in dataSet]
        meanValue = sum(targets) / float(len(targets))
        return meanValue
    
    def findBestFeatureToSpilt(self , dataSet):
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
                subLeft, subRight = self.spiltData(dataSet , i , value)
                errSum  = self.calaError(subLeft)
                errSum += self.calaError(subRight)
                if errSum < errMin:
                    errMin = errSum
                    bestIndex = i
                    bestValue = value
        return bestIndex , bestValue , errMin
    
    def buildTree(self , dataSet , depth , leafSize):
        # 出口信息
        if depth == 0:
            return self.majortyValue(dataSet)
        if len(dataSet) < 2 * leafSize:
            return self.majortyValue(dataSet)
        myTree = {}
        spIndex , spValue , errMin = self.findBestFeatureToSpilt(dataSet)
        subLeft , subRight = self.spiltData(dataSet ,spIndex , spValue)
        if len(subLeft) < leafSize or len(subRight) < leafSize:
            return self.majortyValue(dataSet)
        
        myTree['spIndex'] = spIndex
        myTree['spValue'] = spValue
        myTree['left'] = self.buildTree(subLeft , depth -1 , leafSize)
        myTree['right'] = self.buildTree(subRight , depth -1 , leafSize)
        
        return myTree
    
    def fit(self , dataSet):
        self.tree = self.buildTree(dataSet , self.depth , self.leafSize)
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
    import numpy as np
    
    boston = load_boston()
    dataSetMat = np.column_stack((boston.data , boston.target))
    dataSet = dataSetMat[:100 , (0,1,2,3,-1)].tolist()
    return dataSet

     
if __name__ == '__main__' :

    
    dataSet = loadData()
    
    dt = cartReg(5,1)
    dt.fit(dataSet)
    print dt.predict([0.00632, 18.0, 2.31, 0.0, -7.8799999999999955])
    
        