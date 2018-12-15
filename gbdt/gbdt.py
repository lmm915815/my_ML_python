# -*- coding: utf-8 -*-
"""
Created on Sun Dec  9 18:28:03 2018

@author:    tanliqing2010@163.com
csdn        https://blog.csdn.net/tanliqing2010
"""
import cart_regression as cart

import math


class GBDTBase(object):
    
    def __init__(self , nTrees , depth , leafSize):
        self.nTrees = nTrees
        self.depth = depth
        self.leafSize = leafSize
        self.allTrees  = []
        self.fValue = None
        self.lost = []
        
    # 步骤1：初始化f(x) 的初始值
    def initFValue(self , targets):
        self.fValue = [0 for i in targets]
    
    # 2. 计算负梯度
    def calaResidual(self,yReal , fValue):
        raise NotImplementedError
    # 3. 构造回归树
    # 4. 叶子节点最佳拟合值       
    def updateLeafValue(self , tree , dataSet , target):
        raise NotImplementedError
    # 更新f值
    def updateFValue(self ,dataSet ,tree):
        raise NotImplementedError
    def calaLost(self, target):
        raise NotImplementedError
    
    
    def fit(self , dataSet , target):
        # 1 初始化f0
        self.initFValue(target)
        
        for i  in range(self.nTrees):
            # 2 计算负梯度
            newTarget = self.calaResidual(target , self.fValue)
            print newTarget[0]
            # 3 构造回归树
            tree = cart.cartReg().buildTree(dataSet ,newTarget, self.depth , self.leafSize)
            # 4 拟合叶子节点
            self.updateLeafValue(tree , dataSet , newTarget)
            # 5 更新f值
            self.updateFValue(dataSet  , tree)
            
            self.allTrees.append(tree)
            self.lost.append(self.calaLost(target))
        return self
    
    def predict(self , predList):
        value = 0.0
        for tree in self.allTrees:
            value +=  cart.cartReg().predict0(predList, tree)
            #print value
        return value
    
class GBDTRegression(GBDTBase):
    def __init__(self , nTrees , depth , leafSize):
        # GBDTBase.__init__(self, nTrees , depth , leafSize)
        super(GBDTRegression,self).__init__(nTrees , depth , leafSize)
        
    def initFValue(self , target):
        super(GBDTRegression,self).initFValue(target)
        

    
    def calaResidual(self, yReal , fValue):
#        ret = [0 for f in yReal]
#        for i in range(len(yReal)):
#            ret[i] = yReal[i] - yPre[i]
        
        ret = [ i-y for i,y in zip(yReal,fValue) ]
        
        return ret
        
    def updateLeafValue(self ,tree, dataSet,target):
        pass
    
    def updateFValue(self,dataSet , tree):
        
        for i in range(len(self.fValue)):
            data = dataSet[i]
            self.fValue[i] += cart.cartReg().predict0(data , tree)
            
    def calaLost(self,target):
        err = 0.0
        for i in range(len(target)):
            err += (target[i] - self.fValue[i]) ** 2
        return err


class GBDTClissity(GBDTBase):
    def __init__(self, nTree, depth, leafSize):
        super(GBDTClissity, self).__init__(nTree, depth, leafSize)
    
    def initFValue(self, target):
        self.fValue = [t / (abs(t) * (2 - abs(t))) for t in target]

    def calaResidual(self, yReal, fValue):
        target = [2.0 * y / (1 + math.exp(2 * y * f)) for y, f in zip(yReal, fValue) ]
        return target
    
    def calaLeafValue(self , target):
        sum0 = 0.0
        for t in target:
            sum0 += abs(t) * (2 - abs(t))
            if sum0 == 0:
                pass
        return sum(target) / sum0
    
    def updateLeafValue(self, tree, dataSet, target):
        # 通过tree字典去切分数据
        
        spIndex = tree['spIndex']
        spValue = tree['spValue']
        subLeft, subRight, tarLeft, tarRight = cart.cartReg.spiltData(dataSet, target, spIndex, spValue)
        if not isinstance(tree['left'], dict):
            tree['left'] = self.calaLeafValue(tarLeft)
        else:
            self.updateLeafValue(tree['left'], subLeft, tarLeft)
        
        if not isinstance(tree['right'], dict):
            tree['right'] = self.calaLeafValue(tarRight)
        else:
            self.updateLeafValue(tree['right'], subRight, tarRight)
        
        
    def updateFValue(self, dataSet , tree):
        for i in range(len(self.fValue)):
            data = dataSet[i]
            self.fValue[i] += cart.cartReg().predict0(data , tree)
    
    def calaLost(self, target):
        err = 0.0
        for i in range(len(target)):
            # err += (target[i] - self.fValue[i]) ** 2
            err += math.log(1 + math.exp( -2 * target[i] * self.fValue[i]))
        return err
    
    def predict(self, predList, threshold = 0.5):
        p = self.predict_prob(predList)
        if p >= threshold:
            return 1
        return -1
    
    def predict_prob(self, predList):
        value = super(GBDTClissity, self).predict(predList)
        # print value
        p = 1 / (1 + math.exp(-value))
        # print p
        return p

def loadData():
    from sklearn.datasets import load_iris
    iris = load_iris()
    dataSet = iris.data[:100].tolist()
    target = iris.target[:100].tolist()
    def zero2(x):
        if x == 0:
            return -1
        return x
    target = map(zero2, target)
    
    return dataSet, target

def testClassify():
    dataSet , target = loadData()    
    classify = GBDTClissity(20 , 5 ,1)
    classify.fit(dataSet , target)
    value = classify.predict([4.9, 3. , 1.4, 0.2]) # -1
    print value 
    return classify

if __name__ == '__main__1':
    dataSet ,target= cart.loadData()
    
    gbdt = GBDTRegression(20 , 5, 1)
    gbdt.fit(dataSet , target)
    print gbdt.lost
    value = gbdt.predict([0.0686, 0.0, 2.89, 0.0]) # 33.2
    print value
    
    #print allTrees


if __name__ == '__main__':
    #regression()
    dataSet , target = loadData()    
    clf = GBDTClissity(20 , 5 ,1)
    clf.fit(dataSet , target)
    value = clf.predict([4.9, 3. , 1.4, 0.2]) # -1
    print value 
    value2 = clf.predict([5.7, 2.8, 4.1, 1.3]) # 1
    print value2