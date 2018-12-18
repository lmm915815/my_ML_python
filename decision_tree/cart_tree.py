# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 11:36:16 2018

@author: Administrator
"""

import decision_tree as dt

def calaGini(dataSet):
    label = [ fea[-1] for fea in dataSet ]
    feaDict = {}
    for fea in label:
        if fea not in feaDict:
            feaDict[fea] = 0
        feaDict[fea] += 1
    
    gini0 = 0
    for key in feaDict.keys():
        prob =  feaDict[key] / float(len(label))
        gini0 += prob * prob
    return 1 - gini0

def spiltDataSet(dataSet , col ,value):
    retSetLeft = []
    retSetRight= []
    for fea in dataSet:
        if fea[col] == value:
            feaVec = fea[:col]
            feaVec.extend(fea[col+1 : ])
            retSetLeft.append(feaVec)
        else:
            feaVec = fea[:col]
            feaVec.extend(fea[col+1:])
            retSetRight.append(feaVec)
    return retSetLeft , retSetRight


def chooseBestFeatureToSpilt(dataSet , imp = 0):
    '''
    '''
    numFeat = len(dataSet[0]) -1
    bestFeat  = -1
    minGini = 100000
    bestValue = 9999
    for col in range(numFeat):
        colList = [fea[col] for fea in dataSet ]
        feat = set(colList)  
        gini = 0
        for value in feat:
            subData1 , subData2 = spiltDataSet(dataSet , col , value)
            prob1 = len(subData1) / float(len(dataSet))
            prob2 = len(subData2) / float(len(dataSet))
            gini = prob1 * calaGini(subData1) + prob2*calaGini(subData2)
            if gini < minGini:
                minGini = gini
                bestFeat = col
                bestValue = value
    return bestFeat   , bestValue , gini
    
class Tree(object):
    def __init__(self , fea , value , isLeaf = False):
        self.left = None
        self.right = None
        self.isLeaf = isLeaf
        self.fea = fea
        self.value = value
    def myPrint(self):
        
        
        if  self.isLeaf:
            return {(self.fea , self.value):(None,None)}
    
        return {(self.fea, self.value):(self.left.myPrint() , self.right.myPrint())}
        
        
def createTree(dataSet , labels):
    '''
    '''
    classList = [fea[-1] for fea in dataSet]
    # 递归出口
    if classList.count(classList[0]) == len(classList):
        return Tree(None , classList[0]  , True)
    if len(dataSet[0]) == 1:
        return Tree(None ,dt.majorityCnt(classList) , True)
    fea , value , gini = chooseBestFeatureToSpilt(dataSet)
    feaLabel = labels[fea]
    myTree = Tree(feaLabel , value)
    dataLeft , dataRight = spiltDataSet(dataSet , fea , value)
    newLabels = labels[:fea]
    newLabels.extend(labels[fea+1:])
    myTree.left = createTree(dataLeft , newLabels)
    myTree.right = createTree(dataRight , newLabels)
    
    return myTree

def predict(myTree , labels ,preVec):
    
    if myTree.isLeaf :
        return myTree.value
    fea = myTree.fea
    value = myTree.value
    index = labels.index(fea)
    preValue = preVec[index]
    if value == preValue:
        return predict(myTree.left , labels , preVec)
    else:
        return predict(myTree.right , labels , preVec)
        

if __name__ == '__main__':
    import sys
    stdo = sys.stdout
    reload(sys)
    sys.setdefaultencoding('utf-8')
    sys.stdout= stdo
    
    dataSet , labels = dt.createDataSet1()
    gini = calaGini(dataSet)
    #feat , value = chooseBestFeatureToSpilt(dataSet)
    
    myTree = createTree(dataSet , labels)
    
    pre = predict(myTree , labels , [0,1,0,1])
    
        