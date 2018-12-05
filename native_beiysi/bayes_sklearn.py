# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 10:49:00 2018

@author: Administrator
"""

from sklearn.naive_bayes import GaussianNB , MultinomialNB , BernoulliNB
'''
GaussianNB
     针对连续值
MultinomialNB
    离散值     可以多次出现
BernoulliNB
    离散值     要么存在要么不存在
'''

import numpy as np

import beiyesi as bys


if __name__ == "__main__":
    data , target = bys.loadData()
    worldList = bys.createWorldVec(data)
    
    trainMat = []
    for line in data:
        trainMat.append(bys.setOfWords2Vec(worldList , line))

    clf0 = GaussianNB()
    clf1 = MultinomialNB()
    clf2 = BernoulliNB()
    clfVec =[]
    clfVec.append(clf0)
    clfVec.append(clf1)
    clfVec.append(clf2)
    for clf in clfVec:
        clf.fit(np.array(trainMat) , np.array(target))
        
        testDoc = ['love' , 'my' , 'dalmation']
        testVec = np.array(bys.setOfWords2Vec(worldList , testDoc))
        ret = clf.predict(testVec.reshape((1,-1)))
        ret1 = clf.predict_proba(testVec.reshape((1,-1)))
        ret2 = clf.predict_log_proba(testVec.reshape((1,-1)))
        print ret , ret1 , ret2