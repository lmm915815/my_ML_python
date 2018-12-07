# -*- coding: utf-8 -*-
"""
Created on Fri Dec  7 15:56:27 2018

@author:    tanliqing2010@163.com
csdn        https://blog.csdn.net/tanliqing2010
"""

from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

iris = load_iris()
clf1 = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1) , n_estimators=100)
clf = AdaBoostClassifier( n_estimators=100 , learning_rate=0.8)
clf.fit(iris.data , iris.target)
scores = cross_val_score(clf , iris.data , iris.target)
print scores.mean()