# -*- coding: utf-8 -*-
"""
Created on Mon Dec  3 15:53:48 2018

@author: Administrator
"""
from sklearn.linear_model.logistic import LogisticRegression
import logic_regression 
from sklearn.cross_validation import train_test_split
X , y =  logic_regression.loadData()
clss = LogisticRegression()
X_train , X_test , y_train , y_test = train_test_split(X , y , train_size = 0.7)
clss.fit(X_train,y_train)
print clss.coef_
print clss.intercept_

print clss.score(X_test , y_test)