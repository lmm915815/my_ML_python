# -*- coding: utf-8 -*-
"""
Created on Mon Dec  3 11:10:26 2018

@author: Administrator
"""

import sklearn as sklearn

import sklearn.datasets as datasets
from  sklearn.metrics import r2_score , mean_squared_error
boston = datasets.load_boston()
X_train , X_test , y_train , y_test = sklearn.model_selection.train_test_split(
            boston.data , boston.target , test_size =0.3 ,random_state=0)

lr = sklearn.linear_model.LinearRegression()
lr.fit(X_train , y_train)
y_pred = lr.predict(X_test)
print lr.coef_
print lr.intercept_


score = r2_score(y_test , y_pred)
mean = mean_squared_error(y_test , y_pred)

