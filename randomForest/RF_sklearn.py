# -*- coding: utf-8 -*-
"""
Created on Sat Dec  8 19:23:24 2018

@author:    tanliqing2010@163.com
csdn        https://blog.csdn.net/tanliqing2010
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification


X , y = make_classification(n_samples=1000 , n_features=4,
                            n_informative=2 , n_redundant=0,
                            random_state=0 , shuffle=False )

clf = RandomForestClassifier(n_estimators=100 , max_depth=2 ,
                             random_state=0, oob_score=True)

clf.fit(X, y)
# 这里是输出每个特征重要比例   ，这里从侧面也可以理解生成的数据
print clf.feature_importances_
print clf.predict_proba([[0,0,0,0]])

oob = clf.oob_score_
print clf.oob_decision_function_
