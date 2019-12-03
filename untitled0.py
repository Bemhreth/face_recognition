# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 00:07:27 2018

@author: bemhret gezahegn
"""

import sklearn
from sklearn import tree
features=[[140,1],[130,1],[150,0],[170,0]]
lables=[0,0,1,1]
clf=tree.DecisionTreeClassifier()
clf=clf.fit(features,lables)
d=clf.predict([[110,1]])
print(d)