#!usr/bin/env/python 
# -*- coding: utf-8 -*-

# 使用TPOT自动选择scikit-learn机器学习模型和参数
# http://blog.topspeedsnail.com/archives/10709

from sklearn import model_selection
from tpot import TPOTClassifier
import numpy as np
import pandas as pd
 
df = pd.read_csv('breast-cancer-wisconsin.data')
 
df.replace('?', np.nan, inplace=True)  # -99999
df.dropna(inplace=True)
df.drop(['id'], 1, inplace=True)
 
X = np.array(df.drop(['class'], 1))
Y = np.array(df['class'])
 
X_trian,X_test,Y_train,Y_test = model_selection.train_test_split(X, Y, test_size=0.2)
 
tpot = TPOTClassifier(generations=6, verbosity=2)
tpot.fit(X_trian, Y_train)
tpot.score(X_test, Y_test)
# 导出
tpot.export('pipeline.py')


'''
from sklearn import model_selection
from sklearn.ensemble import ExtraTreesClassifier
import numpy as np
import pandas as pd
 
df = pd.read_csv('breast-cancer-wisconsin.data')
 
df.replace('?', np.nan, inplace=True)  # -99999
df.dropna(inplace=True)
df.drop(['id'], 1, inplace=True)
 
X = np.array(df.drop(['class'], 1))
Y = np.array(df['class'])
 
X_trian,X_test,Y_train,Y_test = model_selection.train_test_split(X, Y, test_size=0.2)
 
clf = ExtraTreesClassifier(criterion="gini", max_features=0.3, n_estimators=500)
#clf = neighbors.KNeighborsClassifier()
clf.fit(X_trian, Y_train)
 
accuracy = clf.score(X_test, Y_test)
print(accuracy)
'''