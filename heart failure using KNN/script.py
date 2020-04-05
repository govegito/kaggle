# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 13:38:13 2019

@author: jayesh
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv('data.csv')

'''sns.violinplot(df['target'], df['age'])
plt.show()'''

'''sns.pairplot(df,hue='target')
plt.shoe()'''

'''corr=df.corr()
plt.subplots(figsize=(11.7,11.7))
sns.heatmap(corr,cmap="YlGnBu",annot=True)
plt.show()'''

X=df.iloc[:,:-1].values
Y=df.iloc[:,-1].values.reshape([303,1])

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2, random_state=1)

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)


'''from LOGISTIC REGRESSION

from sklearn.linear_model import LogisticRegression
logclassifier=LogisticRegression()
logclassifier.fit(X_train,Y_train)

logY_pred=logclassifier.predict(X_test)
 
from sklearn.metrics import confusion_matrix
logcm=confusion_matrix(logY_pred,Y_test)
 
p1=sns.heatmap(logcm,cmap="YlGnBu",annot=True)

from sklearn.metrics import accuracy_score, precision_score
acclog=accuracy_score(Y_test,logY_pred)
prclog=precision_score(Y_test,logY_pred)'''

'''FROM KNN BEST METHODDD'''

from sklearn.neighbors import KNeighborsClassifier
kclassifier=KNeighborsClassifier(n_neighbors=9)
kclassifier.fit(X_train,Y_train)

kY_pred=kclassifier.predict(X_test)
 
from sklearn.metrics import confusion_matrix
kcm=confusion_matrix(kY_pred,Y_test)
 
p2=sns.heatmap(kcm,cmap="YlGnBu",annot=True)


from sklearn.metrics import accuracy_score, precision_score
acck=accuracy_score(Y_test,kY_pred)
prck=precision_score(Y_test,kY_pred)

from sklearn.model_selection import cross_val_score
accuracies=cross_val_score(estimator=kclassifier,X=X_train,y=Y_train, cv=10)
ACC=accuracies.mean()


from sklearn.model_selection import GridSearchCV
params=[{'n_neighbors':[1,2,3,4,5,6,7,8,9], 'weights':['uniform','distance']}]
grid_src=GridSearchCV(estimator=kclassifier,param_grid=params,scoring='accuracy',cv=10,n_jobs=-1)
grid_src=grid_src.fit(X_train,Y_train)
bestacc=grid_src.best_score_




'''from SVM

from sklearn.svm import SVC
svmclassifier = SVC(kernel = 'rbf', random_state = 0)
svmclassifier.fit(X_train, Y_train)

svmY_pred=svmclassifier.predict(X_test)

from sklearn.metrics import confusion_matrix
svmcm=confusion_matrix(svmY_pred,Y_test)
 
p2=sns.heatmap(svmcm,cmap="YlGnBu",annot=True)


from sklearn.metrics import accuracy_score, precision_score
accsvm=accuracy_score(Y_test,svmY_pred)
prcsvm=precision_score(Y_test,svmY_pred)'''

'''FROM RANDOM FOREST

from sklearn.ensemble import RandomForestClassifier
rfclassifier=RandomForestClassifier(n_estimators = 5)
rfclassifier.fit(X_train,Y_train)

rfY_pred=svmclassifier.predict(X_test)

from sklearn.metrics import confusion_matrix
rfcm=confusion_matrix(rfY_pred,Y_test)
 
p2=sns.heatmap(rfcm,cmap="YlGnBu",annot=True)


from sklearn.metrics import accuracy_score, precision_score
accrf=accuracy_score(Y_test,rfY_pred)
prcrf=precision_score(Y_test,rfY_pred)'''