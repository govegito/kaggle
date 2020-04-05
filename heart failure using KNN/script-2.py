# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 16:15:48 2019

@author: jayesh
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data=pd.read_csv('data.csv')

# let's change the names of the  columns for better understanding

data.columns = ['age', 'sex', 'chest_pain_type', 'resting_blood_pressure', 'cholesterol', 'fasting_blood_sugar', 'rest_ecg', 'max_heart_rate_achieved',
       'exercise_induced_angina', 'st_depression', 'st_slope', 'num_major_vessels', 'thalassemia', 'target']
data['sex'][data['sex'] == 0] = 'female'
data['sex'][data['sex'] == 1] = 'male'

data['chest_pain_type'][data['chest_pain_type'] == 1] = 'typical angina'
data['chest_pain_type'][data['chest_pain_type'] == 2] = 'atypical angina'
data['chest_pain_type'][data['chest_pain_type'] == 3] = 'non-anginal pain'
data['chest_pain_type'][data['chest_pain_type'] == 4] = 'asymptomatic'

data['fasting_blood_sugar'][data['fasting_blood_sugar'] == 0] = 'lower than 120mg/ml'
data['fasting_blood_sugar'][data['fasting_blood_sugar'] == 1] = 'greater than 120mg/ml'

data['rest_ecg'][data['rest_ecg'] == 0] = 'normal'
data['rest_ecg'][data['rest_ecg'] == 1] = 'ST-T wave abnormality'
data['rest_ecg'][data['rest_ecg'] == 2] = 'left ventricular hypertrophy'

data['exercise_induced_angina'][data['exercise_induced_angina'] == 0] = 'no'
data['exercise_induced_angina'][data['exercise_induced_angina'] == 1] = 'yes'

data['st_slope'][data['st_slope'] == 1] = 'upsloping'
data['st_slope'][data['st_slope'] == 2] = 'flat'
data['st_slope'][data['st_slope'] == 3] = 'downsloping'

data['thalassemia'][data['thalassemia'] == 1] = 'normal'
data['thalassemia'][data['thalassemia'] == 2] = 'fixed defect'
data['thalassemia'][data['thalassemia'] == 3] = 'reversable defect'

data['sex'] = data['sex'].astype('object')
data['chest_pain_type'] = data['chest_pain_type'].astype('object')
data['fasting_blood_sugar'] = data['fasting_blood_sugar'].astype('object')
data['rest_ecg'] = data['rest_ecg'].astype('object')
data['exercise_induced_angina'] = data['exercise_induced_angina'].astype('object')
data['st_slope'] = data['st_slope'].astype('object')
data['thalassemia'] = data['thalassemia'].astype('object')



Y=data.iloc[:,-1].values.reshape([303,1])

print("Shape of y:", Y.shape)

data = pd.get_dummies(data, drop_first=True)
X=data


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

