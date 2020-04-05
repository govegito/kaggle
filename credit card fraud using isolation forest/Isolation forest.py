# -*- coding: utf-8 -*-
"""
Created on Tue Dec 31 11:08:11 2019

@author: jayesh
"""


import cufflinks as cf
cf.go_offline()
import plotly.offline as py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv('data.csv')
print(df.columns)
print(df.info())
descdf=df.describe()
print(descdf)

df.hist(figsize = (20, 20))
plt.show()

fraud=df[df['Class']==1]
valid=df[df['Class']==0]

'''fig, ax = plt.subplots()

a_heights, a_bins = np.histogram(fraud['Amount'])
b_heights, b_bins = np.histogram(valid['Amount'], bins=a_bins)

width = (a_bins[1] - a_bins[0])/3

ax.bar(a_bins[:-1], a_heights, width=width, facecolor='cornflowerblue')
ax.bar(b_bins[:-1]+width, b_heights, width=width, facecolor='seagreen')'''

outlier_fraction = len(fraud)/float(len(valid))
print(outlier_fraction)


# Get all the columns from the dataFrame
columns = df.columns.tolist()

# Filter the columns to remove data we do not want
columns = [c for c in columns if c not in ["Class"]]

# Store the variable we'll be predicting on
target = "Class"

X = df[columns]
Y = df[target]

# Print shapes
print(X.shape)
print(Y.shape)

from sklearn.ensemble import IsolationForest
classifier=IsolationForest(n_estimators=100,max_samples=len(X),contamination=outlier_fraction,random_state=1)

classifier.fit(X)
y_pred = classifier.predict(X)

y_pred[y_pred == 1] = 0
y_pred[y_pred == -1] = 1

from sklearn.metrics import confusion_matrix
cf=confusion_matrix(Y,y_pred)
sns.heatmap(cf,annot=True)
plt.show()

from sklearn.metrics import classification_report, accuracy_score
print(classification_report(Y, y_pred))