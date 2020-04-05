# -*- coding: utf-8 -*-
"""
Created on Sat Dec 21 22:12:37 2019

@author: jayesh
"""
from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv('data.csv')
df=df.drop('id',axis=1)
df=df.drop('room_id/id', axis=1)

temp_in=df[df['out/in']=='In']
temp_out=df[df['out/in']=='Out']

temp_in.drop_duplicates('noted_date', keep=False, inplace=True)
temp_out.drop_duplicates('noted_date', keep=False, inplace=True)

temp_in.drop('out/in', axis=1,inplace=True)
temp_out.drop('out/in', axis=1,inplace=True)

temp_in.columns=['date','temp_in']
temp_out.columns=['date','temp_out']

master=pd.merge(temp_in,temp_out,on='date')

info=master.describe()


f,axes=plt.subplots(1,2)
sns.violinplot(master['temp_in'],color='#abffee',ax=axes[0])
sns.violinplot(master['temp_out'],color='#ffafee',ax=axes[1])
plt.show()

sns.distplot(master['temp_out'],bins=30)
plt.show()


plt.scatter(master['temp_in'], master['temp_out'])
plt.show()

def split_the_date(df):
    
    
    date,time=df['date'].split(' ')
    day,month,year=date.split('-')
        
    if month=='12':
        month='dec'
    elif month=='11':
        month='nov'
    elif month=='10':
        month='oct'
    elif month=='09':
        month='sep'
    elif month=='08':
        month='aug'
    elif month=='07':
        month='jul'
    
    df['month']=month
    df['day']=int(day)
    df['time']=int(''.join([str(elem) for elem in time.split(':')]))
    
    return df
    
master=master.apply(split_the_date, axis=1)
temp_in=temp_in.apply(split_the_date, axis=1)
temp_out=temp_out.apply(split_the_date, axis=1)


master.drop('date', axis=1, inplace=True)
'''from sklearn.preprocessing import LabelEncoder, OneHotEncoder
onehotencoder = OneHotEncoder(categories=3)
onehotencoder.fit_transform(master) '''
 
data = pd.get_dummies(master, drop_first=True)   
    
    
    


