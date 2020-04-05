# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 14:55:43 2019

@author: jayesh
"""


import cufflinks as cf
cf.go_offline()
import plotly.offline as py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data=pd.read_csv('data.csv', index_col=0)
print(data.head(5))

data.index=pd.to_datetime(data.index)
data.columns=['energy prod']

#data.plot()


'''the function to test stattionarity'''
def test_stat(data):
    #rolling mean
    rolmean=data.rolling(window=12).mean()
    
    #rolling variance
    rolstd=data.rolling(window=12).std()
    
    grig=plt.plot(data,color='blue',label='original')
    mean=plt.plot(rolmean,color='red',label='rolling-mean')
    std=plt.plot(rolstd,color='black',label='rolling-std')
    plt.legend(loc='best')
    plt.title('rolling mean and std')
    plt.show(block=False)
    
    # AD FULLER TEST
    
    from statsmodels.tsa.stattools import adfuller
    dftest=adfuller(data['energy prod'],autolag='AIC')
    dfoutput=pd.Series(dftest[0:4],index=['T-stat','p-val','# lag used','# of observ used'])
    for key,val in dftest[4].items():
        dfoutput['critical val (%s)'%key]=val
    print(dfoutput)


test_stat(data)

# WITH LOG SCALE
data_log=np.log(data)
#data_log.plot()

test_stat(data_log)

#subtracting the moving mean from log data
rolmean_log=data_log.rolling(window=12).mean()
data_logminusavg=data_log-rolmean_log
data_logminusavg.dropna(inplace=True)
#data_logminusavg.plot()

test_stat(data_logminusavg)

#Weighted average of time series

data_log_w_mean=data_log.ewm(halflife=12,min_periods=0, adjust=True).mean()
plt.plot(data_log)
plt.plot(data_log_w_mean,color='red')

# subtracring weighted mean from log scale
data_log_minus_w_mean=data_log-data_log_w_mean

test_stat(data_log_minus_w_mean)

#  shifting the data_log by 1 unit
data_log_shifted=data_log-data_log.shift()
data_log_shifted.dropna(inplace=True)
test_stat(data_log_shifted)

from statsmodels.tsa.seasonal import seasonal_decompose
result=seasonal_decompose(data_log,model='multiplicative')
#result.plot()

# residual test for stationairty
decompossed_residual=result.resid
decompossed_residual.dropna(inplace=True)
test_stat(decompossed_residual)


'''plotting acf and pacf'''
from statsmodels.tsa.stattools import acf, pacf

lag_acf=acf(data_log_shifted,nlags=20)
lag_pacf=pacf(data_log_shifted, nlags=20, method='ols')

plt.subplot(121)
plt.plot(lag_acf)
plt.axhline(y=0,linestyle='--',color='grey')
plt.axhline(y=1.96/np.sqrt(len(data_log_shifted)),linestyle='--',color='grey')
plt.axhline(y=-1.96/np.sqrt(len(data_log_shifted)),linestyle='--',color='grey')
plt.title("Autocorrelation func")

plt.subplot(122)
plt.plot(lag_pacf)
plt.axhline(y=0,linestyle='--',color='grey')
plt.axhline(y=1.96/np.sqrt(len(data_log_shifted)),linestyle='--',color='grey')
plt.axhline(y=-1.96/np.sqrt(len(data_log_shifted)),linestyle='--',color='grey')
plt.title(" partial Autocorrelation func")
plt.tight_layout()

from statsmodels.tsa.arima_model import ARIMA
model=ARIMA(data_log,order=(3,1,2))
result_ar=model.fit(disp=-1)
plt.plot(data_log_shifted)
plt.plot(result_ar.fittedvalues, color='red')

result_ar.plot_predict(1,264)

future_forecast_1year = result_ar.forecast(13)[0]