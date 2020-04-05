# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 11:18:03 2019

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

from statsmodels.tsa.seasonal import seasonal_decompose
result=seasonal_decompose(data,model='multiplicative')

#fig=result.plot()

from pmdarima import auto_arima

stepwise_model = auto_arima(data, start_p=1, start_q=1,
                           max_p=3, max_q=3, m=12,
                           start_P=0, seasonal=True,
                           d=1, D=1, trace=True,
                           error_action='ignore',  
                           suppress_warnings=True, 
                           stepwise=True)
print(stepwise_model.aic())

# For the Test: we'll need to chop off a portion of our latest data, say from 2016, Jan.
test = data.loc['2016-01-01':]

# Fore the Train: we'll train on the rest of the data after split the test portion
train = data.loc['1939-01-01':'2015-12-01']

stepwise_model.fit(train)


#From 2016-01-01 to 2019-05-01 (*the latest update from the source*), we have 41 rows:
future_forecast = stepwise_model.predict(n_periods=41)
print(future_forecast)

future_forecast = pd.DataFrame(future_forecast, index=test.index, columns=['Prediction'])

pd.concat([test, future_forecast], axis=1).plot()

pd.concat([data,future_forecast],axis=1).plot()

stepwise_model.fit(data)

future_forecast_1year = stepwise_model.predict(n_periods=13)

next_year = [pd.to_datetime('2019-05-01'),
            pd.to_datetime('2019-06-01'),
            pd.to_datetime('2019-07-01'),
            pd.to_datetime('2019-08-01'),
            pd.to_datetime('2019-09-01'),
            pd.to_datetime('2019-10-01'),
            pd.to_datetime('2019-11-01'),
            pd.to_datetime('2019-12-01'),
            pd.to_datetime('2020-01-01'),
            pd.to_datetime('2020-02-01'),
            pd.to_datetime('2020-03-01'),
            pd.to_datetime('2020-04-01'),
            pd.to_datetime('2020-05-01')]

future_forecast_1year = pd.DataFrame(future_forecast_1year, index=next_year, columns=['Prediction'])

pd.concat([data,future_forecast_1year],axis=1).plot()