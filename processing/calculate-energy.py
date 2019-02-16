# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 21:50:30 2019

@author: didi

"""
import json
import numpy as np
import pandas as pd
from scipy import signal
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

from statsmodels.tsa.stattools import adfuller
def test_stationarity(timeseries):
    ts_log = np.log(timeseries)
    rolmean = ts_log.rolling(window=12).mean()
    rolstd = ts_log.rolling(window=12).std()

    #Plot rolling statistics:
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    # plt.show()

    #Perform Dickey-Fuller test:
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)

# with open('../data/princeton.json', 'r') as f:
#     production_forecast_data = json.load(f)

# production_forecast = []
# production_forecast_times = []

# for d in production_forecast_data['forecasts']:
#     production_forecast.append(d['pv_estimate'])
#     production_forecast_times.append(d['period_end'])

# production_forecast_times = np.array(production_forecast_times)

# print("production_forecast:")
# print(production_forecast)
# print("production_forecast_times:")
# print(production_forecast_times)

## production_forecast_data.index = pd.to_datetime(production_forecast_data.index, unit='s')
## consumption = production_forecast_data.resample('1H')

# consumption_data = pd.read_csv('../data/buildings-2018-3.csv')
# consumption = consumption_data.iloc[:,1].tolist();
# consumption_times = np.array(consumption_data.iloc[:,0].tolist());

# dateparse = lambda dates: pd.datetime.strptime(dates, '%y-%m-%d %H:%M:%S')

consumption_file = '../data/buildings-2018-3.csv'
# dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
consumption_data = pd.read_csv(consumption_file, parse_dates=['Time (UTC)'], index_col='Time (UTC)')

print(consumption_data.head())
ts = consumption_data['Wilcox.kW']

print('\n\n1. original')
test_stationarity(ts)

# make timeseries more stationary
ts_log = np.log(ts)

print('\n\n2. moving_avg')
moving_avg = ts_log.rolling(window=12).mean()
ts_log_moving_avg_diff = ts_log - moving_avg
# print(ts_log_moving_avg_diff.head(12))
ts_log_moving_avg_diff.dropna(inplace=True)
# print(ts_log_moving_avg_diff.head(12))
test_stationarity(ts_log_moving_avg_diff)

print('\n\n3. expweighted_avg')
expweighted_avg = pd.DataFrame.ewm(ts_log, halflife=12).mean()
ts_log_ewma_diff = ts_log - expweighted_avg
test_stationarity(ts_log_ewma_diff)

print('\n\n4. diff')
ts_log_diff = ts_log - ts_log.shift()
# plt.plot(ts_log_diff)
# plt.show()
ts_log_diff.dropna(inplace=True)
test_stationarity(ts_log_diff)

print('\n\n5. decomposition')
from statsmodels.tsa.seasonal import seasonal_decompose
decomposition = seasonal_decompose(ts_log)

trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

plt.subplot(411)
plt.plot(ts_log, label='Original')
plt.legend(loc='best')
plt.subplot(412)
plt.plot(trend, label='Trend')
plt.legend(loc='best')
plt.subplot(413)
plt.plot(seasonal,label='Seasonality')
plt.legend(loc='best')
plt.subplot(414)
plt.plot(residual, label='Residuals')
plt.legend(loc='best')
plt.tight_layout()
# plt.show()

ts_log_decompose = residual
ts_log_decompose.dropna(inplace=True)
# print(ts_log_decompose)
test_stationarity(ts_log_decompose)


# ts_log = ts_log_diff

#ACF and PACF plots:
from statsmodels.tsa.stattools import acf, pacf
lag_acf = acf(ts_log_diff, nlags=20)
lag_pacf = pacf(ts_log_diff, nlags=20, method='ols')
#Plot ACF:
plt.subplot(121)
plt.plot(lag_acf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
plt.title('Autocorrelation Function')
#Plot PACF:
plt.subplot(122)
plt.plot(lag_pacf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
plt.title('Partial Autocorrelation Function')
plt.tight_layout()
plt.show()


from statsmodels.tsa.arima_model import ARIMA

model = ARIMA(ts_log, order=(2, 1, 0))
results_AR = model.fit(disp=-1)
plt.plot(ts_log_diff)
plt.plot(results_AR.fittedvalues, color='red')
plt.title('RSS: %.4f'% sum((results_AR.fittedvalues-ts_log_diff)**2))
plt.show()

model = ARIMA(ts_log, order=(0, 1, 2))
results_MA = model.fit(disp=-1)
plt.plot(ts_log_diff)
plt.plot(results_MA.fittedvalues, color='red')
plt.title('RSS: %.4f'% sum((results_MA.fittedvalues-ts_log_diff)**2))
plt.show()

model = ARIMA(ts_log, order=(2, 1, 2))
results_ARIMA = model.fit(disp=-1)
plt.plot(ts_log_diff)
plt.plot(results_ARIMA.fittedvalues, color='red')
plt.title('RSS: %.4f'% sum((results_ARIMA.fittedvalues-ts_log_diff)**2))
plt.show()

predictions_ARIMA_diff = pd.Series(results_ARIMA.fittedvalues, copy=True)
print(predictions_ARIMA_diff.head())

predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()
print(predictions_ARIMA_diff_cumsum.head())

predictions_ARIMA_log = pd.Series(ts_log.ix[0], index=ts_log.index)
predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum,fill_value=0)
predictions_ARIMA_log.head()

predictions_ARIMA = np.exp(predictions_ARIMA_log)
plt.plot(ts)
plt.plot(predictions_ARIMA)
plt.title('RMSE: %.4f'% np.sqrt(sum((predictions_ARIMA-ts)**2)/len(ts)))
plt.show()


# plt.plot(ts_log)
# plt.plot(moving_avg, color='red')
# plt.show()

# plt.plot(ts)
# plt.show()
# print(ts.head(10))
# print(ts)
# print(consumption_data.index)
# print(consumption_data.head())
# print('\n Data Types:')
# print(consumption_data.dtypes)
# print("consumption:")
# print(consumption)
# print("consumption_times:")
# print(consumption_times)

# energy = np.subtract(production_forecast, consumption_forecast)

# print("energy:")
# print(energy)


