import pandas as pd
import numpy as np
import datetime
import pandas_datareader.data as web
from pandas import Series, DataFrame
import quandl
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from matplotlib import style
import matplotlib as mpl

start = datetime.datetime(2015, 1, 1)
end = datetime.date.today()

# prices = web.DataReader("GOOG", 'yahoo', start, end)
prices = quandl.get("WIKI/AMZN", start_date='2018-01-01', end_date='2019-01-01')
prices = prices[['Adj. Close']]
print(prices.head())

# Days to predict
forecast_out = 30
prices['Prediction'] = prices[['Adj. Close']].shift(-forecast_out)
print(prices.tail())

X = np.array(prices.drop(['Prediction'],1))
X = X[:-forecast_out]
print(X)

y = np.array(prices['Prediction'])
y = y[:-forecast_out]
print(y)

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

svr_linear = SVR(kernel='linear', C=1e3)
svr_poly = SVR(kernel='poly', C=1e3, degree=7)
svr_rfb = SVR(kernel='rbf', C=1e3, gamma=0.1)

svr_linear.fit(x_train, y_train)
svr_poly.fit(x_train, y_train)
svr_rfb.fit(x_train, y_train)

linear_confidence = svr_linear.score(x_test, y_test)
print("svm confidence: ", linear_confidence)

x_forecast = np.array(prices.drop(['Prediction'],1))[-forecast_out:]
print(x_forecast)

lin_prediction = svr_linear.predict(x_forecast)
pol_prediction = svr_poly.predict(x_forecast)
rfb_prediction = svr_rfb.predict(x_forecast)

mpl.rc('figure', figsize=(8,7))
mpl.__version__
style.use('ggplot')

plt.xlabel='Data'
plt.ylabel='Price'
plt.title='Support vector regression'

plt.plot(lin_prediction, color='red', label='Linear Model')
plt.plot(pol_prediction, color='green', label='Poly Model')
plt.plot(rfb_prediction, color='blue', label='RFB Model')

plt.legend()
plt.show()

