import csv
import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from matplotlib import style
import matplotlib as mpl
import pandas as pd
import datetime
import pandas_datareader.data as web #datasource
from pandas import Series, DataFrame

# def get_data_csv(filename):
#     with open(filename, 'r') as csvfile:
#         csvFileReader = csv.reader(csvfile)
#         next(csvFileReader)
#         for row in csvFileReader:
#             dates.append(int(row[0].split('-')[0]))
#             prices.append(float(row[1]))
#     return

def predict_prices(dates2, prices, x):
    dates3 = np.array(dates2)
    dates3 = dates3.reshape(-1,1)
    # print(dates3)
    
    svr_lin = SVR(kernel='linear',C=1e3)
    # svr_pol = SVR(kernel='poly',C=1e3, degree = 2)
    # svr_rbf = SVR(kernel='rbf',C=1e3, gamma=0.1)

    svr_lin.fit(dates3, prices)
    # svr_pol.fit(dates3, prices)
    # svr_rbf.fit(dates3, prices)

    # Matplotlib configuration
    # mpl.rc('figure', figsize=(8,7))
    # mpl.__version__
    # style.use('ggplot')
    
    plt.scatter(dates3, prices, color='black', label='Data')
    plt.plot(dates3, svr_lin.predict(dates3), color='red', label='Linear Model')
    # plt.plot(dates3, svr_pol.predict(dates3), color='green', label='Poly Model')
    # plt.plot(dates3, svr_rbf.predict(dates3), color='blue', label='RBF Model')

    plt.xlabel='Data'
    plt.ylabel='Price'
    plt.title='Support vector regression'

    plt.legend()
    plt.show()

    #return svr_lin.predict(x)[0],  svr_pol.predict(x)[0],  svr_rbf.predict(x)[0]

def get_data_web(start=datetime.datetime(2010,1,1), end=datetime.date.today(), stock='AAPL', source='yahoo'):
    df = web.DataReader('AAPL', 'yahoo', start, end) # Getting AAPL stock from yahoo source
    return df

# main
# Variables
stock = 'AAPL'

# Get stock data from web
data = get_data_web(start=datetime.datetime(2019,6,1), stock=stock)
# print(data)
prices = data['Adj Close']
dates = data.index
# print(prices, dates)

# Train model and return predictions
# predict_price = predict_prices(dates, prices, 29)
predict_prices(dates, prices, 29)

# Adding plots
# prices.plot(label=stock)
# predict_price[0].plot(label='m1')
# predict_price[1].plot(label='m2')
# predict_price[2].plot(label='m3')
# plt.legend()
# plt.show()

# get_data_csv('stock.csv')
# predict_price = predict_prices(dates, prices, 29)
# print(predict_price)