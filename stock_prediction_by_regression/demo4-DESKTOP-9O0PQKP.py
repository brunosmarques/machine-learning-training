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

#dates = []
#prices =[]

# def get_data_csv(filename):
#     with open(filename, 'r') as csvfile:
#         csvFileReader = csv.reader(csvfile)
#         next(csvFileReader)
#         for row in csvFileReader:
#             dates.append(int(row[0].split('-')[0]))
#             prices.append(float(row[1]))
#     return

def predict_prices(dates, prices, x):
    dates = np.reshape(dates, len(dates),1)

    svr_lin = SVR(kernel='linear',C=1e3)
    svr_pol = SVR(kernel='poly',C=1e3, degree = 2)
    svr_rbf = SVR(kernel='rbf',C=1e3, gamma=0.1)

    svr_lin.fit(dates, prices)
    svr_pol.fit(dates, prices)
    svr_rbf.fit(dates, prices)

    plt.scatter(dates, prices, color='black', label='Data')
    plt.plot(dates, svr_lin.predict(dates), color='red', label='Linear Model')
    plt.plot(dates, svr_pol.predict(dates), color='green', label='Poly Model')
    plt.plot(dates, svr_rbf.predict(dates), color='blue', label='RBF Model')

    plt.xlabel='Data'
    plt.ylabel='Price'
    plt.title='Support vector regression'

    plt.legend()
    plt.show()

    return svr_lin.predict(x)[0],  svr_pol.predict(x)[0],  svr_rbf.predict(x)[0]

def get_data_web(start=datetime.datetime(2010,1,1),end=datetime.date.today()):
    df = web.DataReader('AAPL', 'yahoo', start, end) # Getting AAPL stock from yahoo source
    return df

#main
mpl.rc('figure', figsize=(8,7))
mpl.__version__
style.use('ggplot')

data = get_data_web()
print(data)

outF = open("myOutFile.txt", "w")
for line in data:
  # write line to output file
  outF.write(line)
  outF.write("\n")
outF.close()
# get_data_csv('stock.csv')
# predict_price = predict_prices(dates, prices, 29)
# print(predict_price)



