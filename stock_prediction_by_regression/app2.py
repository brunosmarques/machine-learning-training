import csv
import datetime
with open('GOOG.csv', 'r') as f:
    reader = csv.reader(f)
    prices = list(reader)

prices = prices[1:]
print(datetime.datetime.strptime(prices[0][0]))