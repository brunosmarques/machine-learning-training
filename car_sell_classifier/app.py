# Pandas
import pandas as pd

# Others
from datetime import datetime

# Numpy
import numpy as np

# SKLearn
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

# Get data from web (used car sales)
uri = 'https://gist.githubusercontent.com/guilhermesilveira/4d1d4a16ccbf6ea4e0a64a38a24ec884/raw/afd05cb0c796d18f3f5a6537053ded308ba94bf7/car-prices.csv'
data = pd.read_csv(uri)

# Filter and normalize data
to_rename = {
    'yes' : 1,
    'no' : 0
}
data.sold = data.sold.map(to_rename)
current_year = datetime.today().year
data['model_age'] = current_year - data.model_year
data['km_per_year'] = data.mileage_per_year * 1.60934
data = data.drop(columns=['Unnamed: 0','mileage_per_year', 'model_year'], axis=1)

# Definne x and y
x = data[['price', 'km_per_year', 'model_age']]
y = data['sold']

# Configure numpy to get consistent results
SEED = 5
np.random.seed(SEED)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, stratify = y)
model = LinearSVC()
model.fit(x_train, y_train)
predictions = model.predict(x_test)

accuracy = accuracy_score(y_test, predictions) * 100
print("The accuracy is {0:.2f}%".format(accuracy))