# Pandas
import pandas as pd

# Others
from datetime import datetime
import graphviz
# Numpy
import numpy as np

# SKLearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz

# Get data from web (used car sales)
uri = 'https://gist.githubusercontent.com/guilhermesilveira/4d1d4a16ccbf6ea4e0a64a38a24ec884/raw/afd05cb0c796d18f3f5a6537053ded308ba94bf7/car-prices.csv'
uri = 'car_prices.csv' # local file if you don't have internet access
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

# Configure numpy to get consistent resultcs
SEED = 5
np.random.seed(SEED)

raw_x_train, raw_x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, stratify = y)
# scaler = StandardScaler()
# scaler.fit(raw_x_train)
# x_train = scaler.transform(raw_x_train)
# x_test = scaler.transform(raw_x_test)
x_train = raw_x_train
x_test = raw_x_test

model = DecisionTreeClassifier(max_depth=2)
model.fit(x_train, y_train)

# predictions = model.predict(x_test)
accuracy = model.score(x_test, y_test) * 100
print("The model accuracy is {0:.2f}%".format(accuracy))

dummy = DummyClassifier()
dummy.fit(x_train, y_train)
accuracy = dummy.score(x_test, y_test) * 100
print("The dummy accuracy is {0:.2f}%".format(accuracy))

features = x.columns
dot_data = export_graphviz(model, out_file='tree.plt',
                           filled = True, rounded = True,
                           feature_names = features,
                          class_names = ["no", "yes"])
graph = graphviz.Source(dot_data)
graph
