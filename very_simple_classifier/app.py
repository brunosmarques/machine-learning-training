import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn import metrics

# Defining seed
SEED = 20

# Dataset URI
uri = 'https://gist.githubusercontent.com/guilhermesilveira/2d2efa37d66b6c84a722ea627a897ced/raw/10968b997d885cbded1c92938c7a9912ba41c615/tracking.csv'

# Using pandas to read and filter data from URI
data = pd.read_csv(uri)
print(data)

# X are the features, Y is the target
x = data[['home', 'how_it_works', 'contact']]
y = data['bought']

# Separating dataset into train/test for x and y
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=SEED, test_size=0.25, stratify=y)

# Defining the model
model = LinearSVC()

# Training the model
model.fit(x_train, y_train)

# Making predictions with model created
predictions = model.predict(x_test)

# Accuracy test
accuracy = metrics.accuracy_score(y_test, predictions)
print('{}% of accuracy'.format(accuracy*100))