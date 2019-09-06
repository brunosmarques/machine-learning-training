import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.preprocessing import StandardScaler

uri = 'https://gist.githubusercontent.com/guilhermesilveira/1b7d5475863c15f484ac495bd70975cf/raw/16aff7a0aee67e7c100a2a48b676a2d2d142f646/projects.csv'
data = pd.read_csv(uri)
print(data)

to_map = {
    0 : 1,
    1 : 0
}
data['finished'] = data.unfinished.map(to_map)

# Defining seed
SEED = 5
np.random.seed = SEED

# X are the features, Y is the target
x = data[['expected_hours', 'price']]
y = data['finished']

# Separating dataset into train/test for x and y
raw_x_train, raw_x_test, y_train, y_test = train_test_split(x, y, random_state=SEED, test_size=0.25, stratify=y)

# Defining the model
scaler = StandardScaler()
scaler.fit(raw_x_train)
x_train = scaler.transform(raw_x_train)
x_test = scaler.transform(raw_x_test)
model = SVC()
# model = LinearSVC()

# Training the model
model.fit(x_train, y_train)

# Making predictions with model created
predictions = model.predict(x_test)

# Accuracy test
accuracy = metrics.accuracy_score(y_test, predictions)
print('{0:.2f}% of accuracy'.format(accuracy*100))

# Boundary
x_data = x_test[:,0]
y_data = x_test[:,1]

x_min = x_data.min()
x_max = x_data.max()
y_min = y_data.min()
y_max = y_data.max()

pixels = 100
x_axis = np.arange(x_min, x_max, (x_max-x_min)/pixels)
y_axis = np.arange(y_min, y_max, (y_max-y_min)/pixels)

xx, yy = np.meshgrid(x_axis, y_axis)
points = np.c_[xx.ravel(), yy.ravel()]
Z = model.predict(points)
Z = Z.reshape(xx.shape)

#sns.scatterplot(x = "expected_hours", y="price", data=data, hue="finished")
plt.contourf(xx, yy, Z, alpha=0.3)
plt.scatter(x_data, y_data, c = y_test, s=1)

plt.show()