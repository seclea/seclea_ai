# Import scikit-learn dataset library
from sklearn import datasets
from seclea_utils.data.transmission.resquests_wrapper import RequestWrapper

r = RequestWrapper('localhost:8000/')
import pickle
import os


class ModelManager:
    def __init__(self):
        self.tmp_path = './.seclea/'
        self.tmp_file='tmp'
        if not os.path.exists(self.tmp_path):
            os.makedirs(self.tmp_path)

        pass

    def _save_to_tmp(self, model_encoded):
        open(self.tmp_path+self.tmp_file, 'wb').write(model_encoded)


class SklearnManager(ModelManager):
    def __init__(self):
        super().__init__()
        pass

    def save_model(self, model):
        self._save_to_tmp(pickle.dumps(model))
        r.send_file(self.tmp_path+self.tmp_file)

    def load_model(self, model):
        return pickle.loads(model)


m = SklearnManager()

# Load dataset
iris = datasets.load_iris()
# print the label species(setosa, versicolor,virginica)
print(iris.target_names)

# print the names of the four features
print(iris.feature_names)

# print the iris data (top 5 records)
print(iris.data[0:5])

# print the iris labels (0:setosa, 1:versicolor, 2:virginica)
print(iris.target)

# Creating a DataFrame of given iris dataset.
import pandas as pd

data = pd.DataFrame({
    'sepal length': iris.data[:, 0],
    'sepal width': iris.data[:, 1],
    'petal length': iris.data[:, 2],
    'petal width': iris.data[:, 3],
    'species': iris.target
})
data.head()

# Import train_test_split function
from sklearn.model_selection import train_test_split

X = data[['sepal length', 'sepal width', 'petal length', 'petal width']]  # Features
y = data['species']  # Labels

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)  # 70% tra

# Import Random Forest Model
from sklearn.ensemble import RandomForestClassifier

# Create a Gaussian Classifier
clf = RandomForestClassifier(n_estimators=100)
# Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(X_train, y_train)
m.save_model(clf)

y_pred = clf.predict(X_test)

# Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics

# Model Accuracy, how often is the classifier correct?
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

clf.predict([[3, 5, 4, 2]])

from sklearn.ensemble import RandomForestClassifier

# Create a Gaussian Classifier
clf = RandomForestClassifier(n_estimators=100)

# Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(X_train, y_train)
import pandas as pd

feature_imp = pd.Series(clf.feature_importances_, index=iris.feature_names).sort_values(ascending=False)

import matplotlib.pyplot as plt
import seaborn as sns

# Creating a bar plot
sns.barplot(x=feature_imp, y=feature_imp.index)
# Add labels to your graph
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Visualizing Important Features")
plt.legend()
plt.show()

# Import train_test_split function
from sklearn.model_selection import train_test_split

# Split dataset into features and labels
X = data[['petal length', 'petal width', 'sepal length']]  # Removed feature "sepal length"
y = data['species']
# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.70, random_state=5)  # 70% training and 30% test

from sklearn.ensemble import RandomForestClassifier

# Create a Gaussian Classifier
clf = RandomForestClassifier(n_estimators=100)

# Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(X_train, y_train)

# prediction on test set
y_pred = clf.predict(X_test)

# Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics

# Model Accuracy, how often is the classifier correct?
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
