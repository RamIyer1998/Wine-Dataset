import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor

from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.externals import joblib

#Loading in the dataset
dataset_url = 'http://mlr.cs.umass.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
data=pd.read_csv(dataset_url, sep=';')

#Getting a feel for the data
print("---Data Logistics---")
print(data.head)
print("")
print(data.shape)
print("")
print(data.describe())
print("--------------------\n")

#Isolating the target characteristic, and separating the data into the testing and training categories
y = data.quality
X = data.drop('quality', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state = 123, stratify = y)

#Preprocessing steps, and tests to make sure it worked properly
scaler = preprocessing.StandardScaler().fit(X_train)

X_train_scaled = scaler.transform(X_train)

print("--Transform Test----")
print(X_train_scaled.mean(axis=0))
print("")
print(X_train_scaled.std(axis=0))
print("")

X_test_scaled = scaler.transform(X_test) 

print(X_test_scaled.mean(axis=0))
print("")
print(X_test_scaled.std(axis=0))
print("--------------------\n")

#Declaring Hyperparameters to tune through cross-validation
pipeline = make_pipeline(preprocessing.StandardScaler(), RandomForestRegressor(n_estimators=100))

print("--Hyperparameters--")
print(pipeline.get_params())
print("-------------------\n")

hyperparameters = { 'randomforestregressor__max_features' : ['auto', 'sqrt', 'log2'],
                  'randomforestregressor__max_depth': [None, 5, 3, 1]}


#Perform Cross Validation 
clf = GridSearchCV(pipeline, hyperparameters, cv=10)
 
# Fit and tune model
clf.fit(X_train, y_train)

print("--Best Parameters--")
print(clf.best_params_)
print("-------------------\n")

#Running tests on the model, and printing results
y_pred = clf.predict(X_test)

print("------Results------")
print(r2_score(y_test, y_pred))
print("")
print(mean_squared_error(y_test, y_pred))
print("-------------------\n")

#Dumping the model to be used in comparisons later on
joblib.dump(clf, 'rf_regressor.pkl')
