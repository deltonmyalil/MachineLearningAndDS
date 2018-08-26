# Multiple Linear Regression
# 

# Data Preprocessing
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv') # Replace with filename

X = dataset.iloc[:, :-1].values # It's fine because most datasets will contain depVar in the last col
y = dataset.iloc[:, 4].values # Need to change with the index of last col

# State is categorical. We need to encode the third column.
# LabelEncoder changes text to numbers. OneHotENcoder changes numbers to 0/1 labels ie DUmmy Vars
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelEncoder_X = LabelEncoder()
X[:, 3] = labelEncoder_X.fit_transform(X[:, 3])
oneHotEncoder = OneHotEncoder(categorical_features=[3])
X = oneHotEncoder.fit_transform(X).toarray()

# Avoiding the Dummy Var Trap. NOTE: Python libs automaticallly takes care of this, but still...
# Take all the cols of X starting from index 1
X = X[:, 1:] # Removing the zeroth column from X

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0) # you may choose your own test size

# Feature Scaling NO NEED, the lib will take care of this in case of linregression
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

# Fitting multiple linear regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Prediction of the test set results
y_pred = regressor.predict(X_test)