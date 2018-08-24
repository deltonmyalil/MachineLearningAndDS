# My own Simple Linear Regression
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv') # Replace with filename
X = dataset.iloc[:, :-1].values # It's fine because most datasets will contain depVar in the last col
y = dataset.iloc[:, 1].values # Need to change with the index of last col

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0) # you may choose your own test size

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

# Preprocessing done

# Simple Linear Regression
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
# Fit the regressor object to the training set
regressor.fit(X_train, y_train)
# It learned the correlations between X_train and y_train

# Predicting the test set results
y_pred  = regressor.predict(X_test)