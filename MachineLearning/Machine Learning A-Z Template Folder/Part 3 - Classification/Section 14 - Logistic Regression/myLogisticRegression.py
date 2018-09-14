## My Logistic Regression

# importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# importing the dataset
dataset = pd.read_csv("Social_Network_Ads.csv")
X = dataset.iloc[:, [2,3]].values # only considering age and estimated salary
y = dataset.iloc[:,4].values

# data partitioning
from sklearn.model_selection import train_test_split # sklearn.cross_validation deprecated, now use sklearn.model_selection
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test) # crap, I forgot to scale the test set

# fitting logistic regression to the training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)

# predicting the test set results
y_pred = classifier.predict(X_test)
