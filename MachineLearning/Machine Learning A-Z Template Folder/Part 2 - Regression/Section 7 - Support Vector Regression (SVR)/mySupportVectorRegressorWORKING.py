# From Regression Template
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values # This will only take first column as upper bound of 2 is not considered
y = dataset.iloc[:, 2].values

'''
# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
'''
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = y.reshape(-1,1)
y = sc_y.fit_transform(y)


# Fitting the SVR to the dataset (also considering polyReg)
from sklearn.svm import SVR
regressor = SVR(kernel='rbf') # as we are doing a poly.regression. NOTE: rbf is default
regressor.fit(X, y)


# Predicting a new result 
# Since the feature is scaled here, you cannot directly put 6.5 here as follows
# y_pred = regressor.predict(6.5)
# use sc_X here
# transform function expects a numpy array
y_pred = regressor.predict(sc_X.transform(np.array([[6.5]]))) # No need to fit again
# the y_pred returned here is still scaled. Use inverse transform
y_pred = sc_y.inverse_transform(y_pred)

# Visualising the SVR results (for higher resolution and smoother curve)
# For higher resolution and smoother curve, do the next commented lines
#X_grid = np.arange(min(X), max(X), 0.1)
#X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X, regressor.predict(X) , color = 'blue')
# for smoother curve, do this instead of the above
# plt.plot(X, regressor.predict(X , color = 'blue')
plt.title('Truth or Bluff - SVR')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
