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

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
'''

# Fitting the SVR to the dataset (also considering polyReg)
from sklearn.svm import SVR
regressor = SVR(kernel='rbf') # as we are doing a poly.regression. NOTE: rbf is default
regressor.fit(X, y)


# Predicting a new result 
y_pred = regressor.predict(6.5)

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
