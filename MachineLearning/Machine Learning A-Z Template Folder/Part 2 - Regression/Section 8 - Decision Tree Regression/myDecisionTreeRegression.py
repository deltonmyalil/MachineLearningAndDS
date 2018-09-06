# Polynomial Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values # we only need the levels ie column1, column 2 is not included
y = dataset.iloc[:, 2].values # col2 is selected here as Y

# Splitting the dataset into the Training set and Test set
'''from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
''' # I dont think I wanna split a dataSet of 10 attributes
'''
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
''' # no need to scale data

# Fitting the Regression model to the dataset (also considering polyReg)
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(X, y)

# Predicting a new result 
y_pred = regressor.predict(6.5)


''' The below code is not possible as Decision tree reg is a non continuous non linear reg model
So I am rewriting this code for it
# Visualising the Decision Tree Regression results (for higher resolution and smoother curve)
# For higher resolution and smoother curve, do the next commented lines
#X_grid = np.arange(min(X), max(X), 0.1)
#X_grid = X_grid.reshape((len(X_grid), 1))e
plt.scatter(X, y, color = 'red')
plt.plot(X, regressor.predict(X) , color = 'blue')
# for smoother curve, do this instead of the above
# plt.plot(X, regressor.predict(X , color = 'blue')
plt.title('Truth or Bluff - Decision Tree Regression model')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
'''

# Visualising the Regression results (for higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 0.01) # 0.1 is the step size
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Regression Model)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()