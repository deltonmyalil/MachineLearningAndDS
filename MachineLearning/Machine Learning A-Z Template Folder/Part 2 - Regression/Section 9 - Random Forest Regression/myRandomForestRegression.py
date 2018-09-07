# Random Forest Regression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# No need to split into training and test sets

# No need for feature scaling

# Fitting the regression model to the dataset:
from sklearn.ensemble import RandomForestRegressor
# n_estimators is the number of individual decision trees in the forest
# criterion is set as mean square error
# You gotta play around with n_estimators until you get accurate predictions
regressor = RandomForestRegressor(n_estimators=300, criterion='mse', random_state=0)
regressor.fit(X, y)

# Predicting the result
y_pred = regressor.predict(6.5)

# Visualizing the regression results for higher resolution
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color='red')
plt.plot(X_grid, regressor.predict(X_grid), color='blue')
plt.title('Truth or Bluff - Random Forest Model')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()
