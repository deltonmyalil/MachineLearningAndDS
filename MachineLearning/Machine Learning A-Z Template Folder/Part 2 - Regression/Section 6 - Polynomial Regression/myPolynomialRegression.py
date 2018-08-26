# Polynomial Regression 

# Data Preprocessing
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv') # Replace with filename
# Position dependent on Level: Multicollinearity
# Take only the level

# X = dataset.iloc[:, 1].values # Only 1st column is needed
# X should not be considered as a vector as it has only one dimension
# Therefore do this
X = dataset.iloc[:, 1:2].values # Now, this is taken as a column matrix
y = dataset.iloc[:, 2].values # 2nd column is the output variable

# No categorical variables, Therefore, no need of LabelEncoder and OneHotEncoder

# No need of splitting the dataset into the Training set and Test set as
# the number of observations is very less

# No need of feature scaling

# Preprocessing complete

# Creating Simple Linear Reg model
from sklearn.linear_model import LinearRegression
linReg = LinearRegression()
linReg.fit(X, y)

# Creatinig polynomial regression model with degree 2
from sklearn.preprocessing import PolynomialFeatures
polyReg = PolynomialFeatures(degree=2)
X_poly = polyReg.fit_transform(X) # We got a new matrix that contains a new set of inputVars that is the squares of the first ones
linReg2 = LinearRegression()
linReg2.fit(X_poly, y)

# Creatinig polynomial regression model with degree 3
from sklearn.preprocessing import PolynomialFeatures
polyReg2 = PolynomialFeatures(degree=3)
X_poly2 = polyReg2.fit_transform(X) # We got a new matrix that contains a new set of inputVars that is the squares of the first ones
linReg3 = LinearRegression()
linReg3.fit(X_poly2, y)

# Creatinig polynomial regression model with degree 4
from sklearn.preprocessing import PolynomialFeatures
polyReg3 = PolynomialFeatures(degree=4)
X_poly3 = polyReg3.fit_transform(X) # We got a new matrix that contains a new set of inputVars that is the squares of the first ones
linReg4 = LinearRegression()
linReg4.fit(X_poly2, y)



# Visualizing the results
# Visualizing Real data in Red
plt.scatter(X,y, color= "red")
# Visualizing LinReg in blue
plt.plot(X, linReg.predict(X), color= 'blue')
# Visualizing PolyReg in green
plt.plot(X, linReg2.predict(polyReg.fit_transform(X)), color= 'green')
# Visualizing polyReg2 in black
plt.plot(X, linReg3.predict(polyReg2.fit_transform(X)), color= 'black')
# Visualizing polyReg3 in yellow
plt.plot(X, linReg4.predict(polyReg3.fit_transform(X)), color= 'yellow')

plt.title("Truth or bluff")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()
# Degree 4 is more accurate

# Predicting with linear model
linReg.predict(6.5)  # Executing this results in Out[16]: array([330378.78787879])

# Predicting with PolyModel with degree 2
linReg2.predict(polyReg.fit_transform(6.5)) # output: array([189498.10606061])