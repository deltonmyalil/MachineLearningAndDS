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

# Building optimal model using backward elimination
import statsmodels.formula.api as sm
# Append an array of 1s to the 0th col of the matrix for BackElim to work
# the reason is y = b0 + b1x1 + b2x2 + ..., b0 needs a variable as well, else it will be discarded I guess
X = np.append(arr= np.ones((50,1)).astype(int), values = X, axis=1)
# Now preprocessing for backwardElim is complete

# Doing BackwardElim
# Here, let us do BackwardElim with for p > 0.05 ie p > 5%
X_opt = X[:,[0,1,2,3,4,5]] # As in the algo, fit all input vars first
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()  # OrdinaryLeastSquares
regressor_OLS.summary()
# In the summary, we see that highest p value is for col 2 ie 99% (NOTE: Col start from 0)
# In the next iteration, remove that inputVar ie col2
# ie Copy paste the above three lines without col 2
X_opt = X[:,[0,1,3,4,5]] # As in the algo, remove that, fit rest
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()  # OrdinaryLeastSquares
regressor_OLS.summary()
# In the summary, we see that highest p value is for col 1 ie inputVar1 ie 94%
# In the next iteration, remove that inputVar ie col1 ie inputVar1
# ie Copy paste the above three lines without col 1 ie inputVar1
X_opt = X[:,[0,3,4,5]] # As in the algo, remove that, fit rest
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()  # OrdinaryLeastSquares
regressor_OLS.summary()
# In the summary, we see that highest p value is for col 2 ie inputVar4 ie 60%, our p should be less than 5%
# In the next iteration, remove that inputVar ie col2 ie inputVar4
# ie Copy paste the above three lines without col2 ie inputVar4
X_opt = X[:,[0,3,5]] # As in the algo, remove that, fit rest
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()  # OrdinaryLeastSquares
regressor_OLS.summary()
# In the summary, we see that highest p value is for col 3 ie inputVar5 ie 6%, our p should be less than 5%
# In the next iteration, remove that inputVar ie col3 ie inputVar5
# ie Copy paste the above three lines without col3 ie inputVar5
X_opt = X[:,[0,3]] # As in the algo, remove that, fit rest
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()  # OrdinaryLeastSquares
regressor_OLS.summary()
# The Feature Subset Selection with BackwardElim led to the feature Subset [0,3]
# R&D Spend contributes most to the profit

'''
### Also, in python, you can do automatic backward elemination

## Backward Elimination with p values only

import statsmodels.formula.api as sm
def backwardElimination(x, sl):
    numVars = len(x[0])
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        if maxVar > sl:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    x = np.delete(x, j, 1)
    regressor_OLS.summary()
    return x
 
SL = 0.05
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
X_Modeled = backwardElimination(X_opt, SL)

# Backward elimination with p values and adjusted R-Squared
import statsmodels.formula.api as sm
def backwardElimination(x, SL):
    numVars = len(x[0])
    temp = np.zeros((50,6)).astype(int)
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        adjR_before = regressor_OLS.rsquared_adj.astype(float)
        if maxVar > SL:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    temp[:,j] = x[:, j]
                    x = np.delete(x, j, 1)
                    tmp_regressor = sm.OLS(y, x).fit()
                    adjR_after = tmp_regressor.rsquared_adj.astype(float)
                    if (adjR_before >= adjR_after):
                        x_rollback = np.hstack((x, temp[:,[0,j]]))
                        x_rollback = np.delete(x_rollback, j, 1)
                        print (regressor_OLS.summary())
                        return x_rollback
                    else:
                        continue
    regressor_OLS.summary()
    return x
 
SL = 0.05
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
X_Modeled = backwardElimination(X_opt, SL)

'''
# Training this feature subset with the regressor
X_opt_train, X_opt_test, y_opt_train, y_opt_test = train_test_split(X_opt, y, test_size = 0.2, random_state = 0) # you may choose your own test size
backwardElimRegressor = LinearRegression()
backwardElimRegressor.fit(X_opt_train, y_opt_train)
# Predict new results with this backwardEliminated subset regressor
y_backwardElimPred = backwardElimRegressor.predict(X_opt_test)












