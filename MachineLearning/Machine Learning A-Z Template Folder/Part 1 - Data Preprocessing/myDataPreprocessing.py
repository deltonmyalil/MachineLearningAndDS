# -*- coding: utf-8 -*-
"""
Data Preprocessing

"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# import dataset
dataSet = pd.read_csv("Data.csv")
# To get matrix of input variables
# iloc[left of the comma means all the rows, right of the comma is columns]
# iloc[colon here means we need all the rows, :-1 means all columns except the last]
X = dataSet.iloc[:, :-1].values

# To get the dependent variable vector
# [we need all rows, we need the 3rd column(couting starts from 0)]
Y = dataSet.iloc[:, 3].values

# Replace missing values with mean of all the vals
from sklearn.preprocessing import Imputer
# make obj of Imputer class
# Replace missing values marked as NaN with the global mean among the 0th axis ie col
imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
imputer = imputer.fit(X[:,1:3]) # over all rows, for columns 1 to 2 (upper bound of 3 is ignored but should be specified)
X[:,1:3] = imputer.transform(X[:,1:3])

# Now do for Categorical data as columns 0 and 3 are categorical
# Country contains 3 cats, Purchased contain 2 cats
# Encode the text in categorical vars into numbers
from sklearn.preprocessing import LabelEncoder
labelEncoderX = LabelEncoder()
X[:, 0] = labelEncoderX.fit_transform(X[:, 0]) # Encode all the rows of the oth column and replace X's first col with it
# The output is the array with numerical encoding for each of the countries
# However, the machine learning model will think that France < Spain as 0<2.
# To prevent that, we will introduce dummy variables
# For that, use this
from sklearn.preprocessing import OneHotEncoder
oneHotEncoder = OneHotEncoder(categorical_features=[0]) # 0th col has categorical values
X = oneHotEncoder.fit_transform(X).toarray()
# Now we added dummyVars to X to negate the effect of one categorical var
# First col corresponds to France, second to Ger and third to Spain

# Purchased col is dependend variable, no need to encode with dummy Vars
labelEncoderY = LabelEncoder()
Y = labelEncoderY.fit_transform(Y) # 0 is No and 1 is Yes

# NOTE: sklearn.model_selection has now replaced the old sklearn.cross_validation

# Splitting the data into training set and test set
from sklearn.model_selection import train_test_split
# X is the input matrix, Y is the  dependend vaariable array
# test_size means what ratio of the data is the test set
# random_state is the seed given to the randNumGen to sample the test_set
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size = 0.2, random_state = 0)

# Feature Scaling
# We need to pupt the variables in the same scale
# Age ranges from 20 to 50 and salary ranges from 40k to 80k
# These need to be in the same scale
# These should be in the same range say -1 to 1
# Eg Standardization ie Xstand = (x - mean)/(sd)
# eg Normalization ie Xnorm = (x - min(x))/(max(x)-min(x))
from sklearn.preprocessing import StandardScaler
scX = StandardScaler()
Xtrain = scX.fit_transform(Xtrain)
Xtest = scX.transform(Xtest)
  