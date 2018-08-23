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