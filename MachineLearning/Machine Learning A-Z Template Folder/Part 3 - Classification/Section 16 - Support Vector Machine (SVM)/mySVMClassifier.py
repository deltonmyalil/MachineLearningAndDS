# my support vector machine classifier
# importing the dataset
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# reading the dataset
dataset = pd.read_csv("Social_Network_Ads.csv")
X = dataset.iloc[:, [2,3]].values
y = dataset.iloc[:, 4].values
