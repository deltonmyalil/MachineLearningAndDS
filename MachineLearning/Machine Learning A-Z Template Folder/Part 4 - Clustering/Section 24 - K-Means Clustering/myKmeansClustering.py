#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 19 15:40:43 2018
WCSS: Optimal numbe rof clusters is the one where the graph between WCSS and Number of clusters shows an elbow
@author: deltonmyalil
"""
'''
WCSS: WIthin Cluster Sum of Squares
Minimizing WCSS is maximizing the distance betwee clusters
'''  
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv("Mall_Customers.csv")

X = dataset.iloc[:,[3,4]] # All rows, column 3 and 4 ie Income and SpendingScore

# We dont know the optimal number of clusters
# Use the elbow method to find it

%matplotlib auto 
# This is to make the graph appear in separate window instead of inline in console
# To revert back to inline use %matplotlib inline
from sklearn.cluster import KMeans
wcss = [] # Note: wcss is called intertia in scikit_
for i in range(1,11):
    # perform clustering for each i
    # init means the random centroid initialization method, here we are using k-means++
    #   This prevents the random initialization trap of forming meaningless clusters when you
    #   initialize the centroids randomly and it went to the most unluckiest places
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(X)
    wcssValue = kmeans.inertia_ # Sum of squared distances of samples to their closest cluster center (returns float)
    wcss.append(wcssValue)
plt.plot(range(1,11), wcss)
plt.title("Elbow method")
plt.xlabel("Number of Clusters")
plt.ylabel("Within Cluster Sum of Squares")
plt.show()
# We can see that 5 (truth) is the elbow (Truth is a matter of perspective - Capt. John Price, Call of Duty Modern Warfare 2)

# Now that we found out that the k value is arguable 5, we fit it to the mall dataset
kmeans = KMeans(n_clusters=5, init='k-means++', max_iter=300, n_init=10, random_state=0)
# We use fit_predict here instead of fit as we need results
y_kmeans = kmeans.fit_predict(X)
# y_kmeans is a vector with index equals customer number or ID and value equals the cluster number which belongs to [0,4] for k = 5

# Visualizing the unclustered dataset
plt.scatter(dataset.iloc[:,[3]], dataset.iloc[:,[4]])
plt.show()

# Visualizing the clusters
plt.scatter(X[y_kmeans==0, 0], X[y_kmeans==0, 1], s=100, c='red', label='Cluster1')
plt.scatter(X[y_kmeans==1, 0], X[y_kmeans==1, 1], s=100, c='green', label='Cluster2')
plt.scatter(X[y_kmeans==2, 0], X[y_kmeans==2, 1], s=100, c='blue', label='Cluster3')
plt.scatter(X[y_kmeans==3, 0], X[y_kmeans==3, 1], s=100, c='yellow', label='Cluster4')
plt.scatter(X[y_kmeans==4, 0], X[y_kmeans==4, 1], s=100, c='orange', label='Cluster5')
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s=300, c='violet', label='Centroids')
plt.title("Customer Clustering with k = 5")
plt.xlabel("Annual Income")
plt.ylabel("SpendingScore")
plt.legend()
plt.show()

# Hell Yeah! 5 clusters