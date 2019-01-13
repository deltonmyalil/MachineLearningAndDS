# -*- coding: utf-8 -*-
# Apriori uses support and confidence (Brace yourselves, Painful BDA memories incoming)
'''
Support for product A = number of transactions with A divided by total number of transactions
Confidence for product A will lead to product B is number of trans with A and B divided by no of trans with A
Lift for A leads to B is confidence of A leads to B divided by support of B
'''
# 1110, I am so sorry (for saying the rest of the measures probably wont come in the exam)

'''
Apriori ALgorithm
1. Set min support and min conf
2. Take all subsets inn transactions having support higher than min support
3. Take all the rules in these subsets having higher conf than min conf
4. Sort the rules by decreasing lift # Highest lift rules are the most likely
'''

'''
apyori.py is the python library. Either keep the poor thing (apyori.py) in the running folder or 
do a pip3 install apyori in the spyder(anaconda) terminal - which has to be installed separately
by
$conda install -c conda-forge spyder-terminal
I did the second to make future projects easy
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib auto

dataset = pd.read_csv("Market_Basket_Optimisation.csv", header=None)

# Apyori expects the dataset as a list of lists of all transactions where each item is rep in string
transactions = []
row = dataset.shape[0] # returns the number of rows
col = dataset.shape[1]


for i in range(row):
	transactions.append([str(dataset.values[i,j]) for j in range(col)])

from apyori import apriori # you either need to install apyori in conda or need to provide apyori file
'''
Data is recorded over a week
For support, lets take a support of item purchased three times a day
Therefore, for the week, it is 3*7.
Then Support = 3*7/#TotalTrans = 21/7500
'''
rules = apriori(transactions, min_support=0.003, min_confidence=0.2, min_lift=3, min_length=2)
results = list(rules)
results_list = []
for i in range(0, len(results)):
    results_list.append('RULE:\t' + str(results[i][0]) + '\nSUPPORT:\t' + str(results[i][1]))


















