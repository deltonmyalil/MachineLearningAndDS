# k nearest neighbour classifier
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# read the data to the dataset
dataset = pd.read_csv("SNAds.csv")

# changing the Gender feature from String to number
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
dataset.Gender = le.fit_transform(dataset.Gender)

# making matrices
X = dataset.iloc[:,[1,2,3]].values
y = dataset.iloc[:, 4].values

# data partitioning
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)



# fitting the classifier to the training set
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2) # considering 5 neighbors with metric = minkowski and p = 2 which means we are considering euclidian distance
classifier.fit(X_train, y_train)

# predicting the classifier results
y_pred = classifier.predict(X_test)

# create the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('K-NN (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

# After including gender, the performance is worse by one misclassification
