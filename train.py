#!/Users/weezygeezer/anaconda3/bin/python3

import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
cmap = ListedColormap(['#FF0000','#00FF00','#0000FF'])

iris = datasets.load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=1234)

# 0. Look at your data
# print (X_train.shape)
# print(X_train[0])
# # 120: # of samples 4: # of features
# print(y_train.shape)
# # 3 class problem; 0, 1, 2 red, green, blue
# print(y_train) 

# 0.1 Plot your data
# plt.figure()
# # first two features X[:, 0] X[:, 1] 
# plt.scatter(X[:,0],X[:,1],c=y,cmap=cmap,edgecolor='k',s=20)
# plt.show()

# KNN Implementation
from knn import KNN
clf = KNN(k=4)
clf.fit(X_train,y_train)
predictions = clf.predict(X_test)

acc = np.sum(predictions == y_test) / len(y_test)
print(acc)

