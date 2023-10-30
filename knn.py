import numpy as np
from collections import Counter


# def euclidean distance
def euclidean_distance(x1, x2):
	# distance between 2 feature vectors
	return np.sqrt(np.sum((x1-x2)**2))


class KNN:

	def __init__(self, k):
		self.k = k
	
	# follow conventions from ML libraries
	
	def fit(self, X, y):
		# store training samples, X
		self.X_train = X
		self.y_train = y
	
	def predict(self, X):
		predicted_labels = [self._predict(x) for x in X]
		return np.array(predicted_labels)

	def _predict(self, x):
		# helped method; only get 1 sample
		# 1. Calculate Distance
		distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
		# 2. Sort & count nearest samples, labels
		k_indices = np.argsort(distances)[:self.k]
		k_nearest_labels = [self.y_train[i] for i in k_indices]
		# 3. Do Majority vote
		most_common = Counter(k_nearest_labels).most_common(1)
		return most_common[0][0]
		
