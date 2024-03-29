#!/Users/weezygeezer/anaconda3/bin/python3

import numpy as np



def unit_step_func(x):
	return np.where(x>0, 1, 0)

# perceptron vs naive bayes
# does not make strong independence assumptions involving features; must be linearly separable! (think xor)
# Solution to XOR: introdude bigrams! Perceptron & unigrams do not work for data not linearly separable.
# discriminative linear classifier 


class Perceptron:
	
	def __init__(self, learning_rate=0.1,n_iters=1000):
		self.lr = learning_rate
		self.n_iters = n_iters
		self.activation_func = unit_step_func
		self.weights = None
		self.bias = None
	
	
	def fit(self, X, y):
		n_samples, n_features = X.shape

		# init parameters
		# online learning; weights change after examples; adding or subtracting
		self.weights = np.zeros(n_features)
		self.bias = 0
		
		y_ = np.where(x>0, 1, 0)
	
		# learn weights; look over epochs, or until convergence
		for _ in range(self.n_iters):
			for idx, x_i in enumerate(X):
				# Make a prediction based on the current weights (+1, -1)
				linear_output = np.dot(x_i, self.weights) + self.bias
				y_predicted = self.activation_func(linear_output) 
				
				# Update weights if an error was made 
				update = self.lr * (y_[idx] - y_predicted)
				self.weights +=  update * x_i
				self.bias += update
				
	def predict(self, X):
		linear_ouput = np.dot(X, self.weights) + self.bias
		y_predicted = self.activation_func(linear_output)
		return y_predicted


