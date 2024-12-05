import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

class LogisticRegression:
    def __init__(self, learning_rate=0.01, iterations=1000):
        self.learning_rate = learning_rate
        self.iterations = iterations

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        self.weights = np.zeros(X.shape[1])
        for i in range(self.iterations):
            z = np.dot(X, self.weights)
            predictions = self.sigmoid(z)
            # Gradient descent
            gradient = np.dot(X.T, (predictions - y)) / y.size
            self.weights -= self.learning_rate * gradient

    def predict(self, X):
        z = np.dot(X, self.weights)
        return [1 if i > 0.5 else 0 for i in self.sigmoid(z)]
    
    def accuracy(self, y_true, y_pred):
        accuracy = np.sum(y_true == y_pred) / len(y_true)
        return accuracy

class LogisticRegression_sklearn:
    def __init__(self):
        self.model = LogisticRegression()

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def accuracy(self, y_true, y_pred):
        accuracy = self.model.accuracy_score(y_true, y_pred)
        print(f"Accuracy: {accuracy}")
        return accuracy
