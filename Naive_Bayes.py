import pandas as pd
import numpy as np

class NaiveBayes:
    def __init__(self):
        self.phi = None
        self.phi_y = None
        

    def fit(self, x_train, y_train):
        m = y_train.shape[0] # Number of training examples.
        input_feature = x_train.shape[1] # Number of input features. In our case, 2500.
        x_train = x_train.reshape(m, -1)

        # Laplace Smoothing
        x_train = np.append(x_train, [np.ones(input_feature),np.ones(input_feature)], axis=0)
        y_train = np.append(y_train, [0,1])

        self.phi = np.zeros(2)
        self.phi_y = np.zeros((2, input_feature))

        for i in range(m+2):
            label = y_train[i]
            self.phi[label] += 1
            for j in range(input_feature):
                self.phi_y[label, j] += x_train[i,j]
        for label in range(2):
            for j in range(input_feature):        
                self.phi_y[label, j] = float(self.phi_y[label, j]) / self.phi[label]
            self.phi[label] = float(self.phi[label]) / m

    # Smart version, equivalent
    #    for label in range(2):
    #        indices = (y_train == label)     
    #        phi[label] = float(np.sum(indices)) / m
    #        phi_y[label] = np.mean(x_train[indices, :], axis=0)

        return self.phi, self.phi_y

    def predict(self, x_tests):
        input_feature = x_tests.shape[1]  # Number of input features. In our case, 2500.
        number_tests = x_tests.shape[0]
        # flatten the test data
        x_tests = x_tests.reshape(number_tests, -1)
        scores = np.zeros((number_tests, 2)) 
        for i in range(number_tests):
            for label in range(2):
                scores[i, label] = np.log(self.phi[label])            
                for j in range(input_feature):
                    if x_tests[i, j]:
                        scores[i, label] += np.log(self.phi_y[label][j])
                    else:
                        scores[i, label] += np.log(1 - self.phi_y[label][j])
        predictions = np.argmax(scores, axis=1)
        return predictions 
    
    def accuracy(self, y_test, predictions):
        return np.mean(y_test == predictions)
