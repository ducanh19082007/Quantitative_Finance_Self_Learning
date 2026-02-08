import pandas as pd
import numpy as np

class Polynomial_Regression_Sc:
    def __init__(self, degree, learning_rate = 0.01, epochs = 1000):
        self.learning_rate = learning_rate
        self.degree = degree
        self.epochs = epochs
        self.slope = np.zeros(self.degree + 1)
            
    def fit(self, X, y):
        n = len(X)
        
        for _ in range(self.epochs):
            y_pred = np.zeros
            for j in range(self.degree + 1):
                y_pred += self.slope[j] * X ** j
                
            for j in range(self.degree + 1):
                dw = (-2 / n) * np.sum((y - y_pred) * (X ** j))
                self.slope -= self.learning_rate * dw
    
    def predict(self, X):
        y_pred = np.zeros(len(X))
        for j in range(self.degree + 1):
            y_pred += self.slope[j] * (X ** j)
        return y_pred
