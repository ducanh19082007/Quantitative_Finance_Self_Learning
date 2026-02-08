import pandas as pd
import numpy as np

class Linear_Regression_Sc:
    def __init__(self, learning_rate = 0.01, epochs = 1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.slope = 0 #slope
        self.bias = 0 #intercepts
    
    def fit(self, X, y):
        """
        since by definition the error function for linear regression is:
        error = y - y_pred
        given the formula of any linear regression function: y = wx + b
        For that we could deduce the Loss function as
        L(w,b) = (1/n)*np.sum(y_i - (w*x_i + b))^2
        as we do know in the vanilla gradient descent equation as:
        w_new = w_old - (d/dw)L(m,b)
        b_new = b_old - (d/db)L(m,b)
        
        hence when we do some derivative and calculus:
        w_new = w_old - (-2 / n) * np.sum(X * (y - y_pred))
        b_new = b_old - (-2 / n) * np.sum(y - y_pred)
        
        """
        n = len(X)
        
        for _ in range(self.epochs):
            y_pred = X*self.slope + self.bias
            dm = (-2 / n) * np.sum(X * (y - y_pred))
            db = (-2 / n) * np.sum(y - y_pred)
            
            self.slope -= self.learning_rate * dm
            self.bias -= self.learning_rate * db
            
    def predict(self, X):
        return self.slope * X + self.bias

    