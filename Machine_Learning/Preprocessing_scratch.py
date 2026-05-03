import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class Standard_Scaler_Sc:
    def fit(self, X):
        self.mean = np.mean(X)
        self.std = np.std(X)
    
    def transform(self, X):
        return (X - self.mean) / self.std # "(x - phi ) / sigma"
    
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)