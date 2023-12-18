import numpy as np
from regression.estimators import OLS

class WLS():
    
    def __init__(self, standardize=True, add_bias=True):
        self.standardize_flag = standardize
        self.add_bias_flag = add_bias
        self.beta = None
        self.yh = None
        self.residuals = None
        self.sigma2_naive = None
        self.sigma2_corrected = None

    def standardize(self, x):
        return (x - np.mean(x, axis=0)) / np.std(x, axis=0)

    def add_bias(self, x):
        return np.insert(x, 0, 1, axis=1)
      
    def fit(self, x, y, w):
        if self.standardize_flag:
            x = self.standardize(x)
            
        if self.add_bias_flag:
            x = self.add_bias(x)
            
        self.N, self.P = x.shape
            
        if w.ndim == 1:
            w = np.diag(w)

        self.beta = np.linalg.inv(x.T @ w @ x) @ (x.T @ w @ y)

        self.yh = x @ self.beta
        self.residuals = y - self.yh
        self.rss = np.sum(self.residuals ** 2)
        self.sigma2_naive = self.rss / self.N
        self.sigma2_corrected = self.rss / (self.N - self.P)
        
        return self
    
    def predict(self, x):
        if self.beta is None:
            raise ValueError("The model has not been fitted yet. Call fit() before predict().")
        
        if self.standardize_flag:
            x = self.standardize(x)
            
        if self.add_bias_flag:
            x = self.add_bias(x)
            
        return x @ self.beta

    def score(self, y, adjusted = False):
        if self.yh is None:
            raise ValueError("The model has not been fitted yet. Call fit() before score().")
        
        y_mean = np.mean(y)
        r2 = np.sum((self.yh - y_mean) ** 2) / np.sum((y - y_mean) ** 2)
        
        if adjusted:
            r2 = 1 - (1 - r2) * (self.N / (self.N - self.P))
            
        return r2
    
    def mse(self, y):
        mse = np.mean((y - self.yh) ** 2)
        
        return mse
    
    def feasible_wls(self, x, y):
        ols_model = OLS(standardize=self.standardize_flag, add_bias=self.add_bias_flag)
        ols_model.fit(x, y)
        
        residuals = ols_model.residuals
        squared_residuals = residuals ** 2
        ols_model.fit(x, squared_residuals)
        predictions = ols_model.yh
        weights = 1 / predictions
        
        self.fit(x, y, weights)
        
        return self.beta
    
    def hat_matrix(self, x, w): # returns the Hat Matrix
        H = x @ np.linalg.inv(x.T @ w @ x) @ x.T @ w
        return H

    def compute_aic(self, x, y, w): # computes the Akaike Information Criterion
        residuals = y - self.yh
        ssr = np.sum(residuals ** 2)
        aic = self.N + self.N * np.log(2 * np.pi * (1/self.N) * ssr) + (2 * np.trace(self.h_matrix(x, w)))
        return aic

    def compute_bic(self, x, y, w): # computes the Bayesian Information Criterion
        residuals = y - self.yh
        ssr = np.sum(residuals ** 2)
        bic = self.N + (self.N * np.log(2 * np.pi * (1/self.N) * ssr)) + (np.log(self.N) * np.trace(self.h_matrix(x, w)))
        return bic