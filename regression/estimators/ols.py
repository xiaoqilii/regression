import numpy as np
from scipy.stats import t, f
from pprint import pprint
    
class OLS():
    
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
      
    def fit(self, x, y):
        if self.standardize_flag:
            x = self.standardize(x)
            
        if self.add_bias_flag:
            x = self.add_bias(x)
            
        self.N, self.P = x.shape
        self.invDn = np.linalg.inv(x.T @ x)
        self.beta = self.invDn @ (x.T @ y)

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

    def t_test(self, j, bj, alpha, use_p_value = False, print_output = True):
        if self.beta is None:
            raise ValueError("Model must be fitted before performing a T-test. Call fit() before performing tests.")
        
        t_stat = (self.beta[j] - bj) / (np.sqrt(self.sigma2_corrected * self.invDn[j, j]))
        df = self.N - self.P
        
        if use_p_value:
            p_value = 2 * (1 - t.cdf(np.abs(t_stat), df))
            reject_H0 = p_value < alpha
        else:
            critical_value = t.ppf(1 - alpha/2, df)
            reject_H0 = np.absolute(t_stat) > critical_value
            
        if print_output:
            if reject_H0:
                print(f"With significance level {alpha*100}%, we REJECT H0.")
            else:
                print(f"With significance level {alpha*100}%, we do NOT reject H0.")
        
        accept_H0 = not reject_H0
        return accept_H0
    
    def f_test_reduced_model(self, x_reduced, y, alpha, print_output = True):
        if self.beta is None:
            raise ValueError("Model must be fitted before performing an F-test. Call fit() before performing tests.")
        
        if self.standardize_flag:
            x_reduced = self.standardize(x_reduced)
            
        if self.add_bias_flag:
            x_reduced = self.add_bias(x_reduced)
            
        p1 = x_reduced.shape[1]
        p2 = self.P - p1
        beta_reduced = np.linalg.inv((x_reduced.T).dot(x_reduced)).dot((x_reduced.T).dot(y))
        yh_reduced = x_reduced.dot(beta_reduced)
        residuals_reduced = y - yh_reduced
        sigma2_corrected_reduced = np.sum(residuals_reduced ** 2) / (self.N - p1)
        
        # Calculate the F-statistic
        f_stat = ((sigma2_corrected_reduced - self.sigma2_corrected) / (p2)) / (self.sigma2_corrected / (self.N - self.P))
        df1 = p2
        df2 = self.N - self.P
        
        # Calculate the critical value or p-value
        p_value = 1 - f.cdf(f_stat, df1, df2)
        reject_H0 = p_value < alpha
        
        if print_output:
            if reject_H0:
                print(f"With significance level {alpha*100}%, we REJECT H0, i.e., the full model is correct.")
            else:
                print(f"With significance level {alpha*100}%, we do NOT reject H0, i.e., the reduced model is correct.")
        
        accept_H0 = not reject_H0
        return accept_H0

    def f_test_constraints(self, R, r, alpha, print_output = True):
        if self.beta is None:
            raise ValueError("Model must be fitted before performing an F-test. Call fit() before performing tests.")

        k = r.shape[0]
        numerator = (R @ self.beta - r).T @ (np.linalg.inv(R @ self.invDn @ R.T)) @ (R @ self.beta - r)
        f_stat = (numerator / k) / self.sigma2_corrected
        df1 = k
        df2 = self.N - self.P
        
        # Calculate the p-value
        p_value = 1 - f.cdf(f_stat, df1, df2)
        reject_H0 = p_value < alpha
        
        if print_output:
            if reject_H0:
                print(f"With significance level {alpha*100}%, we REJECT H0, i.e., the constraints do NOT hold for this model.")
            else:
                print(f"With significance level {alpha*100}%, we do NOT reject H0, i.e., the constraints hold for this model.")
        
        return not reject_H0
    
    def confidence_interval_single(self, j, alpha, print_output = True):
        if self.beta is None:
            raise ValueError("Model has not yet been fitted. Call fit() before computing confidence intervals.")
        
        df = self.N - self.P
        critical_value = t.ppf(1 - alpha/2, df)
        
        left = self.beta[j] - critical_value * np.sqrt(self.sigma2_corrected * self.invDn[j, j])
        right = self.beta[j] + critical_value * np.sqrt(self.sigma2_corrected * self.invDn[j, j])
        
        if print_output:
            print(f"The interval [{left}, {right}] has EXACT coverage probability {(1 - alpha)*100}% for coefficient B_{j}.")
        
        return left, right
        
    def confidence_interval_bonferonni(self, coefficients, alpha, print_output = True):
        if self.beta is None:
            raise ValueError("Model has not yet been fitted. Call fit() before computing confidence regions.")
        
        p = len(coefficients)
        alpha_j = alpha / p
        
        intervals = []
        for j in coefficients:
            left, right = self.confidence_interval_single(j, alpha_j, print_output = False)
            intervals.append((left, right))
        
        if print_output:
            print(f"The region defined by") 
            pprint(intervals)
            print(f"has coverage probability GREATER THAN {(1 - alpha)*100}% " 
                  f"for coefficients {coefficients}."
            )

        return intervals
    
    def confidence_ellipsoid_test(self, test_beta, alpha, print_output = True):
        if self.beta is None:
            raise ValueError("Model has not yet been fitted. Call fit() before computing confidence regions.")
        
        numerator = (self.beta - test_beta).T @ np.linalg.inv(self.invDn) @ (self.beta - test_beta)
        stat = numerator / self.sigma2_corrected
        df1 = self.P
        df2 = self.N - self.P
        critical_value = f.ppf(1 - alpha, df1, df2)
        radius = self.P * critical_value
        
        contained = stat <= radius
        
        if print_output:
            if contained:
                print(f"The {(1 - alpha)*100}% confidence region for Beta is an ellipsoid centered at\n" 
                      f"{self.beta}\n" 
                      f"with radius {radius}.\n"
                      f"This region CONTAINS your particular Beta = \n"
                      f"{test_beta}."
                )
            else:
                print(f"The {(1 - alpha)*100}% confidence region for Beta is an ellipsoid centered at\n" 
                      f"{self.beta}\n" 
                      f"with radius {radius}.\n"
                      f"This region DOES NOT contain your particular Beta = \n"
                      f"{test_beta}."
                )
        
        return contained
    
    def confidence_interval_predicted(self, x_new, alpha, for_y = False, print_output = True):
        if self.beta is None:
            raise ValueError("Model has not yet been fitted. Call fit() before computing confidence intervals for new points.")

        df = self.N - self.P
        m_new = x_new.T @ self.beta
        critical_value = t.ppf(1 - alpha/2, df)
        hxx = x_new.T @ self.invDn @ x_new
        if for_y: hxx += 1
        
        left = m_new - critical_value * np.sqrt(self.sigma2_corrected * hxx)
        right = m_new + critical_value * np.sqrt(self.sigma2_corrected * hxx)
        
        if print_output:
            if for_y:
                print(f"The interval [{left}, {right}] has EXACT coverage probability {(1 - alpha)*100}% for y_new.")
            else:
                print(f"The interval [{left}, {right}] has EXACT coverage probability {(1 - alpha)*100}% for m(x_new).")
            
        return left, right

    def h_matrix(self, x): # returns the Hat Matrix
        H = x @ np.linalg.inv(x.T @ x) @ x.T
        return H

    def compute_aic(self, x, y): # Computes the Akaike Information Criterion
        residuals = y - self.yh
        ssr = np.sum(residuals ** 2)
        aic = self.N + self.N * np.log(2 * np.pi * (1/self.N) * ssr) + (2 * np.trace(self.h_matrix(x)))
        return aic
    
    def compute_bic(self, x, y): # Computes the Bayesian Information Criterion
        residuals = y - self.yh
        ssr = np.sum(residuals ** 2)
        bic = self.N + (self.N * np.log(2 * np.pi * (1/self.N) * ssr)) + (np.log(self.N) * np.trace(self.h_matrix(x)))
        return bic