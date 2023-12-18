import numpy as np
from regression.utilities import sort_by_group
from scipy.stats import f
from prettytable import PrettyTable

class ANOVA():
    def __init__(self, x, y):
        self.categories, self.sorted_y = sort_by_group(x, y)
        self.G = len(self.categories)
        self.group_counts = np.array([np.sum(x == g) for g in self.categories])
        self.means = None
        self.bss = None
        self.wss = None
        self.tss = None
    
    def compute_means(self):
        self.means = np.zeros(self.G)
        start_index = 0
        for i in range(self.G):
            end_index = start_index + self.group_counts[i]
            self.means[i] = np.mean(self.sorted_y[start_index:end_index])
            start_index = end_index

        return self.means
    
    def compute_ss(self):
        if self.means is None: self.compute_means()
        
        overall_mean = np.mean(self.sorted_y)
        self.tss = np.sum((self.sorted_y - overall_mean) ** 2)
        self.bss = np.sum(self.group_counts * (self.means - overall_mean) ** 2)
        self.wss = self.tss - self.bss
        
        return self
    
    def f_test(self, alpha):
        if self.tss is None: self.compute_ss()
        
        N = len(self.sorted_y)
        df1 = self.G - 1
        df2 = N - self.G
        f_stat = (self.bss / df1) / (self.wss / df2)
        
        p_value = 1 - f.cdf(f_stat, df1, df2)
        reject_H0 = p_value <= alpha
        
        if reject_H0:
            print(f"With significance level {alpha*100}%, we REJECT H0, i.e., there exists a group with different mean.")
        else:
            print(f"With significance level {alpha*100}%, we do NOT reject H0, i.e., all groups have the same mean.")
        
        return reject_H0
    
    def table(self):
        if self.tss is None: self.compute_ss()
        
        N = len(self.sorted_y)
        df1 = self.G - 1
        df2 = N - self.G
        f_stat = (self.bss / df1) / (self.wss / df2)
        p_value = 1 - f.cdf(f_stat, df1, df2)
        
        anova_table = {
            "Effect": ["Factor", "Residual"],
            "DF": [df1, df2],
            "Effect SS": [self.bss, self.wss],
            "Effect MSE": [self.bss / df1, self.wss / df2],
            "F-stat": [f_stat, ""],
            "P-value": [p_value, ""]
        }
        
        output = PrettyTable()
        output.field_names = ["Effect", "DF", "Effect SS", "Effect MSE", "F-Stat", "P-Value"]
        output.add_row(["Factor", df1, self.bss, self.bss / df1, f_stat, p_value])
        output.add_row(["Residual", df2, self.tss, self.tss / df2, "", ""])

        print(output)
        
        return anova_table