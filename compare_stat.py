import numpy as np
import pandas as pd
from scipy.stats import ttest_ind

class StatisticalAnalyzer:
    def __init__(self, std_dev_threshold=3.0):
        self.std_dev_threshold = std_dev_threshold

    def remove_outliers(self, series):
        """Standard Z-Score Outlier Removal."""
        series = np.array(series)
        mean = np.mean(series)
        std = np.std(series)
        lower, upper = mean - (self.std_dev_threshold * std), mean + (self.std_dev_threshold * std)
        clean = series[(series >= lower) & (series <= upper)]
        return clean

    def compare_bands(self, band_name, data_a, data_b):
        """
        Compares two arrays of band power values.
        Input: Two 1D arrays (lists or numpy arrays) of power values for a specific band.
        Output: Dictionary containing stats results.
        """
        # 1. Clean Data
        clean_a = self.remove_outliers(data_a)
        clean_b = self.remove_outliers(data_b)

        if len(clean_a) < 2 or len(clean_b) < 2:
            return {
                "valid": False,
                "error": "Not enough data points after cleaning."
            }

        # 2. Welch's T-Test
        t_stat, p_val = ttest_ind(clean_b, clean_a, equal_var=False)
        
        mean_a = np.mean(clean_a)
        mean_b = np.mean(clean_b)
        pct_change = ((mean_b - mean_a) / mean_a) * 100 if mean_a != 0 else 0

        # 3. Interpretation
        significant = p_val < 0.05
        if significant:
            conclusion = f"File B has significantly {'HIGHER' if mean_b > mean_a else 'LOWER'} {band_name}."
        else:
            conclusion = "No significant difference."

        return {
            "valid": True,
            "band_name": band_name,
            "mean_a": mean_a,
            "mean_b": mean_b,
            "pct_change": pct_change,
            "p_value": p_val,
            "significant": significant,
            "conclusion": conclusion,
            "clean_a": clean_a,
            "clean_b": clean_b
        }