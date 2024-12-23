#Write a python program to explore statistical methods for machine learning

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_regression

# 1. Central Tendency (Mean, Median, Mode)
data = np.array([2, 4, 6, 8, 10, 10, 12, 14, 16, 18])
mean = np.mean(data)
median = np.median(data)
mode = stats.mode(data)[0]
print(f"Mean: {mean}, Median: {median}, Mode: {mode}")

# 2. Dispersion (Variance, Standard Deviation)
variance = np.var(data)
std_deviation = np.std(data)
print(f"Variance: {variance}, Standard Deviation: {std_deviation}")

# 3. Correlation
# Creating two sets of data points
x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 6, 8, 10])
correlation, _ = stats.pearsonr(x, y)
print(f"Correlation between x and y: {correlation}")

# 4. Linear Regression (Simple Linear Model)
# Generate synthetic data for regression
X, y = make_regression(n_samples=100, n_features=1, noise=10, random_state=42)

# Train a linear regression model
model = LinearRegression()
model.fit(X, y)

# Coefficients of the linear model
print(f"Linear Regression Coefficients: {model.coef_[0]}, Intercept: {model.intercept_}")

# Plotting the regression line
plt.scatter(X, y, color='blue', label='Data points')
plt.plot(X, model.predict(X), color='red', label='Regression line')
plt.title("Linear Regression Example")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.show()

# 5. Z-Score Normalization
# Standardizing a dataset (Z-score normalization)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print(f"Original data: {X[:5]}")
print(f"Scaled data: {X_scaled[:5]}")

# 6. Hypothesis Testing (P-value)
# Example: Testing if the mean of the data is equal to 10
t_statistic, p_value = stats.ttest_1samp(data, 10)
print(f"T-statistic: {t_statistic}, P-value: {p_value}")

# If p-value < 0.05, we reject the null hypothesis that the mean is 10
if p_value < 0.05:
    print("Reject the null hypothesis: The sample mean is significantly different from 10.")
else:
    print("Fail to reject the null hypothesis: The sample mean is not significantly different from 10.")
