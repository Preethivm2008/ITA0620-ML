import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error

# Dataset
X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
y = np.array([1, 4, 9, 16, 25])

# Linear Regression
linear = LinearRegression()
linear.fit(X, y)
y_linear = linear.predict(X)

# Polynomial Regression
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

polynomial = LinearRegression()
polynomial.fit(X_poly, y)
y_poly = polynomial.predict(X_poly)

# Compare results
print("Actual Values:             ", y)
print("Linear Predictions:        ", np.round(y_linear, 2))
print("Polynomial Predictions:    ", np.round(y_poly, 2))

print("Linear Regression MSE:", round(mean_squared_error(y, y_linear), 2))
print("Polynomial Regression MSE:", round(mean_squared_error(y, y_poly), 2))