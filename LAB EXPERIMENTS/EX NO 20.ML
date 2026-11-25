# Import libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Create sample sales dataset
months = np.array([
    1, 2, 3, 4, 5, 6,
    7, 8, 9, 10, 11, 12
]).reshape(-1, 1)

sales = np.array([
    100, 120, 130, 150,
    170, 180, 200, 220,
    230, 250, 270, 300
])

# Create Linear Regression model
model = LinearRegression()

# Train model
model.fit(months, sales)

# Predict existing sales
predicted_sales = model.predict(months)

# Calculate performance
print("Mean Absolute Error:",
      mean_absolute_error(sales, predicted_sales))

print("R2 Score:", r2_score(sales, predicted_sales))

# Predict future sales for next 6 months
future_months = np.array([
    13, 14, 15, 16, 17, 18
]).reshape(-1, 1)

future_sales = model.predict(future_months)

# Display future sales
print("\nFuture Sales Prediction:")

for month, sale in zip(future_months.flatten(), future_sales):
    print("Month", month, "Predicted Sales:", round(sale, 2))

# Display graph
plt.figure(figsize=(8, 5))

plt.scatter(months, sales, color='blue', label='Actual Sales')
plt.plot(months, predicted_sales, color='red', label='Predicted Sales')
plt.plot(future_months, future_sales, 'g--', label='Future Sales')

plt.title("Future Sales Prediction using Linear Regression")
plt.xlabel("Month")
plt.ylabel("Sales")
plt.legend()
plt.grid(True)
plt.show()
