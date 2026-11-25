# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# Create sample mobile dataset
data = {
    'RAM': [2, 3, 4, 6, 8, 12, 4, 6, 8, 12],
    'Battery': [3000, 3500, 4000, 4500, 5000, 6000, 4200, 4800, 5200, 6500],
    'Memory': [32, 64, 64, 128, 128, 256, 128, 128, 256, 512],
    'Screen_Size': [5.0, 5.5, 6.0, 6.2, 6.5, 6.8, 6.1, 6.3, 6.7, 6.9],
    'Price': [5000, 8000, 10000, 15000, 20000, 35000, 12000, 18000, 25000, 45000]
}

df = pd.DataFrame(data)

# Select features and target
X = df[['RAM', 'Battery', 'Memory', 'Screen_Size']]
y = df['Price']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create model
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train model
model.fit(X_train, y_train)

# Predict price
y_pred = model.predict(X_test)

# Evaluate model
print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# Predict price for a new mobile
new_mobile = [[8, 5000, 128, 6.5]]
predicted_price = model.predict(new_mobile)

print("\nPredicted Mobile Price: Rs.", round(predicted_price[0], 2))

# Display graph
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Mobile Price Prediction")
plt.show()
