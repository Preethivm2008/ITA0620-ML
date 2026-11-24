import pandas as pd
from sklearn.linear_model import LinearRegression

# Create dataset
data = {
    'Year': [2018, 2019, 2020, 2021, 2022, 2023],
    'Kilometers': [50000, 40000, 30000, 20000, 15000, 10000],
    'Price': [500000, 550000, 650000, 750000, 850000, 950000]
}

df = pd.DataFrame(data)

# Train model
X = df[['Year', 'Kilometers']]
y = df['Price']

model = LinearRegression()
model.fit(X, y)

# Predict car price
new_car = pd.DataFrame([[2022, 12000]], columns=['Year', 'Kilometers'])
price = model.predict(new_car)

print("Predicted Car Price: ₹", round(price[0], 2))