import pandas as pd
from sklearn.linear_model import LinearRegression

# Create dataset
data = {
    'Area': [1000, 1500, 1800, 2000, 2500, 3000],
    'Bedrooms': [2, 3, 3, 4, 4, 5],
    'Price': [2000000, 3000000, 3600000, 4000000, 5000000, 6000000]
}

df = pd.DataFrame(data)

# Train model
X = df[['Area', 'Bedrooms']]
y = df['Price']

model = LinearRegression()
model.fit(X, y)

# Predict house price
new_house = pd.DataFrame([[2200, 4]], columns=['Area', 'Bedrooms'])
price = model.predict(new_house)

print("Predicted House Price: ₹", round(price[0], 2))