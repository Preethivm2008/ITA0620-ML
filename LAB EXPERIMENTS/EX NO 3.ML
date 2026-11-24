import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.preprocessing import LabelEncoder

data = pd.DataFrame({
    'Outlook': ['Sunny', 'Sunny', 'Rain', 'Rain', 'Overcast', 'Overcast'],
    'Wind': ['Weak', 'Strong', 'Weak', 'Strong', 'Weak', 'Strong'],
    'Play': ['Yes', 'No', 'Yes', 'No', 'Yes', 'Yes']
})

# Encode categorical data
encoders = {}
for col in data.columns:
    encoders[col] = LabelEncoder()
    data[col] = encoders[col].fit_transform(data[col])

X = data[['Outlook', 'Wind']]
y = data['Play']

# Build ID3 decision tree
model = DecisionTreeClassifier(criterion='entropy')
model.fit(X, y)

print("Decision Tree:")
print(export_text(model, feature_names=['Outlook', 'Wind']))

# Classify a new sample
sample = pd.DataFrame({
    'Outlook': [encoders['Outlook'].transform(['Sunny'])[0]],
    'Wind': [encoders['Wind'].transform(['Weak'])[0]]
})

result = encoders['Play'].inverse_transform(model.predict(sample))
print("Predicted Class:", result[0])