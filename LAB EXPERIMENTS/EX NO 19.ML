# Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Create bank loan dataset
data = {
    'Age': [25, 35, 45, 28, 50, 40, 23, 60, 32, 48,
            27, 55, 38, 42, 29, 52, 33, 46, 31, 58],
    'Income': [25000, 50000, 70000, 30000, 80000, 60000,
               20000, 90000, 45000, 75000, 28000, 85000,
               55000, 65000, 32000, 78000, 48000, 72000,
               40000, 95000],
    'Credit_Score': [600, 750, 800, 620, 820, 780, 580, 850,
                     700, 810, 610, 830, 760, 790, 640, 820,
                     710, 800, 680, 860],
    'Employment': ['No', 'Yes', 'Yes', 'No', 'Yes', 'Yes',
                   'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes',
                   'Yes', 'Yes', 'No', 'Yes', 'Yes', 'Yes',
                   'No', 'Yes'],
    'Loan_Status': ['No', 'Yes', 'Yes', 'No', 'Yes', 'Yes',
                    'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes',
                    'Yes', 'Yes', 'No', 'Yes', 'Yes', 'Yes',
                    'No', 'Yes']
}

df = pd.DataFrame(data)

# Convert categorical data into numbers
encoder = LabelEncoder()
df['Employment'] = encoder.fit_transform(df['Employment'])
df['Loan_Status'] = encoder.fit_transform(df['Loan_Status'])

# Select input and output
X = df[['Age', 'Income', 'Credit_Score', 'Employment']]
y = df['Loan_Status']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Create Naive Bayes model
model = GaussianNB()

# Train model
model.fit(X_train, y_train)

# Predict output
y_pred = model.predict(X_test)

# Calculate accuracy
print("Accuracy:", accuracy_score(y_test, y_pred) * 100, "%")

# Display confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Display classification report
print("\nClassification Report:")
print(classification_report(
    y_test, y_pred,
    labels=[0, 1],
    target_names=['Rejected', 'Approved'],
    zero_division=0
))

# Predict loan status for a new customer
new_customer = [[30, 50000, 750, 1]]

prediction = model.predict(new_customer)

if prediction[0] == 1:
    print("\nLoan Status: Approved")
else:
    print("\nLoan Status: Rejected")
