import numpy as np
from sklearn.neural_network import MLPClassifier

# Training dataset (XOR)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 0])

# Create ANN using Backpropagation
model = MLPClassifier(
    hidden_layer_sizes=(4,),
    activation='logistic',
    solver='lbfgs',
    max_iter=5000,
    random_state=1
)

# Train the network
model.fit(X, y)

# Test the network
predictions = model.predict(X)

print("Input\tPredicted Output")
for i in range(len(X)):
    print(X[i], "\t", predictions[i])

print("Accuracy:", model.score(X, y) * 100, "%")