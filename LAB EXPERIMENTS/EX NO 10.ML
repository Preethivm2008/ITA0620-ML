# Import required libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

# Generate sample data
np.random.seed(42)

X1 = np.random.normal(5, 1, 100)
X2 = np.random.normal(10, 1, 100)

X = np.concatenate([X1, X2]).reshape(-1, 1)

# Create Gaussian Mixture Model
gmm = GaussianMixture(n_components=2, random_state=42)

# Fit the model using EM algorithm
gmm.fit(X)

# Predict cluster labels
labels = gmm.predict(X)

# Display results
print("Means of Clusters:")
print(gmm.means_)

print("\nVariances of Clusters:")
print(gmm.covariances_)

print("\nMixing Weights:")
print(gmm.weights_)

# Display clustering result
plt.figure(figsize=(8, 5))

plt.scatter(X[:, 0], np.zeros(len(X)),
            c=labels, cmap='viridis', s=50)

plt.title("Expectation-Maximization Algorithm")
plt.xlabel("Data Values")
plt.ylabel("Clusters")
plt.show()
