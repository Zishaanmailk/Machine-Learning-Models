import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

# Simple data
X = np.array([[1, 2], [2, 3], [3, 3], [6, 6], [7, 7], [8, 8]])
y = np.array([0, 0, 0, 1, 1, 1])

# Initialize the KNN model
knn = KNeighborsClassifier(n_neighbors=3)

# Fit the model
knn.fit(X, y)

# Test points for predictions
test_points = np.array([[5, 5], [2, 2]])

# Predict the class for test points
predictions = knn.predict(test_points)

# Plotting the points
plt.scatter(X[:, 0], X[:, 1], c=y, marker='o', label='Training points')
plt.scatter(test_points[:, 0], test_points[:, 1], c=predictions, marker='x', label='Test points', s=100)

# Add legend and labels
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()

# Show the plot
plt.show()
