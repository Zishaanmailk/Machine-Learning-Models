import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from mpl_toolkits.mplot3d import Axes3D
 # For 3D plotting

# Simple 3D data
X = np.array([[1, 2, 3], [2, 3, 4], [3, 3, 3], [6, 6, 6], [7, 7, 8], [8, 8, 9]])
y = np.array([0, 0, 0, 1, 1, 1])

# Initialize the KNN model
knn =KNeighborsClassifier(n_neighbors=3)

# Fit the model
knn.fit(X, y)

# Test points for predictions
test_points = np.array([[5, 5, 5], [2, 2, 2]])

# Predict the class for test points
predictions = knn.predict(test_points)

# Display the predictions using a simple for loop
for i in range(len(test_points)):
    print(f"Test point {test_points[i]} is predicted as class {predictions[i]}")

# Plotting in 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot training points
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, marker='o', label='Training points')

# Plot test points
ax.scatter(test_points[:, 0], test_points[:, 1], test_points[:, 2], c=predictions, marker='x', label='Test points', s=100)

# Add labels and legend
ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.set_zlabel('Feature 3')
ax.legend()

# Show the plot
plt.show()
