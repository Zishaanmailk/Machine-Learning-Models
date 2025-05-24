import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Initial dataset
#For separate data you can use np.column_stack()
data = np.array([
    [1, 2], [1.5, 1.8], [5, 8], [8, 8],
    [1, 0.6], [9, 11], [8, 2], [10, 2],
    [9, 3]
])

# Initial cluster centers
initial_centers = np.array([[1, 1], [5, 5]])

# Define and fit KMeans model with initial data
kmeans = KMeans(n_clusters=2, init=initial_centers, n_init=1, max_iter=300)
kmeans.fit(data)

# New data points
new_data = np.array([[4, 5], [6, 9]])

# Add new data to the original data array and retrain model
data = np.vstack([data, new_data])
kmeans = KMeans(n_clusters=2, init=kmeans.cluster_centers_, n_init=1, max_iter=300)
kmeans.fit(data)

# Predict clusters for the new data points
labels = kmeans.predict(data)
centroids = kmeans.cluster_centers_

# Plotting
plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', marker='o', label="Data Points")
plt.scatter(centroids[:, 0], centroids[:, 1], s=200, c='red', marker='x', label="Centroids")
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Updated K-Means Clustering')
plt.legend()
plt.show()

print("Updated centroids after adding new data:")
print(centroids)
