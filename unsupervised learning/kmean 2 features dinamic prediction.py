import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans


initial_data = np.array([
    [1, 2], [1.5, 1.8], [5, 8], [8, 8],
    [1, 0.6], [9, 11], [8, 2], [10, 2],
    [9, 3], [6, 9]
])

# Initial cluster centroids
initial_centroids = np.array([[1, 1], [5, 5], [9, 9]])

# Initialize MiniBatchKMeans with 3 clusters and pre-set centroids
kmeans = MiniBatchKMeans(n_clusters=3, init=initial_centroids, n_init=1, batch_size=1, max_iter=10)
kmeans.fit(initial_data)


new_data_points = np.array([[4, 5], [6, 9], [3, 4], [7, 8], [2, 3]])


for new_point in new_data_points:
    kmeans.partial_fit([new_point])  # Update model with new point

    # Plot the clusters and centroids
plt.scatter(initial_data[:, 0], initial_data[:, 1], c=kmeans.predict(initial_data), cmap='viridis')
plt.scatter(new_point[0], new_point[1], c='orange', marker='o', s=100, label="New Point")
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=200, c='red', marker='x', label="Centroids")
plt.legend()
plt.show()