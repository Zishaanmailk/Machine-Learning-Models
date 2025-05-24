import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

data = np.array([
    [1, 2,7], [1.5, 1.8,8], [5, 8,9], [8, 8,9],
    [1, 0.6,9], [9, 11,9], [8, 2,6], [10, 2,9],
    [9, 3,9]
])

initial_centers = np.array([[1, 1,9], [5, 5,9]])
kmeans = KMeans(n_clusters=2, init=initial_centers)
kmeans.fit(data)
new_data = np.array([[4, 5,7], [6,8,9]])
data = np.vstack([data, new_data])
kmeans = KMeans(n_clusters=2, init=initial_centers)
kmeans.fit(data)
labels = kmeans.predict(data)
centroids = kmeans.cluster_centers_
ax=plt.axes(projection='3d')
ax.scatter(data[:, 0], data[:, 1],data[:, 2], c=labels, marker='o',label="Data Points")
ax.scatter(centroids[:, 0], centroids[:, 1], s=200, c='red', marker='*', label="Centroids")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
plt.title('Updated K-Means Clustering')
plt.legend()
plt.show()
print("Updated centroids after adding new data:")
print(centroids)