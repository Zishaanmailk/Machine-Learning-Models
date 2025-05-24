import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pandas as pd

dat = load_iris()
input = pd.DataFrame(dat.data)
target = pd.DataFrame(dat.target)

mesh = KMeans(n_clusters=3, random_state=0)  # added random_state to suppress warning
mesh.fit(input)
px = mesh.predict(input)
centroids = mesh.cluster_centers_

fig = plt.figure()  # added to properly create 3D plot
ax = fig.add_subplot(111, projection='3d')  # fixed 3D axis creation

ax.scatter(input[0], input[1], input[2], c=px, marker='+', label="Data Points")
ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2], s=200, c='red', marker='*', label="Centroids")
plt.title('Updated K-Means Clustering')
plt.legend()
plt.show()  # added to display the plot
print("Model Score:", mesh.score(input))