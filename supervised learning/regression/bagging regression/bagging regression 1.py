# issu is there



import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
from sklearn.preprocessing import StandardScaler


X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
y = np.array([5, 7, 9, 11, 13])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = BaggingRegressor(estimator=RandomForestRegressor(n_estimators=50), n_estimators=50)
model.fit(X_scaled, y)



z = model.predict(X)
print(z)
z_grid = z.reshape(x_grid.shape)
print(z_grid)


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:, 0], X[:, 1], y, color='blue', label='Data Points')
ax.plot_surface(x_grid, y_grid, z_grid, cmap='viridis', edgecolor='none', alpha=0.7)

ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.set_zlabel('Target')
plt.show()
