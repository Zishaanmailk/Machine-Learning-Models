import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.tree import DecisionTreeRegressor


x = np.array([[i, i+1] for i in range(1, 6)])
y = np.array([5, 7, 9, 11, 13])


model = DecisionTreeRegressor()
model.fit(x, y)


x_range = np.linspace(x[:, 0].min(), x[:, 0].max(), 100)
y_range = np.linspace(x[:, 1].min(), x[:, 1].max(), 100)
x_grid, y_grid = np.meshgrid(x_range, y_range)
grid_points = np.c_[x_grid.ravel(), y_grid.ravel()]


z_grid = model.predict(grid_points).reshape(x_grid.shape)


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x[:, 0], x[:, 1], y, color='blue', label='Data Points')
ax.plot_surface(x_grid, y_grid, z_grid, color='red', alpha=0.5)

ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.set_zlabel('Target')
plt.show()
