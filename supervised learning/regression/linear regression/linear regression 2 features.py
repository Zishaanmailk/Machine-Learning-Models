import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])


y = np.array([5, 7, 9, 11, 13])  # y = 1x1 + 2x2 + 1


model = LinearRegression()
model.fit(X, y)

# Predict valuesl
y_pred = model.predict(X)

# Print the coefficients (slopes) and intercept
print(f"Slopes (m1, m2): {model.coef_}")
print(f"Intercept (c): {model.intercept_}")

# Plot the 3D graph
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


ax.scatter(X[:, 0], X[:, 1], y, color='blue', label='Original Data')


X1, X2 = np.meshgrid(np.linspace(1, 5, 10), np.linspace(2, 6, 10))

Z = model.intercept_ + model.coef_[0] * X1 + model.coef_[1] * X2


ax.plot_surface(X1, X2, Z, color='red', alpha=0.5, label='Prediction Plane')

# Set labels
ax.set_xlabel('X1 Feature')
ax.set_ylabel('X2 Feature')
ax.set_zlabel('y (Target)')

plt.show()