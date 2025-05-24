import numpy as np
import seaborn as sns
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
# Fixed data for X and y
X = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])
y = np.array([8, 10, 12, 14, 16, 18, 20, 22, 24, 26])
# y = 2x + 6

# Create and fit the linear regression model
model = LinearRegression()
model.fit(X, y)

# Predict values using the model
y_pred = model.predict(X)

# Create a DataFrame for Seaborn
data = pd.DataFrame({'X': X.flatten(), 'y': y, 'y_pred': y_pred})

# Plotting the results using Seaborn
sns.scatterplot(data=data, x='X', y='y', color='blue')
# Actual data points
sns.lineplot(data=data, x='X', y='y_pred', color='red')  # Fitted line
plt.show()
# Show the plot
sns.despine()
# Optional: clean up the plot appearance


