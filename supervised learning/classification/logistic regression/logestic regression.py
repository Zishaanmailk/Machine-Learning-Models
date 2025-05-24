import numpy as np
import seaborn as sns
import pandas as pd
from sklearn.linear_model import LogisticRegression

# Fixed data for X (single feature) and binary target y
X = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])
y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
 # Binary outcome

# Create and fit the logistic regression model
model = LogisticRegression()
model.fit(X, y)

# Predict values using the model
y_pred = model.predict(X)

# Create a DataFrame for Seaborn
data = pd.DataFrame({'X': X.flatten(), 'y': y, 'y_pred': y_pred})

# Plotting the results using Seaborn
sns.scatterplot(data=data, x='X', y='y', color='blue')
# Actual data points
sns.scatterplot(data=data, x='X', y='y_pred', color='red')  # Predicted points

# New input for prediction
new_value = np.array([[7.5]])
  # Example new value
new_prediction =model.predict(new_value)

# Displaying the new prediction
print(f'Prediction for input {new_value[0][0]}: {new_prediction[0]}')

# Show the plot
sns.despine()
  # Optional: clean up the plot appearance