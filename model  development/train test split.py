import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Create some sample data
# Assume we're predicting 'y' based on 'x'
data = {
    'x': np.linspace(1, 100, 100),
    'y': np.linspace(10, 1000, 100) + np.random.normal(0, 50, 100)  # Adding noise
}
df = pd.DataFrame(data)

#Define features and target variable
X = df[['x']]  # Feature
y = df['y']    # Target

# Split the data into training and testing sets
# 20% of data is reserved for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate accuracy metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error (MSE):", mse)
print("R-squared (R²):", r2)

# Compare predictions with actual values
comparis = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print("\nComparison of Actual and Predicted values:")
print(comparis.head())