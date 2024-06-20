# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Example data (replace with your dataset)
# Let's create a simple example dataset
data = {
    'SquareFeet': [1500, 2000, 1800, 1900, 2100],
    'Bedrooms': [3, 4, 3, 4, 4],
    'Bathrooms': [2, 3, 2, 3, 3],
    'Price': [300000, 400000, 350000, 380000, 420000]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Separate features (SquareFeet, Bedrooms, Bathrooms) and target (Price)
X = df[['SquareFeet', 'Bedrooms', 'Bathrooms']]
y = df['Price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Create a linear regression model
model = LinearRegression()

# Train the model using the training sets
model.fit(X_train, y_train)

# Make predictions using the testing set
y_pred = model.predict(X_test)

# Print model coefficients and intercept
print('Coefficients:', model.coef_)
print('Intercept:', model.intercept_)

# Evaluate the model
print('Mean Squared Error:', mean_squared_error(y_test, y_pred))
print('R-squared:', r2_score(y_test, y_pred))

# Example prediction for a new house
new_house = [[2200, 4, 3]]  # SquareFeet = 2200, Bedrooms = 4, Bathrooms = 3
predicted_price = model.predict(new_house)
print('Predicted Price for new house:', predicted_price)
