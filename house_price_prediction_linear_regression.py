
# house_price_prediction_linear_regression.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
# Make sure your CSV has columns like: 'Area', 'Bedrooms', 'Bathrooms', 'Price'
data = pd.read_csv('house_data.csv')

# Drop missing values
data.dropna(inplace=True)

# Features and target
X = data[['Area', 'Bedrooms', 'Bathrooms']]
y = data['Price']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R-squared Score:", r2_score(y_test, y_pred))

# Sample prediction
sample_input = [[1200, 3, 2]]  # Area=1200 sq.ft, 3 Bedrooms, 2 Bathrooms
predicted_price = model.predict(sample_input)
print(f"Predicted Price for {sample_input[0]}: ₹{predicted_price[0]:,.2f}")
