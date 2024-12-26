import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import pickle

# Load data
df = pd.read_csv(r'_YOUR_LOCATION_\house_data.csv')

# Feature selection and engineering
columns = ['bedrooms', 'bathrooms', 'floors', 'yr_built', 'price']
df = df[columns]
df['house_age'] = 2024 - df['yr_built']
df['price'] = np.log1p(df['price'])  # Log transform the target

# Define features and target
X = df[['bedrooms', 'bathrooms', 'floors', 'house_age']]
y = df['price']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.26, random_state=42)

# Train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
print(f"MSE: {mse}, MAE: {mae}")

# Save the model
pickle.dump(model, open('model.pkl', 'wb'))
