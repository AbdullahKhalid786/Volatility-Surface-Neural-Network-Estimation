# src/train.py

import pandas as pd
from sklearn.model_selection import train_test_split

# Import modules
from data.fetch_data import fetch_data
from utils.preprocessing import clean_data, feature_engineering, normalize_data
from models.neural_network import build_model
from utils.visualization import plot_training_loss, plot_predictions, plot_volatility_surface

# Step 1: Fetch the data
calls_df, puts_df = fetch_data()

# Step 2: Preprocess the data
calls_clean, puts_clean = clean_data(calls_df, puts_df)
features = feature_engineering(calls_clean)
scaled_features = normalize_data(features)

# Step 3: Split into training and test sets
X = scaled_features.drop(columns=['lastPrice'])  # Input features
y = scaled_features['lastPrice']  # Target (option price)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Build the model
input_dim = X_train.shape[1]
model = build_model(input_dim)

# Step 5: Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=32)

# Step 6: Evaluate and save the model
loss = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss}")
# Make predictions on the test set
predictions = model.predict(X_test)
plot_training_loss(history)
plot_predictions(y_test, predictions)

# Convert Series to NumPy arrays and ensure they are 1D
y_test_np = y_test.values.flatten()  # Flatten to ensure it is 1D
predictions_np = predictions.flatten()  # Flatten to ensure it is 1D

plot_volatility_surface(y_test, predictions)

model.save('models/option_pricing_model.h5')
