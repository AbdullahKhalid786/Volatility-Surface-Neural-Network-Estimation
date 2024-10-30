# utils/visualization.py

import matplotlib.pyplot as plt
import numpy as np

def plot_training_loss(history):
    plt.plot(history.history['loss'])
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()

def plot_predictions(y_test, y_pred):
    plt.scatter(y_test, y_pred)
    plt.title('Predicted vs Actual Option Prices')
    plt.xlabel('Actual Prices')
    plt.ylabel('Predicted Prices')
    plt.show()

def plot_volatility_surface(actual_volatility, predicted_volatility):
    # Convert Series to NumPy arrays and flatten
    actual_volatility = actual_volatility.values.flatten()
    predicted_volatility = predicted_volatility.flatten()

    # Calculate closest square grid dimensions
    side_length = int(np.ceil(np.sqrt(len(actual_volatility))))
    reshaped_length = side_length ** 2

    # Pad arrays to fit square shape
    actual_volatility = np.pad(actual_volatility, (0, reshaped_length - len(actual_volatility)), constant_values=np.nan)
    predicted_volatility = np.pad(predicted_volatility, (0, reshaped_length - len(predicted_volatility)), constant_values=np.nan)

    # Reshape for plotting
    actual_volatility = actual_volatility.reshape((side_length, side_length))
    predicted_volatility = predicted_volatility.reshape((side_length, side_length))

    # Plot as before
    X, Y = np.meshgrid(range(actual_volatility.shape[1]), range(actual_volatility.shape[0]))

    fig = plt.figure(figsize=(14, 7))
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot_surface(X, Y, actual_volatility, cmap='viridis')
    ax1.set_title('Actual Volatility Surface')
    ax1.set_xlabel('Strike Price')
    ax1.set_ylabel('Time to Maturity')
    ax1.set_zlabel('Actual Volatility')

    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot_surface(X, Y, predicted_volatility, cmap='inferno')
    ax2.set_title('Predicted Volatility Surface')
    ax2.set_xlabel('Strike Price')
    ax2.set_ylabel('Time to Maturity')
    ax2.set_zlabel('Predicted Volatility')

    plt.tight_layout()
    plt.show()
