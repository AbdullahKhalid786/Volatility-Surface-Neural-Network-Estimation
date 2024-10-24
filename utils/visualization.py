# utils/visualization.py

import matplotlib.pyplot as plt

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
