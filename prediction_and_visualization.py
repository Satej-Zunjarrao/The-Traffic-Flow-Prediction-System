"""
prediction_and_visualization.py
--------------------------------
This module handles traffic flow predictions using trained models and visualizes the results.

Author: Satej
"""

import pandas as pd
import matplotlib.pyplot as plt

def predict_with_arima(model, steps):
    """
    Generate predictions using a trained ARIMA model.
    
    Parameters:
        model: Trained ARIMA model.
        steps (int): Number of future time steps to predict.
    
    Returns:
        pd.Series: Predicted values.
    """
    predictions = model.forecast(steps=steps)
    return predictions

def predict_with_lstm(model, data, time_steps):
    """
    Generate predictions using a trained LSTM model.
    
    Parameters:
        model: Trained LSTM model.
        data (pd.DataFrame): Input features for prediction.
        time_steps (int): Number of time steps in the input sequence.
    
    Returns:
        np.ndarray: Predicted values.
    """
    data = data[-time_steps:].values.reshape(1, time_steps, data.shape[1])  # Reshape for LSTM input
    predictions = model.predict(data)
    return predictions.flatten()

def visualize_predictions(actual, arima_predictions, lstm_predictions, timestamps):
    """
    Visualize actual vs. predicted traffic flow values.
    
    Parameters:
        actual (pd.Series): Actual traffic flow values.
        arima_predictions (pd.Series): Predicted values from ARIMA.
        lstm_predictions (np.ndarray): Predicted values from LSTM.
        timestamps (pd.Series): Corresponding timestamps.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(timestamps, actual, label="Actual", linewidth=2, color="blue")
    plt.plot(timestamps, arima_predictions, label="ARIMA Predictions", linestyle="--", color="green")
    plt.plot(timestamps, lstm_predictions, label="LSTM Predictions", linestyle="--", color="orange")
    plt.title("Traffic Flow Prediction vs Actual Values", fontsize=14)
    plt.xlabel("Timestamp", fontsize=12)
    plt.ylabel("Traffic Flow", fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.show()

# Example usage
if __name__ == "__main__":
    # Load data and trained models
    file_path = "/path/to/engineered_data.csv"  # Replace with actual path, e.g., "satej/data/engineered_data.csv"
    data = pd.read_csv(file_path)
    timestamps = data['timestamp'][-10:]  # Last 10 timestamps for predictions

    # Load trained models
    # For simplicity, placeholders are used; replace with actual model loading mechanisms
    arima_model = None  # Replace with ARIMA model object
    lstm_model = None  # Replace with LSTM model object

    # Generate predictions
    arima_predictions = predict_with_arima(arima_model, steps=10)
    lstm_predictions = predict_with_lstm(lstm_model, data.iloc[:, :-1], time_steps=10)

    # Visualize predictions
    visualize_predictions(data['traffic_flow'][-10:], arima_predictions, lstm_predictions, timestamps)
