"""
model_training.py
------------------
This module handles training of ARIMA and LSTM models for traffic flow prediction.
It evaluates models using Mean Absolute Error (MAE) and Mean Squared Error (MSE).

Author: Satej
"""

import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

def train_arima_model(data, order=(5, 1, 0)):
    """
    Train an ARIMA model for time-series forecasting.
    
    Parameters:
        data (pd.Series): Univariate time-series data (e.g., traffic_flow).
        order (tuple): ARIMA model order (p, d, q).
    
    Returns:
        model: Trained ARIMA model.
    """
    model = ARIMA(data, order=order)
    fitted_model = model.fit()
    return fitted_model

def train_lstm_model(data, time_steps=10, epochs=10, batch_size=32):
    """
    Train an LSTM model for time-series forecasting.
    
    Parameters:
        data (pd.DataFrame): Traffic data with lag-based features as input.
        time_steps (int): Number of time steps for LSTM input sequence.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
    
    Returns:
        model: Trained LSTM model.
    """
    X = data.iloc[:, :-1].values  # All features except target
    y = data.iloc[:, -1].values  # Target variable
    
    # Reshape for LSTM input (samples, time_steps, features)
    X = X.reshape((X.shape[0], time_steps, X.shape[1]))
    
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(time_steps, X.shape[2])))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=1)
    return model

def evaluate_model(predictions, actual):
    """
    Evaluate model performance using MAE and MSE.
    
    Parameters:
        predictions (list): Predicted values.
        actual (list): Actual values.
    
    Returns:
        dict: Dictionary containing MAE and MSE scores.
    """
    mae = mean_absolute_error(actual, predictions)
    mse = mean_squared_error(actual, predictions)
    return {"MAE": mae, "MSE": mse}

# Example usage
if __name__ == "__main__":
    file_path = "/path/to/engineered_data.csv"  # Replace with actual path, e.g., "satej/data/engineered_data.csv"
    data = pd.read_csv(file_path)
    
    # ARIMA training
    arima_model = train_arima_model(data['traffic_flow'], order=(5, 1, 0))
    arima_predictions = arima_model.forecast(steps=10)
    
    # LSTM training
    lstm_data = data.dropna()  # Ensure no missing values for LSTM
    lstm_model = train_lstm_model(lstm_data, time_steps=10, epochs=20, batch_size=32)
    lstm_predictions = lstm_model.predict(lstm_data.iloc[:, :-1].values)
    
    # Evaluate models
    actual_values = data['traffic_flow'][-10:].values  # Last 10 actual values
    arima_eval = evaluate_model(arima_predictions, actual_values)
    lstm_eval = evaluate_model(lstm_predictions.flatten(), actual_values)
    
    print("ARIMA Evaluation:", arima_eval)
    print("LSTM Evaluation:", lstm_eval)
