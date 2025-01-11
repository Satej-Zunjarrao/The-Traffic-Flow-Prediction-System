"""
automation_pipeline.py
-----------------------
This module automates the entire pipeline, including data ingestion, preprocessing, 
model retraining, predictions, and updates for real-time traffic flow prediction.

Author: Satej
"""

import pandas as pd
from datetime import datetime

# Import custom modules
from data_ingestion import load_data, preprocess_data
from feature_engineering import create_temporal_features, create_lag_features, create_rolling_features
from model_training import train_arima_model, train_lstm_model
from prediction_and_visualization import predict_with_arima, predict_with_lstm, visualize_predictions

def automated_pipeline(raw_data_path, output_predictions_path):
    """
    Automates the entire pipeline: data ingestion, preprocessing, feature engineering,
    model training, predictions, and visualization.
    
    Parameters:
        raw_data_path (str): Path to the raw traffic data file.
        output_predictions_path (str): Path to save the predictions.
    """
    # Step 1: Load and preprocess data
    print("Step 1: Loading and preprocessing data...")
    raw_data = load_data(raw_data_path)
    cleaned_data = preprocess_data(raw_data)
    
    # Step 2: Feature engineering
    print("Step 2: Performing feature engineering...")
    engineered_data = create_temporal_features(cleaned_data)
    engineered_data = create_lag_features(engineered_data, lag_features=[1, 2, 3])
    engineered_data = create_rolling_features(engineered_data, window_sizes=[3, 6, 12])

    # Step 3: Train models
    print("Step 3: Training models...")
    arima_model = train_arima_model(engineered_data['traffic_flow'], order=(5, 1, 0))
    lstm_model = train_lstm_model(engineered_data.dropna(), time_steps=10, epochs=10, batch_size=32)

    # Step 4: Generate predictions
    print("Step 4: Generating predictions...")
    arima_predictions = predict_with_arima(arima_model, steps=10)
    lstm_predictions = predict_with_lstm(lstm_model, engineered_data.iloc[:, :-1], time_steps=10)

    # Step 5: Visualize predictions
    print("Step 5: Visualizing predictions...")
    visualize_predictions(
        engineered_data['traffic_flow'][-10:],
        arima_predictions,
        lstm_predictions,
        engineered_data['timestamp'][-10:]
    )

    # Save predictions
    predictions_df = pd.DataFrame({
        "Timestamp": engineered_data['timestamp'][-10:].values,
        "ARIMA_Predictions": arima_predictions,
        "LSTM_Predictions": lstm_predictions
    })
    predictions_df.to_csv(output_predictions_path, index=False)
    print(f"Predictions saved to {output_predictions_path}")

# Example usage
if __name__ == "__main__":
    raw_data_path = "/path/to/raw_traffic_data.csv"  # Replace with actual path, e.g., "satej/data/raw_traffic.csv"
    output_predictions_path = "/path/to/predictions.csv"  # Replace with actual path, e.g., "satej/data/predictions.csv"
    
    automated_pipeline(raw_data_path, output_predictions_path)
