"""
feature_engineering.py
-----------------------
This module handles feature engineering for the traffic flow prediction project.
It creates temporal features, rolling averages, and categorical features to improve model performance.

Author: Satej
"""

import pandas as pd

def create_temporal_features(data):
    """
    Create temporal features like hour, day of the week, and month.
    
    Parameters:
        data (pd.DataFrame): Traffic data with a 'timestamp' column.
    
    Returns:
        pd.DataFrame: Data with additional temporal features.
    """
    if 'timestamp' not in data.columns:
        raise ValueError("Data must contain a 'timestamp' column.")
    
    data['hour'] = data['timestamp'].dt.hour
    data['day_of_week'] = data['timestamp'].dt.day_name()
    data['month'] = data['timestamp'].dt.month
    return data

def create_lag_features(data, lag_features):
    """
    Create lag-based features for capturing temporal dependencies.
    
    Parameters:
        data (pd.DataFrame): Traffic data with a 'traffic_flow' column.
        lag_features (list): List of integers representing lag intervals (e.g., [1, 2, 3]).
    
    Returns:
        pd.DataFrame: Data with lag features added.
    """
    if 'traffic_flow' not in data.columns:
        raise ValueError("Data must contain a 'traffic_flow' column.")
    
    for lag in lag_features:
        data[f'lag_{lag}'] = data['traffic_flow'].shift(lag)
    return data

def create_rolling_features(data, window_sizes):
    """
    Create rolling average features for smoothing temporal variations.
    
    Parameters:
        data (pd.DataFrame): Traffic data with a 'traffic_flow' column.
        window_sizes (list): List of integers representing rolling window sizes.
    
    Returns:
        pd.DataFrame: Data with rolling average features added.
    """
    if 'traffic_flow' not in data.columns:
        raise ValueError("Data must contain a 'traffic_flow' column.")
    
    for window in window_sizes:
        data[f'rolling_avg_{window}'] = data['traffic_flow'].rolling(window=window).mean()
    return data

# Example usage
if __name__ == "__main__":
    file_path = "/path/to/cleaned_traffic_data.csv"  # Replace with actual path, e.g., "satej/data/cleaned_data.csv"
    data = pd.read_csv(file_path)
    data['timestamp'] = pd.to_datetime(data['timestamp'])  # Ensure proper datetime format
    
    # Apply feature engineering
    data = create_temporal_features(data)
    data = create_lag_features(data, lag_features=[1, 2, 3])  # Lags for 1, 2, and 3 hours
    data = create_rolling_features(data, window_sizes=[3, 6, 12])  # Rolling averages for 3, 6, and 12 hours
    
    output_path = "/path/to/engineered_data.csv"  # Replace with actual path, e.g., "satej/data/engineered_data.csv"
    data.to_csv(output_path, index=False)
    print(f"Feature-engineered data saved to {output_path}")
