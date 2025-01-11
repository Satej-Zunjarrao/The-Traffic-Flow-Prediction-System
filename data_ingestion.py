"""
data_ingestion.py
-----------------
This module handles the ingestion and preprocessing of historical traffic data.
It includes data loading, cleaning, handling missing values, and formatting timestamps.

Author: Satej
"""

import pandas as pd
import os

def load_data(file_path):
    """
    Load historical traffic data from a CSV file.
    
    Parameters:
        file_path (str): Path to the CSV file containing traffic data.
    
    Returns:
        pd.DataFrame: Loaded data as a pandas DataFrame.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    try:
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        raise RuntimeError(f"Failed to load data from {file_path}: {str(e)}")

def preprocess_data(data):
    """
    Preprocess traffic data to handle missing values and format timestamps.
    
    Parameters:
        data (pd.DataFrame): Raw traffic data.
    
    Returns:
        pd.DataFrame: Preprocessed data.
    """
    # Drop rows with missing timestamps
    if 'timestamp' not in data.columns:
        raise ValueError("Data must contain a 'timestamp' column.")
    
    data['timestamp'] = pd.to_datetime(data['timestamp'], errors='coerce')
    data = data.dropna(subset=['timestamp'])  # Remove rows with invalid timestamps

    # Handle missing traffic values by forward filling
    data = data.sort_values('timestamp').reset_index(drop=True)
    data.fillna(method='ffill', inplace=True)

    return data

def save_clean_data(data, output_path):
    """
    Save the cleaned data to a specified file path.
    
    Parameters:
        data (pd.DataFrame): Cleaned traffic data.
        output_path (str): Path to save the cleaned data.
    """
    try:
        data.to_csv(output_path, index=False)
        print(f"Cleaned data successfully saved to {output_path}")
    except Exception as e:
        raise RuntimeError(f"Failed to save data to {output_path}: {str(e)}")

# Example usage
if __name__ == "__main__":
    file_path = "/path/to/traffic_data.csv"  # Replace with actual path, e.g., "satej/data/raw_traffic.csv"
    output_path = "/path/to/cleaned_traffic_data.csv"  # Replace with actual path, e.g., "satej/data/cleaned_data.csv"
    
    raw_data = load_data(file_path)
    cleaned_data = preprocess_data(raw_data)
    save_clean_data(cleaned_data, output_path)
