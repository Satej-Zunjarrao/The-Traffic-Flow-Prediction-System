"""
utils.py
--------
This module contains helper functions for logging, data validation, 
and metric calculations, which are used across various modules.

Author: Satej
"""

import logging
from sklearn.metrics import mean_absolute_error, mean_squared_error

def setup_logger(log_file_path):
    """
    Set up a logger for the application.
    
    Parameters:
        log_file_path (str): Path to save the log file.
    
    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger("TrafficFlowLogger")
    logger.setLevel(logging.DEBUG)
    
    file_handler = logging.FileHandler(log_file_path)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger

def validate_data(data, required_columns):
    """
    Validate the presence of required columns in the DataFrame.
    
    Parameters:
        data (pd.DataFrame): DataFrame to validate.
        required_columns (list): List of column names that must be present.
    
    Raises:
        ValueError: If any required column is missing.
    """
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

def calculate_metrics(actual, predicted):
    """
    Calculate evaluation metrics (MAE and MSE) for predictions.
    
    Parameters:
        actual (list or pd.Series): Actual values.
        predicted (list or pd.Series): Predicted values.
    
    Returns:
        dict: Dictionary containing MAE and MSE scores.
    """
    mae = mean_absolute_error(actual, predicted)
    mse = mean_squared_error(actual, predicted)
    return {"MAE": mae, "MSE": mse}

# Example usage
if __name__ == "__main__":
    # Set up a logger
    log_path = "/path/to/logfile.log"  # Replace with actual path, e.g., "satej/logs/traffic_flow.log"
    logger = setup_logger(log_path)
    
    logger.info("Logger has been set up.")
    
    # Validate a sample DataFrame
    import pandas as pd
    sample_data = pd.DataFrame({"timestamp": [], "traffic_flow": []})
    try:
        validate_data(sample_data, ["timestamp", "traffic_flow"])
    except ValueError as e:
        logger.error(f"Validation failed: {str(e)}")
    
    # Calculate metrics for example data
    actual = [10, 20, 30, 40, 50]
    predicted = [12, 19, 29, 39, 49]
    metrics = calculate_metrics(actual, predicted)
    logger.info(f"Calculated metrics: {metrics}")
