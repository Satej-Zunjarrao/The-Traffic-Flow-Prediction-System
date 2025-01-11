"""
eda_and_visualization.py
-------------------------
This module performs exploratory data analysis (EDA) on the traffic data and 
visualizes trends, anomalies, and periodic patterns.

Author: Satej
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_traffic_trends(data):
    """
    Visualize traffic flow trends over time.
    
    Parameters:
        data (pd.DataFrame): Traffic data with 'timestamp' and 'traffic_flow' columns.
    """
    if 'timestamp' not in data.columns or 'traffic_flow' not in data.columns:
        raise ValueError("Data must contain 'timestamp' and 'traffic_flow' columns.")
    
    plt.figure(figsize=(12, 6))
    plt.plot(data['timestamp'], data['traffic_flow'], label="Traffic Flow", linewidth=2)
    plt.title("Traffic Flow Over Time", fontsize=14)
    plt.xlabel("Timestamp", fontsize=12)
    plt.ylabel("Traffic Flow", fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.show()

def visualize_hourly_patterns(data):
    """
    Analyze and visualize hourly traffic patterns.
    
    Parameters:
        data (pd.DataFrame): Traffic data with 'timestamp' and 'traffic_flow' columns.
    """
    if 'timestamp' not in data.columns:
        raise ValueError("Data must contain a 'timestamp' column.")
    
    data['hour'] = data['timestamp'].dt.hour
    hourly_data = data.groupby('hour')['traffic_flow'].mean().reset_index()
    
    plt.figure(figsize=(10, 5))
    sns.barplot(x='hour', y='traffic_flow', data=hourly_data, palette='viridis')
    plt.title("Average Traffic Flow by Hour of the Day", fontsize=14)
    plt.xlabel("Hour of Day", fontsize=12)
    plt.ylabel("Average Traffic Flow", fontsize=12)
    plt.show()

def visualize_weekly_patterns(data):
    """
    Analyze and visualize weekly traffic patterns.
    
    Parameters:
        data (pd.DataFrame): Traffic data with 'timestamp' and 'traffic_flow' columns.
    """
    if 'timestamp' not in data.columns:
        raise ValueError("Data must contain a 'timestamp' column.")
    
    data['day_of_week'] = data['timestamp'].dt.day_name()
    weekly_data = data.groupby('day_of_week')['traffic_flow'].mean().reindex([
        'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'
    ]).reset_index()
    
    plt.figure(figsize=(10, 5))
    sns.barplot(x='day_of_week', y='traffic_flow', data=weekly_data, palette='coolwarm')
    plt.title("Average Traffic Flow by Day of the Week", fontsize=14)
    plt.xlabel("Day of the Week", fontsize=12)
    plt.ylabel("Average Traffic Flow", fontsize=12)
    plt.xticks(rotation=45)
    plt.show()

# Example usage
if __name__ == "__main__":
    file_path = "/path/to/cleaned_traffic_data.csv"  # Replace with actual path, e.g., "satej/data/cleaned_data.csv"
    data = pd.read_csv(file_path)
    data['timestamp'] = pd.to_datetime(data['timestamp'])  # Ensure proper datetime format
    
    visualize_traffic_trends(data)
    visualize_hourly_patterns(data)
    visualize_weekly_patterns(data)
