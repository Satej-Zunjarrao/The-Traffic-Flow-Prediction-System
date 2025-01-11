"""
dashboard.py
-------------
This module creates a dashboard for real-time traffic flow monitoring, 
allowing stakeholders to visualize and interact with traffic flow predictions.

Author: Satej
"""

import pandas as pd
import dash
from dash import dcc, html
from dash.dependencies import Input, Output

def load_predictions(file_path):
    """
    Load prediction data from a CSV file.
    
    Parameters:
        file_path (str): Path to the CSV file containing predictions.
    
    Returns:
        pd.DataFrame: DataFrame with prediction data.
    """
    try:
        data = pd.read_csv(file_path)
        data['Timestamp'] = pd.to_datetime(data['Timestamp'])
        return data
    except Exception as e:
        raise RuntimeError(f"Failed to load predictions: {str(e)}")

# Load prediction data
predictions_path = "/path/to/predictions.csv"  # Replace with actual path, e.g., "satej/data/predictions.csv"
predictions_data = load_predictions(predictions_path)

# Initialize Dash app
app = dash.Dash(__name__)

# Define layout
app.layout = html.Div([
    html.H1("Traffic Flow Prediction Dashboard", style={"textAlign": "center"}),
    
    dcc.Graph(id="traffic-flow-graph"),
    
    html.Div([
        dcc.Dropdown(
            id="prediction-type-dropdown",
            options=[
                {"label": "ARIMA Predictions", "value": "ARIMA_Predictions"},
                {"label": "LSTM Predictions", "value": "LSTM_Predictions"}
            ],
            value="ARIMA_Predictions",
            placeholder="Select Prediction Type"
        )
    ], style={"width": "50%", "margin": "0 auto", "padding": "10px"})
])

# Define callback for graph update
@app.callback(
    Output("traffic-flow-graph", "figure"),
    [Input("prediction-type-dropdown", "value")]
)
def update_graph(prediction_type):
    """
    Update the traffic flow graph based on the selected prediction type.
    
    Parameters:
        prediction_type (str): Selected prediction type from dropdown (ARIMA or LSTM).
    
    Returns:
        dict: Figure object for the graph.
    """
    if prediction_type not in predictions_data.columns:
        raise ValueError(f"Invalid prediction type: {prediction_type}")
    
    return {
        "data": [
            {"x": predictions_data["Timestamp"], "y": predictions_data["ARIMA_Predictions"], "type": "line", "name": "ARIMA"},
            {"x": predictions_data["Timestamp"], "y": predictions_data["LSTM_Predictions"], "type": "line", "name": "LSTM"}
        ],
        "layout": {
            "title": "Traffic Flow Predictions",
            "xaxis": {"title": "Timestamp"},
            "yaxis": {"title": "Traffic Flow"}
        }
    }

# Run the app
if __name__ == "__main__":
    app.run_server(debug=True, host="0.0.0.0", port=8050)
