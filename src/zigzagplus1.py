import pandas as pd
import numpy as np
from zigzag import peak_valley_pivots
import matplotlib.pyplot as plt

def calculate_zigzag(df, deviation):
    """
    Calculate the ZigZag indicator.

    :param df: DataFrame with 'Close' prices.
    :param deviation: Percentage deviation for ZigZag calculation.
    :return: Series with ZigZag points.
    """
    pivots = peak_valley_pivots(df['Close'].values, deviation, -deviation)
    zigzag = df['Close'][pivots != 0]
    return zigzag

def plot_zigzag(df, zigzag):
    """
    Plot the ZigZag indicator on the close price.

    :param df: DataFrame with 'Close' prices.
    :param zigzag: Series with ZigZag points.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(df['Close'], label='Close Price')
    plt.scatter(zigzag.index, zigzag, color='red', label='ZigZag')
    plt.legend()
    plt.show()

def detect_patterns(zigzag_points):
    """
    Detect patterns like Higher Highs, Higher Lows, Lower Highs, and Lower Lows.

    :param zigzag_points: DataFrame with ZigZag points.
    :return: List of detected patterns.
    """
    patterns = []
    for i in range(1, len(zigzag_points)):
        current_point = zigzag_points.iloc[i]
        previous_point = zigzag_points.iloc[i-1]
        if current_point['Close'] > previous_point['Close']:
            if previous_point['Close'] > zigzag_points.iloc[i-2]['Close']:
                label = "HH"  # Higher High
            else:
                label = "HL"  # Higher Low
        else:
            if previous_point['Close'] < zigzag_points.iloc[i-2]['Close']:
                label = "LL"  # Lower Low
            else:
                label = "LH"  # Lower High
        patterns.append((current_point['Close'], label))
    return patterns

def generate_alerts(zigzag_points):
    """
    Generate alerts based on detected patterns.

    :param zigzag_points: DataFrame with ZigZag points.
    :return: List of alerts.
    """
    alerts = []
    for i in range(1, len(zigzag_points)):
        current_point = zigzag_points.iloc[i]
        previous_point = zigzag_points.iloc[i-1]
        if current_point['Close'] > previous_point['Close'] and previous_point['Close'] > zigzag_points.iloc[i-2]['Close']:
            alerts.append("New Higher High detected")
        elif current_point['Close'] < previous_point['Close'] and previous_point['Close'] < zigzag_points.iloc[i-2]['Close']:
            alerts.append("New Lower Low detected")
        # Add more conditions based on requirements
    return alerts

if __name__ == "__main__":
    # Sample data
    data = {
        'High': [1, 2, 3, 4, 5, 4.5, 3, 3.5, 4, 3],
        'Low': [0.5, 1, 1.5, 2, 2.5, 2, 1.5, 2, 2.5, 1.5],
        'Close': [0.8, 1.5, 2.5, 3.5, 4.5, 3.5, 2.5, 3, 3.5, 2]
    }
    df = pd.DataFrame(data)

    # ZigZag parameters
    deviation = 5 / 100.0  # Percentage

    # Calculate ZigZag
    zigzag = calculate_zigzag(df, deviation)

    # Plot ZigZag
    plot_zigzag(df, zigzag)

    # Detect patterns
    patterns = detect_patterns(df[df['Close'].isin(zigzag)])
    for pattern in patterns:
        print(f"Point: {pattern[0]}, Label: {pattern[1]}")

    # Generate alerts
    alerts = generate_alerts(df[df['Close'].isin(zigzag)])
    for alert in alerts:
        print(alert)
