import yfinance as yf
import pandas as pd
import numpy as np
from zigzag import peak_valley_pivots
import matplotlib.pyplot as plt


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
        #patterns.append((current_point['Close'], label))
        patterns.append((current_point.name, current_point['Close'], label))
    return patterns

if __name__ == "__main__":
    # Data to convert to DataFrame
    data = {
        "Datetime": [
            "2024-06-02 18:00:00-04:00", "2024-06-02 18:01:00-04:00", "2024-06-02 18:02:00-04:00", 
            "2024-06-02 18:03:00-04:00", "2024-06-02 18:04:00-04:00", "2024-06-02 18:05:00-04:00", 
            "2024-06-02 18:06:00-04:00", "2024-06-02 18:07:00-04:00", "2024-06-02 18:08:00-04:00", 
            "2024-06-02 18:09:00-04:00", "2024-06-02 18:10:00-04:00", "2024-06-02 18:11:00-04:00", 
            "2024-06-02 18:12:00-04:00", "2024-06-02 18:13:00-04:00", "2024-06-02 18:14:00-04:00", 
            "2024-06-02 18:15:00-04:00", "2024-06-02 18:16:00-04:00", "2024-06-02 18:17:00-04:00", 
            "2024-06-02 18:18:00-04:00", "2024-06-02 18:19:00-04:00"
        ],
        "Open": [
            5299.25, 5302.75, 5301.50, 5294.25, 5294.25, 5292.50, 5291.00, 5291.75, 
            5292.50, 5292.50, 5292.25, 5292.75, 5292.75, 5293.00, 5294.25, 5294.75, 
            5296.75, 5297.25, 5299.00, 5299.25
        ],
        "High": [
            5304.25, 5303.75, 5301.75, 5295.50, 5295.00, 5292.50, 5292.00, 5292.75, 
            5293.00, 5293.00, 5293.75, 5292.75, 5293.75, 5294.50, 5294.75, 5297.00, 
            5297.50, 5299.75, 5299.25, 5299.75
        ],
        "Low": [
            5298.25, 5301.25, 5293.75, 5292.75, 5292.50, 5290.75, 5290.50, 5291.00, 
            5291.75, 5292.00, 5291.50, 5292.25, 5292.50, 5293.00, 5293.75, 5294.75, 
            5296.25, 5297.00, 5298.50, 5298.75
        ],
        "Close": [
            5302.75, 5301.50, 5294.25, 5294.00, 5292.75, 5291.00, 5292.00, 5292.50, 
            5292.75, 5292.00, 5293.00, 5292.50, 5293.25, 5294.25, 5294.50, 5297.00, 
            5297.25, 5299.00, 5298.75, 5299.25
        ],
        "Volume": [
            0, 691, 2133, 1044, 552, 792, 311, 276, 214, 99, 293, 75, 243, 202, 215, 625, 296, 682, 172, 283
        ]
    }

    # Create DataFrame
    df = pd.DataFrame(data)
    df["Datetime"] = pd.to_datetime(df["Datetime"])
    df.set_index("Datetime", inplace=True)

    print(df.head(20))
    

    
    