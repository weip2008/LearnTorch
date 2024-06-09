import yfinance as yf
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
    # Create zigzag DataFrame with 'Datetime' and 'Close'
    # zigzag = df[pivots != 0].copy()
   
    # zigzag = zigzag.drop(columns=['Open'])
    # zigzag = zigzag.drop(columns=['High'])
    # zigzag = zigzag.drop(columns=['Low'])
    # zigzag = zigzag.drop(columns=['Volume'])    
                            
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
    return

def convert_list_to_df(patterns):
    # Convert list to DataFrame
    # Specify column names
    columns = ['Datetime', 'Point', 'Label']
    
    # Create DataFrame and set index
    patterns_df = pd.DataFrame(patterns, columns=columns)
    patterns_df['Datetime'] = pd.to_datetime(patterns_df['Datetime'])
    patterns_df.set_index('Datetime', inplace=True)
    return patterns_df

def plot_patterns(df, patterns_df):
    # Plotting to visualize the result
    fig, ax = plt.subplots(figsize=(14, 7))

    ax.plot(df.index, df['Close'], label='Close Price')
    # 1.
    # patterns_df.loc[patterns_df['Label'] == 'HH'].index
    # you are locating rows in patterns_df where the 'Label' column is 'HH' 
    # and then accessing the index of those rows
    ax.scatter(patterns_df.loc[patterns_df['Label'] == 'HH'].index, patterns_df.loc[patterns_df['Label'] == 'HH', 'Point'], color='green', label='HH', marker='^', alpha=1)
    ax.scatter(patterns_df.loc[patterns_df['Label'] == 'LL'].index, patterns_df.loc[patterns_df['Label'] == 'LL', 'Point'], color='red', label='LL', marker='v', alpha=1)
    ax.scatter(patterns_df.loc[patterns_df['Label'] == 'LH'].index, patterns_df.loc[patterns_df['Label'] == 'LH', 'Point'], color='black', label='LH', marker='o', alpha=1)
    ax.scatter(patterns_df.loc[patterns_df['Label'] == 'HL'].index, patterns_df.loc[patterns_df['Label'] == 'HL', 'Point'], color='orange', label='HL', marker='o', alpha=1)

    # 2. 
    # patterns_df.index[patterns_df['Label'] == 'HH']
    # This method first creates a boolean mask with patterns_df['Label'] == 'HH' 
    # and then uses this mask to index patterns_df.index.
    #ax.scatter(patterns_df.index[patterns_df['Label'] == 'HH'], patterns_df['Point'][patterns_df['Label'] == 'HH'], color='green', label='HH', marker='^', alpha=1)
    #ax.scatter(patterns_df.index[patterns_df['Label'] == 'LL'], patterns_df['Point'][patterns_df['Label'] == 'LL'], color='red', label='LL', marker='v', alpha=1)
    #ax.scatter(patterns_df.index[(patterns_df['Label'] == 'LH') | (patterns_df['Label'] == 'HL')], patterns_df['Point'][(patterns_df['Label'] == 'LH') | (patterns_df['Label'] == 'HL')], color='black', label='LH/HL', marker='o', alpha=1)

    ax.set_title('Points with Labels')
    ax.set_xlabel('Datetime')
    ax.set_ylabel('Points')
    ax.legend()

    plt.show()
    return


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
        previous_previous_point = zigzag_points.iloc[i-2]
    
        if current_point['Close'] > previous_previous_point['Close']:
            label = "H"
        else:
            label = "L"
        
        if current_point['Close'] > previous_point['Close']: 
            label += "H"  
        else:
            label += "L"  
        
        label
        #patterns.append((current_point['Close'], label))
        patterns.append((current_point.name, current_point['Close'], label))
    return patterns

def detect_patterns2(zigzag_points):
    """
    Detect patterns like Higher Highs, Higher Lows, Lower Highs, and Lower Lows.

    :param zigzag_points: DataFrame with ZigZag points.
    :return: List of detected patterns.
    """
    patterns = []
    for i in range(1, len(zigzag_points)):
        current_point = zigzag_points.iloc[i]
        previous_point = zigzag_points.iloc[i-1]
        previous_previous_point = zigzag_points.iloc[i-2]
        # 1. Higher High (HH):
        #   If the current closing price is higher than the previous closing price.
        #   Additionally, if the previous closing price is higher than the closing price before it.
        # 2. Higher Low (HL):
        #   If the current closing price is higher than the previous closing price.
        #   However, the previous closing price is lower than the closing price before it.
        # 3. Lower High (LH):
        #   If the current closing price is lower than the previous closing price.
        #   Additionally, if the previous closing price is lower than the closing price before it.
        # 4. Lower Low (LL):
        #   If the current closing price is lower than the previous closing price.
        #   However, the previous closing price is higher than the closing price before it.
        if current_point['Close'] > previous_point['Close']:
            if previous_point['Close'] > previous_previous_point['Close']:
                label = "HH"  # Higher High
            else:
                label = "HL"  # Higher Low
        else:
            if previous_point['Close'] < previous_previous_point['Close']:
                label = "LL"  # Lower Low
            else:
                label = "LH"  # Lower High
        #patterns.append((current_point['Close'], label))
        patterns.append((current_point.name, current_point['Close'], label))
    return patterns

def detect_patterns3(zigzag_points):
    """
    Detect patterns like Higher Highs, Higher Lows, Lower Highs, and Lower Lows.

    :param zigzag_points: DataFrame with ZigZag points.
    :return: List of detected patterns.
    """
    patterns = []
    for i in range(2, len(zigzag_points)):
        current_point = zigzag_points.iloc[i]
        previous_point = zigzag_points.iloc[i - 1]
        previous_previous_point = zigzag_points.iloc[i - 2]

        # Find the valley points
        valleys = zigzag_points[(zigzag_points['Type'] == 'valley') & (zigzag_points.index <= current_point.name)]
        # Find the peak points
        peaks = zigzag_points[(zigzag_points['Type'] == 'peak') & (zigzag_points.index <= current_point.name)]

        if len(valleys) >= 2 and len(peaks) >= 2:
            previous_valley = valleys.iloc[-2]  # Previous valley
            previous_previous_valley = valleys.iloc[-1]  # Previous previous valley
            previous_peak = peaks.iloc[-2]  # Previous peak
            previous_previous_peak = peaks.iloc[-1]  # Previous previous peak

            if current_point['Close'] < previous_valley['Close'] and previous_valley['Close'] < previous_previous_valley['Close']:
                label = "LL"  # Lower Low
            elif current_point['Close'] > previous_peak['Close'] and previous_peak['Close'] > previous_previous_peak['Close']:
                label = "HH"  # Higher High
            elif current_point['Close'] < previous_peak['Close'] and previous_peak['Close'] > previous_previous_peak['Close']:
                label = "LH"  # Lower High
            elif current_point['Close'] > previous_valley['Close'] and previous_valley['Close'] < previous_previous_valley['Close']:
                label = "HL"  # Higher Low

            patterns.append((current_point.name, current_point['Close'], label))

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
    # data = {
    #     'High': [1, 2, 3, 4, 5, 4.5, 3, 3.5, 4, 3],
    #     'Low': [0.5, 1, 1.5, 2, 2.5, 2, 1.5, 2, 2.5, 1.5],
    #     'Close': [0.8, 1.5, 2.5, 3.5, 4.5, 3.5, 2.5, 3, 3.5, 2]
    # }
    # df = pd.DataFrame(data)

    import pandas as pd

    # Data to convert to DataFrame
    # data = {
    #     "Datetime": [
    #         "2024-06-02 18:00:00-04:00", "2024-06-02 18:01:00-04:00", "2024-06-02 18:02:00-04:00", 
    #         "2024-06-02 18:03:00-04:00", "2024-06-02 18:04:00-04:00", "2024-06-02 18:05:00-04:00", 
    #         "2024-06-02 18:06:00-04:00", "2024-06-02 18:07:00-04:00", "2024-06-02 18:08:00-04:00", 
    #         "2024-06-02 18:09:00-04:00", "2024-06-02 18:10:00-04:00", "2024-06-02 18:11:00-04:00", 
    #         "2024-06-02 18:12:00-04:00", "2024-06-02 18:13:00-04:00", "2024-06-02 18:14:00-04:00", 
    #         "2024-06-02 18:15:00-04:00", "2024-06-02 18:16:00-04:00", "2024-06-02 18:17:00-04:00", 
    #         "2024-06-02 18:18:00-04:00", "2024-06-02 18:19:00-04:00"
    #     ],
    #     "Open": [
    #         5299.25, 5302.75, 5301.50, 5294.25, 5294.25, 5292.50, 5291.00, 5291.75, 
    #         5292.50, 5292.50, 5292.25, 5292.75, 5292.75, 5293.00, 5294.25, 5294.75, 
    #         5296.75, 5297.25, 5299.00, 5299.25
    #     ],
    #     "High": [
    #         5304.25, 5303.75, 5301.75, 5295.50, 5295.00, 5292.50, 5292.00, 5292.75, 
    #         5293.00, 5293.00, 5293.75, 5292.75, 5293.75, 5294.50, 5294.75, 5297.00, 
    #         5297.50, 5299.75, 5299.25, 5299.75
    #     ],
    #     "Low": [
    #         5298.25, 5301.25, 5293.75, 5292.75, 5292.50, 5290.75, 5290.50, 5291.00, 
    #         5291.75, 5292.00, 5291.50, 5292.25, 5292.50, 5293.00, 5293.75, 5294.75, 
    #         5296.25, 5297.00, 5298.50, 5298.75
    #     ],
    #     "Close": [
    #         5302.75, 5301.50, 5294.25, 5294.00, 5292.75, 5291.00, 5292.00, 5292.50, 
    #         5292.75, 5292.00, 5293.00, 5292.50, 5293.25, 5294.25, 5294.50, 5297.00, 
    #         5297.25, 5299.00, 5298.75, 5299.25
    #     ],
    #     "Volume": [
    #         0, 691, 2133, 1044, 552, 792, 311, 276, 214, 99, 293, 75, 243, 202, 215, 625, 296, 682, 172, 283
    #     ]
    # }

    # # Create DataFrame
    # df = pd.DataFrame(data)
    # df["Datetime"] = pd.to_datetime(df["Datetime"])
    # df.set_index("Datetime", inplace=True)

    # print(df.head(20))
    
    # Example usage:
    ticker_symbols = [ "MES=F" ]
    # Data interval
    t_interval="1m"

    # Fetch the historical data from the first day it started trading
    stock_data = yf.Ticker("MES=F")
    #stock_hist = stock_data.history(period="max", auto_adjust=False)
    df = stock_data.history(period="max", interval=t_interval, auto_adjust=False)
    # Drop the 'Dividends' and 'Stock Splits' columns
    if 'Dividends' in df.columns:
        df = df.drop(columns=['Dividends'])
    if 'Stock Splits' in df.columns:
        df = df.drop(columns=['Stock Splits'])
    if 'Adj Close' in df.columns:
        df = df.drop(columns=['Adj Close'])    
                            
                    
    print(df.head(20))

    # ZigZag parameters
    deviation = 0.0001  # Percentage

    # Calculate ZigZag
    zigzag = calculate_zigzag(df, deviation)
    print("Zigzag list:\n",zigzag)

    # Plot ZigZag
    plot_zigzag(df, zigzag)

    # Detect patterns
    patterns = detect_patterns(df[df['Close'].isin(zigzag)])
    #for pattern in patterns:
    #    print(f"Datetime: {pattern[0]}, Point: {pattern[1]}, Label: {pattern[2]}")
    #print("Patterns list:\n", patterns)
    
    patterns_df = convert_list_to_df(patterns)
    print(patterns_df)  # Print to verify DataFrame structure

    plot_patterns(df, patterns_df)
    
    # Generate alerts
    # alerts = generate_alerts(df[df['Close'].isin(zigzag)])
    # for alert in alerts:
    #     print(alert)
