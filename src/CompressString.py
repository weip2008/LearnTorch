import sqlite3
from datetime import datetime
import logging
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np

def load_data(query_start, query_end):
    # Connect to the SQLite database
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()

    # Query the data between the specified start and end dates
    query_range = f'''
    SELECT * FROM {table_name}
    WHERE Datetime BETWEEN ? AND ?
    '''
    # Save the query result into a DataFrame object named query_result_df
    query_result_df = pd.read_sql_query(query_range, conn, params=(query_start, query_end))

    ohlc_df = query_result_df
    ohlc_df['Datetime'] = pd.to_datetime(ohlc_df['Datetime'])
    ohlc_df.set_index('Datetime', inplace=True)

    if IsDebug:
        print("Results dataframe length:", len(ohlc_df))
        print("Data read from table:", table_name)
        print(ohlc_df.head(10))
        print(ohlc_df.tail(10))

    return ohlc_df

def plot_df(ohlc_df):
    """Plot the OHLC dataframe."""
    plt.figure(figsize=(12, 6))
    
    if 'AdjClose' in ohlc_df.columns:
        plt.plot(ohlc_df.index, ohlc_df['AdjClose'], label='Adjusted Close Price', linewidth=1)
    else:
        logging.warning("The dataframe does not contain an 'AdjClose' column to plot.")
        return

    plt.title("OHLC Data")
    plt.xlabel("Datetime")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_pie_1(ohlc_df):
    """Plot the first day's data in polar coordinates."""
    second_day = ohlc_df[ohlc_df.index.date == sorted(set(ohlc_df.index.date))[1]]

    if second_day.empty:
        logging.warning("No data available for the first day.")
        return

    if 'AdjClose' not in second_day.columns:
        logging.warning("The dataframe does not contain an 'AdjClose' column to plot.")
        return

    print("Data for the second day:")
    print(second_day)

    prices = second_day['AdjClose'].values
    times = second_day.index.hour + second_day.index.minute / 60.0  # Fractional hours
    angles = -2 * np.pi * times / 24 + (np.pi / 2) % (2 * np.pi)  # Clockwise and start at top (north)

    plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, polar=True)

    # Plot line chart for the prices
    ax.plot(angles, prices, label='AdjClose Prices', linewidth=3, color='red')

    # Adjust ticks to represent hours on the clock
    hour_ticks = -np.linspace(0, 2 * np.pi, 24, endpoint=False) + (np.pi / 2)
    ax.set_xticks(hour_ticks)
    ax.set_xticklabels(range(0, 24), fontsize=8)  # 0 to 23 hours

    ax.set_title("Polar Plot of Second Day's Adjusted Close Prices", va='bottom')
    plt.legend()
    plt.show()


def plot_clock_face():
    """Plot a clock face with full cycle from 0:00 to 23:59, showing labels at 0, 3, 6, 9, etc."""
    plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, polar=True)

    # Generate hour ticks and angles
    hour_ticks = 2 * np.pi * np.arange(24) / 24 - (np.pi / 2)  # Adjust to start at top (north)  # Clockwise ticks, starting from top
    ax.set_xticks(hour_ticks)
    ax.set_xticklabels([str(i) for i in range(0, 24, 3)], fontsize=10)  # Show every 3 hours

    # Plot the clock face
    ax.set_ylim(0, 1)  # Set radial limits
    ax.set_title("Clock Face (0:00 to 23:59)", va='bottom')
    plt.grid(True)
    plt.show()





def plot_clock_face2():
    """Plot a clock face with a full cycle from 0:00 to 23:59, showing labels at 0, 3, 6, 9, etc."""
    plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, polar=True)

    # Generate hour ticks and angles
    full_circle_ticks = np.linspace(0, 2 * np.pi, 24, endpoint=False)  # Full 24-hour cycle
    ax.set_xticks(full_circle_ticks)

    # Set labels for every 3 hours
    labels = [str(hour) if hour % 3 == 0 else '' for hour in range(24)]
    ax.set_xticklabels(labels, fontsize=10)

    # Plot the clock face
    ax.set_ylim(0, 1)  # Set radial limits
    ax.set_title("Clock Face (0:00 to 23:59)", va='bottom')
    plt.grid(True)
    plt.show()


    

if __name__ == "__main__":
    print(pd.__version__)
    logging.basicConfig(
        level=logging.INFO,  # Set the logging level to INFO or DEBUG
        format=' %(levelname)s => %(message)s'
    )

    IsDebug = True

    # Define the table name as a string variable
    table_name = "SPX_1m"
    
    # Define the SQLite database file directory
    data_dir = "data"

    db_file = os.path.join(data_dir, "stock_data_2024.db")

    testing_start_date = "2024-05-27"
    testing_end_date = "2024-09-22"

    now = datetime.now()
    formatted_now = now.strftime("%Y-%m-%d %H:%M:%S")
    print("Current date and time:", formatted_now)

    ohlc_df = load_data(testing_start_date, testing_end_date)

    # Plot the dataframe
    plot_df(ohlc_df)

    plot_pie_1(ohlc_df)
        
    # Plot the clock face
    plot_clock_face2()
    
    # Plot the first day's data in polar coordinates
    #plot_pie(ohlc_df)
