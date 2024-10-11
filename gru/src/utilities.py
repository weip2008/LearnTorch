import sqlite3
import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from zigzag import peak_valley_pivots
import pandas_ta as ta

import zigzagplus1 as zz
from logger import Logger
from config import Config

class DataSource:
    config = Config("gru/src/config.ini")
    log = Logger("gru/log/gru.log", logger_name='data')
    conn = None

    def __init__(self):
        sqliteDB = DataSource.config.sqlite_db
        data_dir = DataSource.config.data_dir

        # Connect to the SQLite database
        db_file = os.path.join(data_dir, sqliteDB)
        DataSource.conn = sqlite3.connect(db_file)

    def queryDB(self, query_start, query_end, timeIndex = False):
        table_name = DataSource.config.table_name
        # Query the data between May 6th, 2024, and May 12th, 2024
        query_range = f'''
        SELECT * FROM {table_name}
        WHERE Datetime BETWEEN ? AND ?
        '''

        # Save the query result into a DataFrame object named query_result_df
        query_result_df = pd.read_sql_query(query_range, DataSource.conn, params=(query_start, query_end))
    
        self.df = query_result_df
        self.df['Datetime'] = pd.to_datetime(self.df['Datetime'])
        if timeIndex:
            self.df.set_index('Datetime', inplace=True)
        self.df, self.smooth_column = smooth_sma(self.df,9, True)
        # Drop NaN values from the smoothed column
        self.df = self.df.dropna(subset=[self.smooth_column])

        DataSource.log.debug(f"Results dataframe length: {len(self.df)}")  
        DataSource.log.debug(f"Data read from table: {table_name}")
        self.zigzag = None
        self.hold_zigzag = None
        self.macd()
        return self

    def slice(self):
        slice_len = int(DataSource.config.slice_length)
        DataSource.log.info(f"Slice length: {slice_len}")
        
        start_index = self.df.index[0]
        # Initialize lists for long and short positions
        long_list, short_list, hold_list = [], [], []
        for index, row in self.zigzag.iterrows():
            if index in self.df.index:
                if index < slice_len+start_index: continue
                slice_df = self.get_slice(index, slice_len)
                self.add_to_list(slice_df, row, long_list, short_list, index)

        for index, row in self.hold_zigzag.iterrows():        
            slice_df = self.get_slice(index, slice_len)
            hold_list.append(slice_df)
            
        return long_list, short_list, hold_list
    
   # Helper function to slice df for slice_len rows before the given index
    def get_slice(self, index, slice_len):
        # Use iloc to get slice_len rows from current_position backward
        return self.df.loc[index - slice_len + 1 : index]

    # New helper function to find smaller peaks and valleys
    def find_smaller_peaks_valleys(self, index, zigzag_type):
        smaller_peaks_valleys = []
        
        # Get the relevant part of the zigzag DataFrame before the current index
        relevant_zigzag = self.zigzag[self.zigzag.index < index]
        
        # Check for smaller peaks or valleys
        for small_index, small_row in relevant_zigzag.iterrows():
            if small_row['zigzag_type'] == zigzag_type:
                # Add to the hold_list if it's a smaller peak or valley
                smaller_peaks_valleys.append({'index': small_index, 'data': small_row})
        
        return smaller_peaks_valleys

    # Helper function to add slices to the correct list based on peak/valley type
    def add_to_list(self, slice_df, row, long_list, short_list, index):
        if row['zigzag_type'] == 'peak':
            short_list.append(slice_df)
        elif row['zigzag_type'] == 'valley':
            long_list.append(slice_df)

    def getHoldZigzag(self):
        if self.hold_zigzag is None:
            self.hold_zigzag = self.calculate_zigzag(float(DataSource.config.deviation_hold))
        # Initialize a new column 'zigzag_type'
        self.hold_zigzag['zigzag_type'] = None
    
        # Iterate over the zigzag DataFrame to classify peaks and valleys
        for i in range(len(self.hold_zigzag) - 1):
            if self.hold_zigzag.index[i] not in self.zigzag.index:
                # Check if it's a peak (higher than both neighbors)
                self.hold_zigzag.loc[self.hold_zigzag.index[i], 'zigzag_type'] = 'hold'

        # Filter out rows that do not have 'hold' in zigzag_type
        self.hold_zigzag = self.hold_zigzag[self.hold_zigzag['zigzag_type'] == 'hold']

        return self.hold_zigzag

    def getZigzag(self):
        if self.zigzag is None:
            self.zigzag = self.calculate_zigzag(float(DataSource.config.deviation))

        # Initialize a new column 'zigzag_type'
        self.zigzag['zigzag_type'] = None
    
        # Iterate over the zigzag DataFrame to classify peaks and valleys
        for i in range(len(self.zigzag) - 1):
            current_price = self.zigzag['Close'].iloc[i]
            previous_price = self.zigzag['Close'].iloc[i - 1]
            next_price = self.zigzag['Close'].iloc[i + 1]
            if i==0:
                previous_price = next_price
            
            # Check if it's a peak (higher than both neighbors)
            if current_price > previous_price and current_price > next_price:
                self.zigzag.loc[self.zigzag.index[i], 'zigzag_type'] = 'peak'
            
            # Check if it's a valley (lower than both neighbors)
            elif current_price < previous_price and current_price < next_price:
                self.zigzag.loc[self.zigzag.index[i], 'zigzag_type'] = 'valley'

        return self.zigzag
    
    def calculate_zigzag(self, deviation):
        """
        Calculate the ZigZag indicator.

        :param df: DataFrame with 'Close' prices.
        :param deviation: Percentage deviation for ZigZag calculation.
        :return: Series with ZigZag points.
        """
        pivots = peak_valley_pivots(self.df[self.smooth_column].values, deviation, -deviation)
        zigzag = self.df[self.smooth_column][pivots != 0]
        
        # Convert to DataFrame and rename the column to 'Close'
        zigzag_df = zigzag.to_frame(name='Close')
        
        return zigzag_df

    def macd(self):
        strategy = ta.Strategy(
        name="ModifiedStrategy",
        ta=[
            {"kind": "stochrsi", "length": 70, "rsi_length": 70, "k": 35, "d": 35},  # Default 14 * 5
            {"kind": "macd", "fast": 12, "slow": 26, "signal": 9}#,  # Default (12, 26, 9)
            # {"kind": "macd", "fast": 60, "slow": 130, "signal": 45}  # Default (12, 26, 9) * 5
            ]
        )

        # Make a copy of the DataFrame to avoid the SettingWithCopyWarning
        df_copy = self.df.copy()
        df_copy.ta.strategy(strategy)
        self.df = df_copy  # Update the original DataFrame if needed
        self.df = self.df.dropna(subset=["STOCHRSIk_70_70_35_35"])
        self.df = self.df.dropna(subset=["STOCHRSId_70_70_35_35"])
        self.df = self.df.dropna(subset=["MACD_12_26_9"])
        self.df = self.df.dropna(subset=["MACDh_12_26_9"])
        self.df = self.df.dropna(subset=["MACDs_12_26_9"])
        self.df = self.date2minutes()
    
    def getDataFrameFromDB(self):
            return self.df

    def date2minutes(self):
        tmp = pd.DataFrame(self.df)
        tmp['Datetime'] = pd.to_datetime(tmp['Datetime'])

        # Extract weekday (as an integer where Monday=0, Sunday=6)
        tmp['Weekday'] = tmp['Datetime'].dt.weekday  # Or use df['Datetime'].dt.day_name() for names

        # Convert time to total minutes (hours * 60 + minutes)
        tmp['Time_in_minutes'] = tmp['Datetime'].dt.hour * 60 + tmp['Datetime'].dt.minute

        return tmp

    def plot_zigzag(self):
        """
        Plot the ZigZag indicator on the close price.

        :param df: DataFrame with 'Close' prices.
        :param zigzag: Series with ZigZag points.
        """
        deviation = DataSource.config.deviation
        hlod_deviation = DataSource.config.deviation_hold
        zigzag_len = len(self.zigzag["Close"])
        hold_zigzag_len = len(self.hold_zigzag["Close"])
        plt.figure(figsize=(10, 5))
        plt.plot(self.df[self.smooth_column], label='Close Price')
        plt.plot(self.zigzag.index, self.zigzag["Close"], 'ro-',label=f'Peak/Valley deviation={deviation}')
        plt.plot(self.hold_zigzag.index, self.hold_zigzag["Close"], 'go-',label=f'Hold deviation={hlod_deviation}')
        plt.title(f"ZigZag Indicator on Close Prices: {zigzag_len} peak/valleys; {hold_zigzag_len} holds")
        plt.legend()
        plt.show()

    def plotDataFrameIndex(self):
        self.df.reset_index(drop=True, inplace=True)
        # Plot ZigZag
        self.plot_zigzag(self.df, self.zigzag)

    def plotPaterns(self):
        filtered_zigzag_df = self.df.loc[self.zigzag.index]
        DataSource.log.debug(f"filtered_zigzag_df list length:{len(filtered_zigzag_df)}\n{filtered_zigzag_df}")

        # Detect patterns
        # df[df['Close'].isin(zigzag)] creates a new DataFrame 
        # that contains only the rows from df 
        # where the 'Close' value is in the zigzag list.
        # patterns = detect_patterns(df[df['Close'].isin(zigzag)])
        patterns = zz.detect_patterns(filtered_zigzag_df)

        patterns_df = zz.convert_list_to_df(patterns)
        zz.plot_patterns(self.df, patterns_df)

    def plot_prices(self):#
        """
        Plots the Close price and Normalized price on the same chart with dual y-axes.

        Parameters:
        df (pandas.DataFrame): DataFrame containing 'Close' and 'Normalized_Price' columns.
        """
        self.df['Normalized_Price'] = normalize(self.df['Close'])
        # Plotting
        fig, ax1 = plt.subplots()

        # Plot Close prices
        ax1.plot(self.df.index, self.df['Close'], color='blue', label='Close Price', linestyle='-', marker='o')
        ax1.set_xlabel('Datetime')
        ax1.set_ylabel('Close Price', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')
        ax1.set_ylim(self.df['Close'].min(), self.df['Close'].max())

        # Create a twin y-axis to plot Normalized Price
        ax2 = ax1.twinx()
        ax2.plot(self.df.index, self.df['Normalized_Price'], color='red', label='Normalized Price', linestyle='-', marker='x')
        ax2.set_ylabel('Normalized Price', color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        ax2.set_ylim(self.df['Normalized_Price'].min(), self.df['Normalized_Price'].max())

        # Add a legend to differentiate the plots
        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')

        fig.tight_layout()
        plt.title('Close Price and Normalized Price')
        plt.show() 

def smooth_sma(df, points, center=False):
    # Calculate 9-point smooth (Simple Moving Average)
    column = "Close_SMA_" + str(points)
    df[column] = df["Close"].rolling(window=points, center=center).mean()
    return df, column


 # Normalization function
def normalize(series):
    return (series - series.min()) / (series.max() - series.min())
    
def convert_to_day_and_time(timestamp):
    # Get the day of the week (Monday=0, Sunday=6)
    day_of_week_numeric = timestamp.weekday() + 1

    # Convert the timestamp to a datetime object (to handle timezone)
    dt = timestamp.to_pydatetime()

    # Calculate the time in float format
    time_float = dt.hour + dt.minute / 60 + dt.second / 3600

    return day_of_week_numeric, time_float


def currentTime():
    log = Logger("gru/log/gru.log")
    config = Config("gru/src/config.ini")
    formatted_now = datetime.now().strftime(config.time_format)
    log.info(f'Current date and time: {formatted_now}')

if __name__ == "__main__":
    # plot training zigzag
    train_ds = DataSource()
    query_start, query_end= DataSource.config.training_start_date, DataSource.config.training_end_date
    train_ds.queryDB(query_start, query_end)
    train_ds.getZigzag()
    train_ds.plot_zigzag()
    # train_ds.plotPaterns()
    # train_ds.plot_prices()

    # plot testing zigzag
    test_ds = DataSource()
    query_start, query_end= DataSource.config.testing_start_date,DataSource.config.testing_end_date
    test_ds.queryDB(query_start, query_end)
    # test_ds.getZigzag()
    # test_ds.plot_zigzag()
    # test_ds.plotPaterns()
 
    DataSource.conn.close()
