import sqlite3
import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

import zigzagplus1 as zz
from logger import Logger
from config import Config

class DataSource:
    config = Config("gru/src/config.ini")
    log = Logger("gru/log/gru.log")
    conn = None

    def __init__(self):
        sqliteDB = DataSource.config.sqlite_db
        data_dir = DataSource.config.data_dir

        # Connect to the SQLite database
        db_file = os.path.join(data_dir, sqliteDB)
        # Connect to the SQLite database
        DataSource.conn = sqlite3.connect(db_file)

    def queryDB(self, query_start, query_end):
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
        self.df.set_index('Datetime', inplace=True)

        DataSource.log.debug(f"Results dataframe length: {len(self.df)}")  
        DataSource.log.debug(f"Data read from table: {table_name}")
        DataSource.log.debug(self.df.head(10))
        DataSource.log.debug(self.df.tail(10))   
        self.zigzag = None

    def getZigzag(self):
        if self.zigzag is None:
            self.zigzag = zz.calculate_zigzag(self.df, float(DataSource.config.deviation))
        return self.zigzag
    
    def getDataFrameFromDB(self):
        return self.df

    def plotDataFrame(self):
        
        # Plot ZigZag
        zz.plot_zigzag(self.df, self.zigzag)

    def plotDataFrameIndex(self):
        self.df.reset_index(drop=True, inplace=True)
        # Plot ZigZag
        zz.plot_zigzag(self.df, self.zigzag)

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
    train_ds.plotDataFrame()
    train_ds.plotPaterns()
    train_ds.plot_prices()

    # plot testing zigzag
    test_ds = DataSource()
    query_start, query_end= DataSource.config.testing_start_date,DataSource.config.testing_end_date
    test_ds.queryDB(query_start, query_end)
    test_ds.getZigzag()
    test_ds.plotDataFrame()
    test_ds.plotPaterns()
 
    DataSource.conn.close()
