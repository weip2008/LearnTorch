# This version read source data from SQLite database tables
#
# [1,0] stand for sell
# [0,1] stand for buy
# [0,1,1,0] test data means 4 set of [sell, buy, buy, sell]

import sqlite3
from datetime import datetime
from datetime import timedelta
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from enum import Enum
import zigzagplus1 as zz
import statistics

class TradePosition(Enum):
    LONG = 1
    HOLD = 0
    SHORT = -1

import pandas as pd
import sqlite3

import pandas as pd
import sqlite3

def cut_slice_sql(ohlc_df, table_name, db_file):
    # List to store the sliced data
    data_slices_list = []
    
    # Connect to the SQLite database
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    
    # Loop through each row in ohlc_df
    for idx, row in ohlc_df.iterrows():
        # Define the current day's 15:00 timestamp as the end point
        end_point = row.name  # This is the Datetime index from ohlc_df
        
        # Calculate the start point as the previous day's 18:00
        start_point = end_point.replace(hour=18, minute=0, second=0) - pd.Timedelta(days=1)
        
        # Convert the start_point and end_point to string format suitable for SQLite
        start_point_str = start_point.strftime('%Y-%m-%d %H:%M:%S')
        end_point_str = end_point.strftime('%Y-%m-%d %H:%M:%S')
        
        # Query the data between start_point and end_point
        query = f'''
        SELECT * FROM {table_name}
        WHERE Datetime BETWEEN ? AND ?
        '''
        # Execute the query and store the result in a DataFrame
        result_df = pd.read_sql_query(query, conn, params=(start_point_str, end_point_str))
        
        # Drop the "Volume" column if it exists
        if 'Volume' in result_df.columns:
            result_df.drop(columns=['Volume'], inplace=True)
            
        # Drop any rows with NaN values
        result_df.dropna(inplace=True)
        
        # Convert the index to a Datetime column, and then set it back as the index
        result_df['Datetime'] = pd.to_datetime(result_df['Datetime'])
        result_df.set_index('Datetime', inplace=True)
        
        print("Results dataframe length:", len(result_df))
        print("Data read from table:", table_name)

        # Print the first few rows of the DataFrame
        print(result_df.head(20))
        print(result_df.tail(20))
        
        # Resample data into 10-minute intervals
        agg_df = result_df.resample('10min').agg({
            'Open': 'first',   # Use the first value in each 10-minute window as Open
            'High': 'max',     # Use the maximum value in each 10-minute window as High
            'Low': 'min',      # Use the minimum value in each 10-minute window as Low
            'Close': 'last'    # Use the last value in each 10-minute window as Close
        })
        
         # Print the first few rows of the DataFrame
        print("Results dataframe length:", len(agg_df))
        print(agg_df.head(10))
        print(agg_df.tail(10))
        
        # If the result is not empty, add it to the data_slices_list
        if not agg_df.empty:
            data_slices_list.append(agg_df)
    
    # Close the database connection
    conn.close()
    
    return data_slices_list

# Example usage:
# data_slices = cut_slice_sql(ohlc_df, "SPX_5m", "data/stock_bigdata_2010-2023.db")



def cut_slice(ohlc_df, end_index, traintest_data_len):
    # Ensure the start_index and end_index are in the DataFrame index
    if end_index not in ohlc_df.index:
    #if start_index not in ohlc_df.index or end_index not in ohlc_df.index:
        # If either index is not found, return None
        return None
    
    # Get the positional indices of the timestamps
    end_pos = ohlc_df.index.get_loc(end_index)  
    #start_pos = ohlc_df.index.get_loc(start_index)
    start_pos = end_pos - traintest_data_len 
    end_pos = end_pos + 2    # ADD one more step after reach H/L point!!! add one more for Python nature
    
    # Ensure start_pos is less than or equal to end_pos
    if start_pos < 0 or start_pos > end_pos:
        return None
    
    # Create a copy of the section of the original DataFrame
    # Start from start_pos up to and including end_pos
    #section_df = ohlc_df.iloc[start_pos:end_pos].copy()
    #try:
    section_df = ohlc_df.iloc[int(start_pos):int(end_pos)].copy()
    #except Exception as e:
    #    print(e)    
    
    # for last section, maybe not enough data for a slice
    if (len(section_df) < (traintest_data_len+2)):   
        return None
   
    section_df.drop(['Open', 'High', 'Low', 'Volume'], axis=1, inplace=True)
    
    return section_df


def gen_hold_list_index(df):
    """
    Generates new index numbers by inserting two integers evenly between consecutive index numbers.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        list: List of newly generated index numbers.
    """
    index_column = df.index
    new_index = []

    for i in range(len(index_column) - 1):
        steps = (index_column[i+1] - index_column[i]) // 3
        new_index.append(index_column[i]+steps)
        new_index.append(index_column[i]+steps*2)

    # Add the last index
    new_index.append(index_column[-1])

    return new_index 


def list_to_string1(price_list):
    return ', '.join(map(str, price_list))

def list_to_string2(price_list):
    return '[' + ', '.join(map(str, price_list)) + ']'

    
def convert_to_day_and_time(timestamp):
    # Get the day of the week (Monday=0, Sunday=6)
    day_of_week_numeric = timestamp.weekday() + 1

    # Convert the timestamp to a datetime object (to handle timezone)
    dt = timestamp.to_pydatetime()

    # Calculate the time in float format
    time_float = dt.hour + dt.minute / 60 + dt.second / 3600

    return day_of_week_numeric, time_float

# Normalization function
def normalize(series):
    return (series - series.min()) / (series.max() - series.min())



def gen_list(processing_df):
    price_list = []

    processing_df['Normalized_Price'] = normalize(processing_df['Close'])
    
    if IsDebug:
        print(processing_df)
        plot_prices(processing_df)
    
    for j in range(0, len(processing_df)):

        normalized_price_current = processing_df.iloc[j]['Normalized_Price']
        #price_current = processing_df.iloc[j]['Close']
        index_current = processing_df.index[j]
        
        #price_list.append((index_current, normalized_price_current))
        price_list.append((normalized_price_current))
        #price_list.append((price_current))

    return price_list

    # Example usage:
    # acceleration_data = calculate_acceleration(velocity_list)

def write_traintest_data(TradePosition, trainingdata_str, csvfile):

    if (TradePosition is TradePosition.SHORT):        
        #result = "-1," + trainingdata_str + "\n"
        result = f"{TradePosition.SHORT.value}," + trainingdata_str + "\n"
        if IsDebug:
            print(result)
        csvfile.write(result)
        return
    
    if (TradePosition is TradePosition.LONG):
        #result = "1," + trainingdata_str + "\n"
        result = f"{TradePosition.LONG.value}," + trainingdata_str + "\n"
        if IsDebug:
            print(result)
        csvfile.write(result)

    return


def write_testing_data(TradePosition, trainingdata_str, csvfile):
    # for testing data, the first number is index of "LONG, SHORT" series!
    # so if it's LONG, SHORT is 1;
    
    #trainingdata_str = list_to_string(data_list)
   
    if (TradePosition is TradePosition.LONG):
        result = "1," + trainingdata_str + "\n"
        if IsDebug:
            print(result)
    
        csvfile.write(result)
        return

        
    if (TradePosition is TradePosition.SHORT):        
        result = "-1," + trainingdata_str + "\n"
        if IsDebug:
            print(result)
        
        csvfile.write(result)
            
    return

# Function to generate training data as a string
def generate_training_data_string(processing_df):
    training_data_str = "["  # Start the string with an opening bracket
    
    for index, row in processing_df.iterrows():
        # Extract the current Datetime
        current_time = pd.to_datetime(index)
        
        # Call the conversion function
        day_of_week_numeric, time_float = convert_to_day_and_time(current_time)
        
        # Form the 5-element tuple as a string
        data_tuple_str = f"({day_of_week_numeric}, {time_float}, {row['Normalized_Price']}, {row['Velocity']}, {row['Acceleration']})"
        
        # Append the tuple string to the main string
        training_data_str += data_tuple_str + ", "
    
    # Remove the last comma and space, and close the string with a closing bracket
    training_data_str = training_data_str.rstrip(", ") + "]"
    
    return training_data_str

def generate_training_data(tddf_highlow_list, position):
    
    filename = 'stockdata/TrainingDataGenLog_'+ str(position)+".log"
    # Open a file in write mode
    outputfile = open(filename, 'w')
     
    # Iterate over each tuple in tddf_highlow_list starting from the second tuple
    for i in range(0, len(tddf_highlow_list)):
        processing_df = tddf_highlow_list[i]
        
        # 1. Use the normalize function to add the "Normalized_Price" column
        processing_df['Normalized_Price'] = normalize(processing_df['Close'])
        # 2. Add the "Velocity" column by calculating the difference of the "Normalized_Price" column
        processing_df['Velocity'] = processing_df['Normalized_Price'].diff()
        # 3. Add the "Acceleration" column by calculating the difference of the "Velocity" column
        processing_df['Acceleration'] = processing_df['Velocity'].diff()
        # Check out the updated DataFrame
        #print(processing_df.head())        
        processing_df = processing_df.dropna()
        #print(processing_df.head())
        
        # Generate the training data as a string
        training_data_string = generate_training_data_string(processing_df)

        # Print the final result
        if IsDebug:
            print(training_data_string)
        
                
        if IsDebug:
            print("\nGenerate training/testing data:")
        
        # Write lengths to the file in the desired format
        outputfile.write(
            f"{len(processing_df)}\n"
        ) 
        
        write_traintest_data(position, training_data_string, datafile)
    
        #print(i)
    outputfile.close()    
    return

def generate_testing_data(tddf_highlow_list, position):
    
    filename = 'stockdata/TestingDataGenLog_'+ str(position)+".log"
    # Open a file in write mode
    outputfile = open(filename, 'w')
 
    # Iterate over each tuple in tddf_highlow_list starting from the second tuple
    for i in range(0, len(tddf_highlow_list)):
        processing_df = tddf_highlow_list[i]
        # 1. Use the normalize function to add the "Normalized_Price" column
        processing_df['Normalized_Price'] = normalize(processing_df['Close'])
        # 2. Add the "Velocity" column by calculating the difference of the "Normalized_Price" column
        processing_df['Velocity'] = processing_df['Normalized_Price'].diff()
        # 3. Add the "Acceleration" column by calculating the difference of the "Velocity" column
        processing_df['Acceleration'] = processing_df['Velocity'].diff()
        # Check out the updated DataFrame
        #print(processing_df.head())        
        processing_df = processing_df.dropna()
        #print(processing_df.head())
        
        # Generate the training data as a string
        testing_data_string = generate_training_data_string(processing_df)

        # Print the final result
        if IsDebug:
            print(testing_data_string)
                        
        if IsDebug:
            print("\nGenerate training/testing data:")
        
        # Write lengths to the file in the desired format
        outputfile.write(
            f"{len(processing_df)}\n"
        ) 
        
        write_traintest_data(position, testing_data_string, datafile)
    
        #print(i)
    outputfile.close()    
    return


def plot_prices(df):#
    """
    Plots the Close price and Normalized price on the same chart with dual y-axes.

    Parameters:
    df (pandas.DataFrame): DataFrame containing 'Close' and 'Normalized_Price' columns.
    """
    # Plotting
    fig, ax1 = plt.subplots()

     # Plot Close prices
    ax1.plot(df.index, df['Close'], color='blue', label='Close Price', linestyle='-', marker='o')
    ax1.set_xlabel('Datetime')
    ax1.set_ylabel('Close Price', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.set_ylim(df['Close'].min(), df['Close'].max())

    # Create a twin y-axis to plot Normalized Price
    ax2 = ax1.twinx()
    ax2.plot(df.index, df['Normalized_Price'], color='red', label='Normalized Price', linestyle='-', marker='x')
    ax2.set_ylabel('Normalized Price', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.set_ylim(df['Normalized_Price'].min(), df['Normalized_Price'].max())

    # Add a legend to differentiate the plots
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')

    fig.tight_layout()
    plt.title('Close Price and Normalized Price')
    plt.show() 
    
    return
  

def check_patterns_length(ohlc_df, patterns_df, traintest_data_len, IsDebug=False):
    short_list = []
    long_list = []
    median_long_hold_time = 0
    median_short_hold_time = 0
    
    # Initialize variables
    in_long_position = False  # Track whether we are in a buy position
    #previous_point = None
    buy_time = None
    sell_time = None
    hold_time = None
    holdtime_list = []
    # Loop through the DataFrame and process each row for LONG position(做多)
    # essentially, "buying low and selling high" is the strategy for a long position.    
    for idx, row in patterns_df.iterrows():
        label = row['Label']
        price = row['Price']
        time = idx  # Access the time from the index directly
        
        start_pos = ohlc_df.index.get_loc(idx)
        if start_pos < traintest_data_len:
            continue
        
        if label[1] == 'L':
            if not in_long_position:
                # Buy in at this point
                buy_price = price
                buy_time = idx
                in_long_position = True
                if IsDebug:
                    print(f"At {time}, LONG buy  price: {buy_price:.2f} at {label} point")
            else:
                if IsDebug:
                    print(f"At {time}, already in long position, ignoring signal {label} at price: {buy_price:.2f}")
                continue
        
        elif label[1] == 'H':
            if in_long_position:
                # Sell out at this point
                sell_price = price
                sell_time = idx
                hold_time = sell_time - buy_time                
                profit = sell_price - buy_price - longtradecost
                if profit > 0: 
                    hold_time_in_minutes = int(hold_time.total_seconds() / 60)
                    holdtime_list.append(hold_time_in_minutes)
                    #section_df = cut_slice(ohlc_df, buy_time, sell_time)
                    # section_df = cut_slice(ohlc_df, sell_time)
                    
                        
                    # if (section_df is not None):
                    #     #print(f"Sliced DataFrame:{len(section_df)}\n", section_df)
                    #     long_list.append(section_df) 
                        
                    in_long_position = False
                    if IsDebug:
                        print(f"At {time}, LONG sell price: {sell_price:.2f} at {label} point, Profit: {profit:.2f}, Hold Time: {hold_time}")
                
                    continue
                else:
                    # if profit not > 0, just drop this L/H pair
                    in_long_position = False
                    if IsDebug:
                        print(f"At {time}, NO sell at price: {sell_price:.2f} at {label} point, Profit: {profit:.2f}, ignore this pair.")         
            else:
                if IsDebug:
                    print(f"At {time}, Not in position, ignore sell signal at {label} point")
                continue        
        else:
            print(f"Error: Not sure how to process this point at {time}, Label: {label}\n")

    # Mean hold time
    mean_hold_time = statistics.mean(holdtime_list)

    # Median hold time
    median_hold_time = statistics.median(holdtime_list)

    # Standard deviation
    std_dev_hold_time = statistics.stdev(holdtime_list)

    print("Long:")
    print("Mean Hold Time:", mean_hold_time)
    print("Median Hold Time:", median_hold_time)
    print("Standard Deviation:", std_dev_hold_time)
    
    long_hold_time = int(np.ceil(mean_hold_time))
    
    if IsDebug:
        print("\n\n=======================================================================\n\n")
        
    # essentially, "selling high and buying low" is the strategy for a short position.    
    in_short_position = False
    previous_point = None
    buy_time = None
    sell_time = None      
    hold_time = None  
    # Loop through the DataFrame and process each row for SHORT position (做空)
    for idx, row in patterns_df.iterrows():
        label = row['Label']
        price = row['Price']
        time = idx  # Access the time from the index directly
        
        start_pos = ohlc_df.index.get_loc(idx)
        if start_pos < traintest_data_len:
            continue
        
        if label[1] == 'H':
            if not in_short_position:
                # Buy in at this point
                buy_price = price
                buy_time = time
                in_short_position = True
                if IsDebug:
                    print(f"At {time}, SHORT sell price: {buy_price:.2f} at {label} point")
            else:
                in_short_position = False
                if IsDebug:
                    print(f"At {time}, already in SHORT position, ignoring signal {label} at price: {buy_price:.2f}. Ignore this pair.")
                continue
        
        elif label[1] == 'L':
            if in_short_position:
                # Sell out at this point
                sell_price = price
                sell_time = idx
                hold_time = sell_time - buy_time                   
                profit = -1 * (sell_price - buy_price) - shorttradecost
                if profit > 0: 
                    hold_time_in_minutes = int(hold_time.total_seconds() / 60)
                    holdtime_list.append(hold_time_in_minutes)
                    # section_df = cut_slice(ohlc_df, sell_time)
                    # #section_df = cut_slice(ohlc_df, buy_time, sell_time)
                        
                    # if (section_df is not None):
                    #     print(f"Sliced DataFrame:{len(section_df)}\n", section_df)
                    #     short_list.append(section_df) 
                    
                    in_short_position = False
                    if IsDebug:
                        print(f"At {time}, SHORT buy  price: {sell_price:.2f} at {label} point, Profit: {profit:.2f}, Hold Time: {hold_time}")
                    
                    continue
                else:
                    # if profit not > 0 , just drop this L/H pair
                    in_short_position = False
                    if IsDebug:
                        print(f"At {time}, NO sell at price: {sell_price:.2f} at {label} point, Profit: {profit:.2f}")         
            else:
                if IsDebug:
                    #print(f"Not in position, ignoring sell signal at {time}, {label}")
                    print(f"At {time}, Not in position, ignore sell signal at {label} point")
                continue
        
        else:
            print(f"Error: Not sure how to process this point at {time}, Label: {label}\n")

    # Mean hold time
    mean_hold_time = statistics.mean(holdtime_list)

    # Median hold time
    median_hold_time = statistics.median(holdtime_list)

    # Standard deviation
    std_dev_hold_time = statistics.stdev(holdtime_list)

    print("Short:")
    print("Mean Hold Time:", mean_hold_time)
    print("Median Hold Time:", median_hold_time)
    print("Standard Deviation:", std_dev_hold_time)
    
    short_hold_time = int(np.ceil(mean_hold_time))
    
    avg_hold_time = int((short_hold_time + long_hold_time) / 2 )
    print("Avg mean Hold Time:", avg_hold_time)
    
    return avg_hold_time

def check_long_patterns(ohlc_df, patterns_df, long_traintest_data_len, IsDebug=False):
    long_list = []
    
    # Initialize variables
    in_long_position = False  # Track whether we are in a buy position
    #previous_point = None
    buy_time = None
    sell_time = None
    hold_time = None  
    # Loop through the DataFrame and process each row for LONG position(做多)
    # essentially, "buying low and selling high" is the strategy for a long position.    
    for idx, row in patterns_df.iterrows():
        label = row['Label']
        price = row['Price']
        time = idx  # Access the time from the index directly
        
        start_pos = ohlc_df.index.get_loc(idx)
        if start_pos < traintest_data_len:
            continue
        
        if label[1] == 'L':
            if not in_long_position:
                # Buy in at this point
                buy_price = price
                buy_time = idx
                in_long_position = True
                if IsDebug:
                    print(f"At {time}, LONG buy  price: {buy_price:.2f} at {label} point")
            else:
                if IsDebug:
                    print(f"At {time}, already in long position, ignoring signal {label} at price: {buy_price:.2f}")
                continue
        
        elif label[1] == 'H':
            if in_long_position:
                # Sell out at this point
                sell_price = price
                sell_time = idx
                hold_time = sell_time - buy_time                
                profit = sell_price - buy_price - longtradecost
                if profit > 0: 
                    #section_df = cut_slice(ohlc_df, buy_time, sell_time)
                    section_df = cut_slice(ohlc_df, sell_time, long_traintest_data_len)                    
                        
                    if (section_df is not None):
                        if IsDebug:
                            print(f"Sliced DataFrame:{len(section_df)}\n", section_df)
                        long_list.append(section_df) 
                        
                    in_long_position = False
                    if IsDebug:
                        print(f"At {time}, LONG sell price: {sell_price:.2f} at {label} point, Profit: {profit:.2f}, Hold Time: {hold_time}")
                
                    continue
                else:
                    # if profit not > 0, just drop this L/H pair
                    in_long_position = False
                    if IsDebug:
                        print(f"At {time}, NO sell at price: {sell_price:.2f} at {label} point, Profit: {profit:.2f}, ignore this pair.")         
            else:
                if IsDebug:
                    print(f"At {time}, Not in position, ignore sell signal at {label} point")
                continue        
        else:
            print(f"Error: Not sure how to process this point at {time}, Label: {label}\n")
    
    return long_list


def check_short_patterns(ohlc_df, patterns_df, short_traintest_data_len, IsDebug=False):
    short_list = []
        
    # essentially, "selling high and buying low" is the strategy for a short position.    
    in_short_position = False
    previous_point = None
    buy_time = None
    sell_time = None      
    hold_time = None  
    # Loop through the DataFrame and process each row for SHORT position (做空)
    for idx, row in patterns_df.iterrows():
        label = row['Label']
        price = row['Price']
        time = idx  # Access the time from the index directly
        
        start_pos = ohlc_df.index.get_loc(idx)
        if start_pos < traintest_data_len:
            continue
        
        if label[1] == 'H':
            if not in_short_position:
                # Buy in at this point
                buy_price = price
                buy_time = time
                in_short_position = True
                if IsDebug:
                    print(f"At {time}, SHORT sell price: {buy_price:.2f} at {label} point")
            else:
                in_short_position = False
                if IsDebug:
                    print(f"At {time}, already in SHORT position, ignoring signal {label} at price: {buy_price:.2f}. Ignore this pair.")
                continue
        
        elif label[1] == 'L':
            if in_short_position:
                # Sell out at this point
                sell_price = price
                sell_time = idx
                hold_time = sell_time - buy_time                   
                profit = -1 * (sell_price - buy_price) - shorttradecost
                if profit > 0: 
                    section_df = cut_slice(ohlc_df, sell_time, short_traintest_data_len)
                    #section_df = cut_slice(ohlc_df, buy_time, sell_time)
                        
                    if (section_df is not None):
                        if IsDebug:
                            print(f"Sliced DataFrame:{len(section_df)}\n", section_df)
                        short_list.append(section_df) 
                    
                    in_short_position = False
                    if IsDebug:
                        print(f"At {time}, SHORT buy  price: {sell_price:.2f} at {label} point, Profit: {profit:.2f}, Hold Time: {hold_time}")
                    
                    continue
                else:
                    # if profit not > 0 , just drop this L/H pair
                    in_short_position = False
                    if IsDebug:
                        print(f"At {time}, NO sell at price: {sell_price:.2f} at {label} point, Profit: {profit:.2f}")         
            else:
                if IsDebug:
                    #print(f"Not in position, ignoring sell signal at {time}, {label}")
                    print(f"At {time}, Not in position, ignore sell signal at {label} point")
                continue
        
        else:
            print(f"Error: Not sure how to process this point at {time}, Label: {label}\n")

    return short_list

def load_data(query_start, query_end):
    # Connect to the SQLite database
    conn = sqlite3.connect(db_file)
    #cursor = conn.cursor()

    # Query for records where the time is 15:00 each day between the specified date range
    query_time_1500 = f'''
    SELECT * FROM {table_name}
    WHERE strftime('%H:%M', Datetime) = '15:00'
    AND Datetime BETWEEN ? AND ?
    '''

    # Query for records where the time is between 14:57 and 15:03 each day
    # query_time_range = f'''
    # SELECT * FROM {table_name}
    # WHERE strftime('%H:%M', Datetime) BETWEEN '14:56' AND '15:04'
    # AND Datetime BETWEEN ? AND ?
    # '''

    # Save the query result into a DataFrame object named query_result_df
    #query_result_df = pd.read_sql_query(query_time_1500, conn, params=(query_start, query_end))
    query_result_df = pd.read_sql_query(query_time_1500, conn, params=(query_start, query_end))

    ohlc_df = query_result_df
    ohlc_df['Datetime'] = pd.to_datetime(ohlc_df['Datetime'])
    ohlc_df.set_index('Datetime', inplace=True)

    print("Results dataframe length:", len(ohlc_df))
    print("Data read from table:", table_name)

    # Print the first few rows of the DataFrame
    print(ohlc_df.head(20))
    print(ohlc_df.tail(20))

    # Close the database connection
    conn.close()    

    return ohlc_df
    

#
# ================================================================================#
if __name__ == "__main__":
    print(pd.__version__)
    #logging.basicConfig(level=logging.DEBUG, format='%(levelname)s - %(message)s')
    logging.basicConfig(
        level=logging.INFO,  # Set the logging level to INFO or DEBUG
        #format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        format=' %(levelname)s => %(message)s'
        )

    IsDebug = False

    #Trainning Data Length
    # average number of working days in a month is 21.7, based on a five-day workweek
    # so 45 days is total for two months working days
    # 200 days is one year working days
    traintest_data_len = 60
 
    # Series Number for output training/testing data set pairs
    SN = "800"
        
    # ZigZag parameters
    deviation = 0.0010  # Percentage
    #deviation = 0.002  # Percentage
        
    symbol = "SPX"
    #symbol = "MES=F"

    # Define the table name as a string variable
    table_name = "SPX_1m"
    #table_name = "MES=F_1m"
    # Define the SQLite database file directory
    data_dir = "data"

    #db_file = os.path.join(data_dir, "stock_bigdata_2019-2023.db")
    db_file = os.path.join(data_dir, "stock_bigdata_2010-2023.db")
    
    # tradecost for each trade
    longtradecost = 1.00
    shorttradecost = 1.00
    
    #============================= Training Data ============================================#
    training_start_date = "2010-11-14"
    training_end_date = "2023-12-30"

    now = datetime.now()
    formatted_now = now.strftime("%Y-%m-%d %H:%M:%S")
    print("Current date and time:", formatted_now)
    
    ohlc_df = load_data(training_start_date, training_end_date)
    
    # Example usage:
    data_slices = cut_slice_sql(ohlc_df, table_name, db_file)
    
    patterns_df = []
    traintest_data_len = check_patterns_length(ohlc_df, patterns_df, 60)    
    tddf_long_list = check_long_patterns(ohlc_df, patterns_df, traintest_data_len)
    tddf_short_list = check_short_patterns(ohlc_df, patterns_df, traintest_data_len)
    
    if IsDebug:
        print(f"tddf_short_list length:{len(tddf_short_list)}\n")
        print(f"tddf_long_list length:{len(tddf_long_list)}\n")

    td_file = os.path.join(data_dir, \
        f"{table_name}_TrainingData_HL_{traintest_data_len}_{SN}.txt")
    
    print(td_file)

    with open(td_file, "w") as datafile:
        generate_training_data(tddf_short_list, TradePosition.LONG)
        generate_training_data(tddf_long_list, TradePosition.SHORT)

    #============================= Testing Data ============================================#
    testing_start_date = "2023-03-01"
    testing_end_date = "2023-09-31"
    
    now = datetime.now()
    formatted_now = now.strftime("%Y-%m-%d %H:%M:%S")
    print("Current date and time:", formatted_now)

    ohlc_df = load_data(testing_start_date, testing_end_date)
    
    #short_traintest_data_len, long_traintest_data_len = check_patterns_length(ohlc_df, patterns_df, 60)    
    tddf_long_list = check_long_patterns(ohlc_df, patterns_df, traintest_data_len)
    tddf_short_list = check_short_patterns(ohlc_df, patterns_df, traintest_data_len)
    
    if IsDebug:
        print(f"tddf_short_list length:{len(tddf_short_list)}\n")
        print(f"tddf_long_list length:{len(tddf_long_list)}\n")
        
    td_file = os.path.join(data_dir, \
        f"{table_name}_TestingData_HL_{traintest_data_len}_{SN}.txt")
    print(td_file)

    with open(td_file, "w") as datafile:
        generate_testing_data(tddf_short_list, TradePosition.LONG)
        generate_testing_data(tddf_long_list, TradePosition.SHORT)

    now = datetime.now()
    formatted_now = now.strftime("%Y-%m-%d %H:%M:%S")
    print("Current date and time:", formatted_now)