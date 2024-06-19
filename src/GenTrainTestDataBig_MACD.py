# This version read source data from SQLite database tables

import sqlite3
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from enum import Enum
import zigzagplus1 as zz

class TradePosition(Enum):
    LONG = 1
    SHORT = -1


# Calculate MACD and Signal Line
def calculate_macd(stock_hist):
    short_ema = stock_hist['Close'].ewm(span=12, adjust=False).mean()
    long_ema = stock_hist['Close'].ewm(span=26, adjust=False).mean()
    macd = short_ema - long_ema
    signal = macd.ewm(span=9, adjust=False).mean()
    macd_histogram = macd - signal
    print(f"MACD list length:{len(macd_histogram)}\n", macd_histogram)

    # Calculate Relative Strength Index (RSI)
    delta = stock_hist['Close'].diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    print(f"RSI list length:{len(rsi)}\n", rsi)

    # Calculate a long-term moving average for trend confirmation
    long_ma = stock_hist['Close'].rolling(window=50).mean()

    # Create a DataFrame to store signals
    signals = pd.DataFrame(index=stock_hist.index)
    signals['MACD'] = macd
    signals['Signal Line'] = signal
    signals['MACD Histogram'] = macd_histogram
    signals['RSI'] = rsi
    signals['Long MA'] = long_ma
    
        
    # Define dynamic thresholds for MACD histogram
    buy_threshold = 0.005
    sell_threshold = -0.005

    # Generate buy/sell signals based on combined criteria
    #based on the MACD Histogram and specific thresholds:
    
    #This is generated when the MACD Histogram crosses above a defined buy_threshold 
    #It indicates a bullish signal based on the MACD histogram moving from below the threshold to above it
    signals['Buy Signal'] = (
        (signals['MACD Histogram'] > buy_threshold) & 
        (signals['MACD Histogram'].shift(1) <= buy_threshold) 
        # shift(1) is previous position
    )
    
    #This is generated when the MACD Histogram crosses below a defined sell_threshold
    #It indicates a bearish signal based on the MACD histogram moving from above the threshold to below it.
    signals['Sell Signal'] = (
        (signals['MACD Histogram'] < sell_threshold) & 
        (signals['MACD Histogram'].shift(1) >= sell_threshold) 
    )
    
    # Add Golden Cross and Death Cross signals using short_ema and long_ema
    signals['Golden Cross'] = (
        (short_ema > long_ema) & 
        (short_ema.shift(1) <= long_ema.shift(1))
    )

    signals['Death Cross'] = (
        (short_ema < long_ema) & 
        (short_ema.shift(1) >= long_ema.shift(1))
    )
    
    if IsDebug:
        print("Combined criteria:\n")
        print(signals.head(10))
        print(signals.tail(10))
        
        # Filter and print Golden Cross and Death Cross signals
        #cross_signals = signals[(signals['Golden Cross']) | (signals['Death Cross'])]
        cross_signals = signals.loc[(signals['Golden Cross']) | (signals['Death Cross']), ['Golden Cross', 'Death Cross']]
        print(f"Golden Cross and Death Cross signals length:{len(cross_signals)}\n", cross_signals)
    
    return cross_signals

def filter_zigzag_exacttime1(zigzag_1min, cross_signals):
    # Ensure the index is datetime if not already
    zigzag_1min.index = pd.to_datetime(zigzag_1min.index)
    cross_signals.index = pd.to_datetime(cross_signals.index)
    
    # Merge the zigzag_1min and cross_signals DataFrames on the Datetime index
    # merge the two DataFrames on their 'Datetime' indices using an inner join, 
    # which ensures only the matching rows are kept.
    merged_df = pd.merge(zigzag_1min, cross_signals, how='inner', left_index=True, right_index=True)
    if IsDebug:
        print(f"Inner join merged dataframe length:{len(merged_df)}\n", merged_df)
    
    # Filter rows where 'Golden Cross' or 'Death Cross' is True
    # filter the merged DataFrame to keep only rows where either 'Golden Cross' or 'Death Cross' is True.
    filtered_zigzag = merged_df[(merged_df['Golden Cross']) | (merged_df['Death Cross'])]
    #print(filtered_zigzag)
    
    # Drop unnecessary columns (Golden Cross and Death Cross)
    filtered_zigzag = filtered_zigzag.drop(columns=['Golden Cross', 'Death Cross'])
    #print(filtered_zigzag)
    
    return filtered_zigzag

def filter_zigzag_exacttime(zigzag_1min, zigzag_5min):
    merged_df = pd.merge(zigzag_1min, zigzag_5min, how='inner', left_index=True, right_index=True)
    if IsDebug:
        print(f"Inner join merged dataframe length:{len(merged_df)}\n", merged_df)
        
    return merged_df

def filter_zigzag_rough2(zigzag_1min, cross_signals, tolerance='5min'):
    print("Starting function")

    # Ensure the indices are datetime if not already
    zigzag_1min.index = pd.to_datetime(zigzag_1min.index)
    cross_signals.index = pd.to_datetime(cross_signals.index)
    
    print("Indices converted to datetime")
    
    # Create a new DataFrame to store the filtered zigzag points
    try:
        # Check if the input is a Series, and convert it to a DataFrame if necessary
        if isinstance(zigzag_1min, pd.Series):
            zigzag_1min = zigzag_1min.to_frame()
        if isinstance(cross_signals, pd.Series):
            cross_signals = cross_signals.to_frame()

        # Check the structure of the DataFrames
        print("zigzag_1min DataFrame:")
        print(zigzag_1min.head())
        print("cross_signals DataFrame:")
        print(cross_signals.head())

        # Create a new DataFrame to store the filtered zigzag points
        filtered_zigzag = pd.DataFrame(columns=zigzag_1min.columns)

        # Iterate through each row in cross_signals
        for cross_time in cross_signals.index:
            print(f"Processing cross signal at {cross_time}")
            
            # Find the closest zigzag point within the tolerance range
            time_window = zigzag_1min.loc[cross_time - pd.Timedelta(tolerance): cross_time + pd.Timedelta(tolerance)]
            print(f"Time window for {cross_time} has {len(time_window)} rows")
            
            if not time_window.empty:
                closest_time = time_window.index[np.argmin(np.abs(time_window.index - cross_time))]
                filtered_zigzag = pd.concat([filtered_zigzag, zigzag_1min.loc[[closest_time]]])
                print(f"Closest zigzag time to {cross_time} is {closest_time}")

        # Remove Duplicates. Ensure unique index after appending
        filtered_zigzag = filtered_zigzag[~filtered_zigzag.index.duplicated(keep='first')]
        
        print("Function completed successfully")
        
        return filtered_zigzag

    except Exception as e:
            print(f"An error occurred: {e}")
    

def filter_zigzag_rough(zigzag_1min, zigzag_5min, tolerance='5min'):

    # Ensure the indices are datetime if not already
    #zigzag_1min.index = pd.to_datetime(zigzag_1min.index)
    #zigzag_5min.index = pd.to_datetime(zigzag_5min.index)
    
    # Create a new DataFrame to store the filtered zigzag points
    try:
        # Check if the input is a Series, and convert it to a DataFrame if necessary
        if isinstance(zigzag_1min, pd.Series):
            zigzag_1min = zigzag_1min.to_frame()
        if isinstance(zigzag_5min, pd.Series):
            zigzag_5min = zigzag_5min.to_frame()

        # Check the structure of the DataFrames
        print("zigzag_1min DataFrame:")
        print(zigzag_1min.head())
        print("zigzag_5min DataFrame:")
        print(zigzag_5min.head())

        # Create a new DataFrame to store the filtered zigzag points
        filtered_zigzag = pd.DataFrame(columns=zigzag_1min.columns)

        # Iterate through each row in cross_signals
        for cross_time in zigzag_5min.index:
            print(f"Processing cross signal at {cross_time}")
            
            # Find the closest zigzag point within the tolerance range
            time_window = zigzag_1min.loc[cross_time - pd.Timedelta(tolerance): cross_time + pd.Timedelta(tolerance)]
            print(f"Time window for {cross_time} has {len(time_window)} rows")
            
            if not time_window.empty:
                closest_time = time_window.index[np.argmin(np.abs(time_window.index - cross_time))]
                filtered_zigzag = pd.concat([filtered_zigzag, zigzag_1min.loc[[closest_time]]])
                print(f"Closest zigzag time to {cross_time} is {closest_time}")

        # Remove Duplicates. Ensure unique index after appending
        filtered_zigzag = filtered_zigzag[~filtered_zigzag.index.duplicated(keep='first')]
        
        print("Function completed successfully")
        
        return filtered_zigzag

    except Exception as e:
            print(f"An error occurred: {e}")
    


def cut_slice(ohlc_1min_df, datetime_index, window_len):
    # Convert the datetime_index to a positional index
    try:
        index = ohlc_1min_df.index.get_loc(datetime_index)
    except KeyError:
        # If the datetime_index is not found in the DataFrame index, return None
        return None
    
    start_index = index - window_len
    # If we don't have enough long data series for this slice, ignore it
    if start_index < 0:
        return None
    
    # Adjust end index to include the last element
    end_index = index + 1
    
    # Create a copy of the section of the original DataFrame
    # Start from start_index up to but not including end_index!
    section_df = ohlc_1min_df.iloc[start_index:end_index].copy()
    section_df.drop(['Open', 'High', 'Low', 'Volume' ], axis=1, inplace=True) 
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


def list_to_string(acceleration_list):
    # Convert each tuple to a string with parentheses and join them with newline characters
    #return ' '.join(['(' + ','.join(map(str, acceleration_tuple)) + '),' for acceleration_tuple in acceleration_list])
    #return ' '.join('(' + ','.join(map(str, acceleration_tuple)) + '),' for acceleration_tuple in acceleration_list)    
    #return ','.join(','.join(map(str, acceleration_tuple)) for acceleration_tuple in acceleration_list)
    #return ','.join(','.join(f'{value:.4f}' for value in acceleration_tuple) for acceleration_tuple in acceleration_list)
    return ','.join(','.join(f'{value}' for value in acceleration_tuple) for acceleration_tuple in acceleration_list)

# Convert training data list to string           
def convert_list_to_string(tddf_list):
    formatted_strings = []
    for section_df in tddf_list:
        formatted_str = "["
        for index, row in section_df.iterrows():
            formatted_str += "({}, {}, {}), ".format(index, row['Price'], row['Volume'])
        formatted_str = formatted_str[:-2]  # Remove the last comma and space
        formatted_str += "]"
        formatted_strings.append(formatted_str)
    return formatted_strings
    
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


def calculate_velocity(processing_df):
    velocity_list = []
    
    # Find the lowest price and corresponding volume in the DataFrame
    #lowest_price, corresponding_volume = find_lowest_price_with_volume(processing_df)
    
    # Normalize 'Volume' and 'Price'
    #processing_df['Normalized_Volume'] = normalize(processing_df['Volume'])
    processing_df['Normalized_Price'] = normalize(processing_df['Close'])
    
    if IsDebug:
        print(processing_df)
    
    for j in range(0, len(processing_df)-1):
        # Extract Price from the current and previous rows
        price_current = processing_df.iloc[j]['Close']
        price_next = processing_df.iloc[j+1]['Close']
        normalized_price_current = processing_df.iloc[j]['Normalized_Price']
        normalized_price_next = processing_df.iloc[j+1]['Normalized_Price']

        #print("Price_current:", Price_current)
        #print("Price_previous:", Price_previous)
        
        #dY = price_current - price_previous
        dY = normalized_price_next - normalized_price_current 
        #print("dY:", dY)
        
        # Extract timestamps from the current and previous rows
        #index_previous = processing_df.index[j - 1]
        index_current = processing_df.index[j]
        index_next = processing_df.index[j+1]
        #print("index_current:", index_current)
        #print("index_previous:", index_next)
        
        #dT = (index_next - index_current) / pd.Timedelta(minutes=1)  
        #dT = index_current - index_previous 
        #dT = (index_next - index_current) / tdLen
        loc_current = processing_df.index.get_loc(index_current)
        loc_next = processing_df.index.get_loc(index_next)

        # Calculate dT based on the difference of locations
        dT = loc_next - loc_current
        #print("dT:", dT)
                
        # Calculate the velocity (dY/dT)
        velocity = dY / dT
        #print("velocity:", velocity)
        
        #datetime_current = processing_df.iloc[j]['Datetime']
        #volume_current = processing_df.iloc[j]['Volume']
        #normalized_volum_current = processing_df.iloc[j]['Normalized_Volume']
        # Append the tuple with the "Velocity" column to tdohlc_df_high_velocity_list
        velocity_list.append((index_current, normalized_price_current, velocity))

    return velocity_list


def calculate_acceleration(velocity_list):
    """
    Calculate acceleration based on a list of tuples containing velocity data.

    Parameters:
    - velocity_list: A list of tuples where each tuple contains velocity data.
                     The tuple structure is assumed to be (index, Price, bb_bbm, velocity).

    Returns:
    - A list of tuples with the "Acceleration" column added.
    """

    acceleration_list = []

    # Iterate over each tuple in velocity_list starting from the second tuple
    for i in range(0, len(velocity_list)-1):
        # Extract velocity data from the current and next tuples
        next_tuple = velocity_list[i+1] 
        current_tuple = velocity_list[i]
        #previous_tuple = velocity_list[i - 1]

        velocity_next = next_tuple[2]
        velocity_current = current_tuple[2]  # velocity is stored at index 2 in the tuple
        #velocity_previous = previous_tuple[2]

        # Calculate the change in velocity
        dV = velocity_next - velocity_current 
        
        #index_current = velocity_list[i].index
        #index_previous = velocity_list[i-1].index
        index_current = i
        index_next = i+1
        i#ndex_previous = i-1
        #dT = index_current - index_previous
        dT = index_next - index_current
        
        # Calculate acceleration (dV/dT)
        acceleration = dV / dT
        
        #current_time = pd.to_datetime(current_tuple[0])
        current_time = current_tuple[0]
        #day_of_week_numeric, time_float = convert_to_day_and_time(index_current)
        day_of_week_numeric, time_float = convert_to_day_and_time(current_time)

        # Append the tuple with the "Acceleration" column to acceleration_list
        #acceleration_list.append((index_current, current_tuple[1], velocity_current, acceleration))
        acceleration_list.append((day_of_week_numeric, time_float, 
                                  current_tuple[1],  current_tuple[2], acceleration))

    return acceleration_list

    # Example usage:
    # acceleration_data = calculate_acceleration(velocity_list)

def write_training_data(TradePosition, acceleration_list, csvfile):
    # Initialize an empty string to store the result
    #result = ""
    
    trainingdata_str = list_to_string(acceleration_list)
    
    # Iterate over each tuple in the acceleration_list
    # for acceleration_tuple in acceleration_list:
    #     # Convert each element of the tuple to a string and concatenate them
    #     result += ",".join(map(str, acceleration_tuple)) 
    
    if (TradePosition is TradePosition.SHORT):        
        result = "0,1," + trainingdata_str + "\n"
        if IsDebug:
            print(result)
        # Parse the input string into separate fields
        #fields = result.split(r',\s*|\)\s*\(', result.strip('[]()'))
        csvfile.write(result)
        return
    
    if (TradePosition is TradePosition.LONG):
        result = "1,0," + trainingdata_str + "\n"
        if IsDebug:
            print(result)
        # Parse the input string into separate fields
        #fields = result.split(r',\s*|\)\s*\(', result.strip('[]()'))
        csvfile.write(result)

    return


def write_testing_data(TradePosition, acceleration_list, csvfile):
    # for testing data, the first number is index of "LONG, SHORT" series!
    # so if it's LONG, SHORT is 1;
    
    trainingdata_str = list_to_string(acceleration_list)
   
    if (TradePosition is TradePosition.LONG):
        result = "0," + trainingdata_str + "\n"
        if IsDebug:
            print(result)
    
        csvfile.write(result)
        return

        
    if (TradePosition is TradePosition.SHORT):        
        result = "1," + trainingdata_str + "\n"
        if IsDebug:
            print(result)
        
        csvfile.write(result)
            
    return



def generate_training_data(tddf_highlow_list, position):
    
    filename = 'stockdata/TrainingDataGenLog_'+ str(position)+".log"
    # Open a file in write mode
    outputfile = open(filename, 'w')
 
    # Initialize an empty list to store tuples with the "Velocity" column
    tddf_velocity_list = []
    tddf_acceleration_list = []
     
    # Iterate over each tuple in tddf_highlow_list starting from the second tuple
    for i in range(0, len(tddf_highlow_list)):
        processing_df = tddf_highlow_list[i]
        if IsDebug:
            print("\ncurrent processing DataFrame size:", len(processing_df), "\n", processing_df)
        
        tddf_velocity_list = calculate_velocity(processing_df)
        if IsDebug:
            print("\nCalculated velocity list length:", len(tddf_velocity_list), "\n",tddf_velocity_list) 
        
        tddf_acceleration_list = calculate_acceleration(tddf_velocity_list)
        if IsDebug:
            print("\nCalculated acceleration list length:", len(tddf_acceleration_list), "\n", tddf_acceleration_list)
        
        if IsDebug:
            print("\nGenerate training data:")
        
        # Write lengths to the file in the desired format
        outputfile.write(
            f"{len(processing_df)},"
            f"{len(tddf_velocity_list)},"
            f"{len(tddf_acceleration_list)}\n"
        ) 
        
        write_training_data(position, tddf_acceleration_list, datafile)
    
    outputfile.close()    
    return

def generate_testing_data(tddf_highlow_list, position):
    
    filename = 'stockdata/TestingDataGenLog_'+ str(position)+".log"
    # Open a file in write mode
    outputfile = open(filename, 'w')
 
    # Initialize an empty list to store tuples with the "Velocity" column
    tddf_velocity_list = []
    tddf_acceleration_list = []
     
    # Iterate over each tuple in tddf_highlow_list starting from the second tuple
    for i in range(0, len(tddf_highlow_list)):
        processing_df = tddf_highlow_list[i]
        if IsDebug:
            print("\ncurrent processing DataFrame size:", len(processing_df), "\n", processing_df)
        
        tddf_velocity_list = calculate_velocity(processing_df)
        if IsDebug:
            print("\nCalculated velocity list length:", len(tddf_velocity_list), "\n",tddf_velocity_list) 
        
        tddf_acceleration_list = calculate_acceleration(tddf_velocity_list)
        if IsDebug:
            print("\nCalculated acceleration list length:", len(tddf_acceleration_list), "\n", tddf_acceleration_list)
        
        if IsDebug:
            print("\nGenerate testing data:")
        
        # Write lengths to the file in the desired format
        outputfile.write(
            f"{len(processing_df)},"
            f"{len(tddf_velocity_list)},"
            f"{len(tddf_acceleration_list)}\n"
        ) 
        
        write_testing_data(position, tddf_acceleration_list, datafile)
    
    outputfile.close()    
    return

def check_patterns(ohlc_1min_df, patterns_df, tdLen):
    
    # filtered_low_points_index = filtered_low_points.index.tolist()
    # filtered_high_points_index = filtered_high_points.index.tolist()

    low_list = []
    high_list = []
    # Loop through the DataFrame and find the first item with the second character of 'L'
    for idx, row in patterns_df.iterrows():
        if row['Label'][1] == 'L':
            #print(f"L point found at {idx}, {row['Label']}")
            section_df = cut_slice(ohlc_1min_df, idx, tdLen+1)
            if (section_df is not None):
                #print("\nSliced DataFrame:\n", section_df)
                low_list.append(section_df) 
            continue
        
        if row['Label'][1] == 'H':
            #print(f"H point found at {idx}, {row['Label']}")
            section_df = cut_slice(ohlc_1min_df, idx, tdLen+1)
            if (section_df is not None):                
                #print("\nSliced DataFrame:\n", section_df)
                high_list.append(section_df) 
            continue
        
        print("Error: Not sure how to process this point!\n")
        
    return low_list, high_list

def gen_highlow_list(query_start, query_end):
    # Connect to the SQLite database
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()

    # Query the data between May 6th, 2024, and May 12th, 2024
    query_range = f'''
    SELECT * FROM {table_name}
    WHERE Datetime BETWEEN ? AND ?
    '''
    # Save the query result into a DataFrame object named query_result_df
    ohlc_1min_df = pd.read_sql_query(query_range, conn, params=(query_start, query_end))

    # print("Length of query result is:", len(query_result_df))
    # print("Datatype of query result:", type(query_result_df))
    # print(query_result_df)

    #ohlc_1min_df = query_result_df
    ohlc_1min_df['Datetime'] = pd.to_datetime(ohlc_1min_df['Datetime'])
    ohlc_1min_df.set_index('Datetime', inplace=True)

    if IsDebug:
        #print("Time elapsed:", time_elapsed, "seconds")
        print("Results dataframe length:", len(ohlc_1min_df))  
        #print("Data read from :", file_path)
        print("Data read from table:", table_name)
        # Print the first few rows of the DataFrame
        print(ohlc_1min_df.head(10))
        print(ohlc_1min_df.tail(10))


    # Calculate zigzag_1min
    zigzag_1min = zz.calculate_zigzag(ohlc_1min_df, deviation)
    if IsDebug:
        print(f"zigzag_1min list length:{len(zigzag_1min)}\n",zigzag_1min)
        
    ohlc_5min_df = ohlc_1min_df.resample('5min').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last'
    })
    
    if IsDebug:
        #print("Time elapsed:", time_elapsed, "seconds")
        print("Results dataframe length:", len(ohlc_5min_df))  
        #print("Data read from :", file_path)
        print(ohlc_5min_df.head(10))
        print(ohlc_5min_df.tail(10))


    # Calculate zigzag_1min
    zigzag_5min = zz.calculate_zigzag(ohlc_5min_df, deviation)
    if IsDebug:
        print(f"zigzag_5min list length:{len(zigzag_5min)}\n",zigzag_5min)
        
    # Plot zigzag_1min
    zz.plot_zigzag(ohlc_1min_df, zigzag_1min)

            
    #cross_signals = calculate_macd(ohlc_1min_df)
    
    # zigzag_counts = df['Close'].value_counts()
    # zigzag_value_counts = zigzag_counts[zigzag_counts.index.isin(zigzag_1min)]
    # print("zigzag_1min value counts:\n", zigzag_value_counts)

    # Filter the original DataFrame using the indices
    # df.loc[zigzag_1min.index]:
    # This expression uses the .loc accessor to select rows from the original DataFrame df.
    # The rows selected are those whose index labels match the index labels of the zigzag_1min DataFrame (or Series).
    # In other words, it filters df to include only the rows where the index (Date) is present in the zigzag_1min index.
    #filtered_zigzag_df = ohlc_1min_df.loc[zigzag_1min.index]
    
    #filtered_zigzag_df = filter_zigzag_exacttime(zigzag_1min, cross_signals)
    # filtered_zigzag_df = filter_zigzag_exacttime(zigzag_1min, zigzag_5min)
    
    # if IsDebug:
    #     print(f"filtered_zigzag_df list length:{len(filtered_zigzag_df)}\n",filtered_zigzag_df)

    #filtered_zigzag_df = filter_zigzag_rough(zigzag_1min, cross_signals)
    filtered_zigzag_df = filter_zigzag_rough(zigzag_1min, zigzag_5min)
    if IsDebug:
        print(f"filtered_zigzag_df list length:{len(filtered_zigzag_df)}\n",filtered_zigzag_df)
              
    # Detect patterns
    # df[df['Close'].isin(zigzag_1min)] creates a new DataFrame 
    # that contains only the rows from df 
    # where the 'Close' value is in the zigzag_1min list.
    # patterns = detect_patterns(df[df['Close'].isin(zigzag_1min)])
    patterns = zz.detect_patterns(filtered_zigzag_df)
    #for pattern in patterns:
    #    print(f"Datetime: {pattern[0]}, Point: {pattern[1]}, Label: {pattern[2]}")
    if IsDebug:
        print("Patterns list:\n", patterns)

    patterns_df = zz.convert_list_to_df(patterns)
    if IsDebug:
        # Print to verify DataFrame structure
        print(f"Patterns dataframe length:{len(patterns_df)}\n",patterns_df)  

    zz.plot_patterns(ohlc_1min_df, patterns_df)
        
    low_list, high_list = check_patterns(ohlc_1min_df, patterns_df, tdLen)
    
    return low_list, high_list

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

    IsDebug = True
    #WindowLen = 5

    #Trainning Data Length
    # average number of working days in a month is 21.7, based on a five-day workweek
    # so 45 days is total for two months working days
    # 200 days is one year working days
    tdLen = 50

    # Series Number for output training/testing data set pairs
    SN = "101"
        
    # zigzag_1min parameters
    deviation = 0.005  # Percentage
        
    symbol = "SPX"
    #symbol = "MES=F"

    # Define the table name as a string variable
    table_name = "SPX_1m"
    #table_name = "MES=F_1m"
    # Define the SQLite database file directory
    data_dir = "data"

    db_file = os.path.join(data_dir, "stock_bigdata_2019-2023.db")

    #============================= Training Data ============================================#
    training_start_date = "2019-01-01"
    training_end_date = "2019-12-31"

    tddf_low_list, tddf_high_list = gen_highlow_list(training_start_date, training_end_date)

    td_file = os.path.join(data_dir, f"{symbol}_TrainingData_{tdLen}_{SN}.csv")

    with open(td_file, "w") as datafile:
        generate_training_data(tddf_low_list, TradePosition.LONG)
        generate_training_data(tddf_high_list, TradePosition.SHORT)


    #============================= Testing Data ============================================#
    testing_start_date = "2019-06-01"
    testing_end_date = "2019-12-31"

    tddf_low_list, tddf_high_list = gen_highlow_list(testing_start_date, testing_end_date)

    td_file = os.path.join(data_dir, f"{symbol}_TestingData_{tdLen}_{SN}.csv")

    with open(td_file, "w") as datafile:
        #generate_training_data(patterns_df)
        generate_testing_data(tddf_low_list, TradePosition.LONG)
        generate_testing_data(tddf_high_list, TradePosition.SHORT)

