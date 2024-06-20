# This version read source data from SQLite database tables

import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import zigzagplus1 as zz


def filter_zigzag_rough(zigzag_1min, zigzag_5min, tolerance='5min'):
    try:
        # Check if the input is a Series, and convert it to a DataFrame if necessary
        if isinstance(zigzag_1min, pd.Series):
            zigzag_1min = zigzag_1min.to_frame()
        if isinstance(zigzag_5min, pd.Series):
            zigzag_5min = zigzag_5min.to_frame()

        # Check the structure of the DataFrames
        if IsDebug:
            print("zigzag_1min DataFrame:")
            print(zigzag_1min.head())
            print("zigzag_5min DataFrame:")
            print(zigzag_5min.head())

        # Create a new DataFrame to store the filtered zigzag points
        filtered_zigzag = pd.DataFrame(columns=zigzag_1min.columns)

        # Iterate through each row in cross_signals
        for cross_time in zigzag_5min.index:
            if IsDebug:
                print(f"Processing cross signal at {cross_time}")
            
            # Find the closest zigzag point within the tolerance range
            time_window = zigzag_1min.loc[cross_time - pd.Timedelta(tolerance): cross_time + pd.Timedelta(tolerance)]
            if IsDebug:
                print(f"Time window for {cross_time} has {len(time_window)} rows")
            
            if not time_window.empty:
                closest_time = time_window.index[np.argmin(np.abs(time_window.index - cross_time))]
                new_entry = zigzag_1min.loc[[closest_time]]
                
                if filtered_zigzag.empty:
                    filtered_zigzag = new_entry
                else:
                    filtered_zigzag = pd.concat([filtered_zigzag, new_entry])
                
                if IsDebug:
                    print(f"Closest zigzag time to {cross_time} is {closest_time}")

        # Remove Duplicates. Ensure unique index after appending
        filtered_zigzag = filtered_zigzag[~filtered_zigzag.index.duplicated(keep='first')]
        
        if IsDebug:
            print("Function completed successfully")
        
        return filtered_zigzag

    except Exception as e:
        print(f"An error occurred: {e}")


def filter_zigzag_rough2(zigzag_1min, zigzag_5min, tolerance='5min'):
    
    # Create a new DataFrame to store the filtered zigzag points
    try:
        # Check if the input is a Series, and convert it to a DataFrame if necessary
        if isinstance(zigzag_1min, pd.Series):
            zigzag_1min = zigzag_1min.to_frame()
        if isinstance(zigzag_5min, pd.Series):
            zigzag_5min = zigzag_5min.to_frame()

        # Check the structure of the DataFrames
        if IsDebug:
            print("zigzag_1min DataFrame:")
            print(zigzag_1min.head())
            print("zigzag_5min DataFrame:")
            print(zigzag_5min.head())

        # Create a new DataFrame to store the filtered zigzag points
        filtered_zigzag = pd.DataFrame(columns=zigzag_1min.columns)

        # Iterate through each row in cross_signals
        for cross_time in zigzag_5min.index:
            if IsDebug:
                print(f"Processing cross signal at {cross_time}")
            
            # Find the closest zigzag point within the tolerance range
            time_window = zigzag_1min.loc[cross_time - pd.Timedelta(tolerance): cross_time + pd.Timedelta(tolerance)]
            if IsDebug:
                print(f"Time window for {cross_time} has {len(time_window)} rows")
            
            if not time_window.empty:
                closest_time = time_window.index[np.argmin(np.abs(time_window.index - cross_time))]
                filtered_zigzag = pd.concat([filtered_zigzag, zigzag_1min.loc[[closest_time]]])
                if IsDebug:
                    print(f"Closest zigzag time to {cross_time} is {closest_time}")

        # Remove Duplicates. Ensure unique index after appending
        filtered_zigzag = filtered_zigzag[~filtered_zigzag.index.duplicated(keep='first')]
        
        if IsDebug:
            print("Function completed successfully")
        
        return filtered_zigzag

    except Exception as e:
            print(f"An error occurred: {e}")



def check_patterns(patterns_df):
    # Initialize variables
    in_position = False  # Track whether we are in a buy position
    buy_price = 0
    total = 0

    # Loop through the DataFrame and process each row
    for idx, row in patterns_df.iterrows():
        label = row['Label']
        price = row['Price']
        
        if label[1] == 'L':
            if not in_position:
                # Buy in at this point
                buy_price = price
                in_position = True
                if IsDebug:
                    print(f"Buy at {idx}, Price: {buy_price:.2f}")
            else:
                if IsDebug:
                    print(f"Already in position, ignoring buy signal at {idx}, {label}")
                continue
        
        elif label[1] == 'H':
            if in_position:
                # Sell out at this point
                sell_price = price
                profit = sell_price - buy_price - cost
                total += profit
                in_position = False
                if IsDebug:
                    print(f"Sell at {idx}, Price: {sell_price:.2f}, Profit: {profit:.2f}")
            else:
                if IsDebug:
                    print(f"Not in position, ignoring sell signal at {idx}, {label}")
                continue
        
        else:
            print(f"Error: Not sure how to process this point at {idx}, Label: {label}\n")
    
    # Print total profit/loss with two decimal points
    if IsDebug:
        print(f"Total Profit/Loss: {total:.2f}")
    
    return total

def get_total_earning(query_start, query_end, deviation):
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
    
    ohlc_len = len(ohlc_1min_df)

    if IsDebug:
        print("Results dataframe length:", len(ohlc_1min_df))  
        print("Data read from table:", table_name)
        print(ohlc_1min_df.head(10))
        print(ohlc_1min_df.tail(10))


    # Calculate zigzag_1min
    zigzag_1min = zz.calculate_zigzag(ohlc_1min_df, deviation)
    if IsDebug:
        print(f"zigzag_1min list length:{len(zigzag_1min)}\n",zigzag_1min)
        
    
    # Plot zigzag_1min
    zz.plot_zigzag(ohlc_1min_df, zigzag_1min)

    ''' ohlc_5min_df = ohlc_1min_df.resample('5min').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last'
        })
        
    if IsDebug:
        print("Results dataframe length:", len(ohlc_5min_df))  
        print(ohlc_5min_df.head(10))
        print(ohlc_5min_df.tail(10))


    # Calculate zigzag_5min
    zigzag_5min = zz.calculate_zigzag(ohlc_5min_df, deviation)
    if IsDebug:
        print(f"zigzag_5min list length:{len(zigzag_5min)}\n",zigzag_5min)
    '''
    # Use 5 minutes Zigzag points filter out some 1 minutes points    
    filtered_zigzag_df = filter_zigzag_rough(zigzag_1min, zigzag_1min)
    zigzag_len = len(filtered_zigzag_df)
    if IsDebug:
        print(f"filtered_zigzag_df list length:{len(filtered_zigzag_df)}\n",filtered_zigzag_df) 
              
    # filtered_zigzag_df = zigzag_1min
    # filtered_zigzag_df.index = pd.to_datetime(filtered_zigzag_df.index)
    
    # #filtered_zigzag_df = filter_zigzag_exacttime(zigzag_1min, zigzag_1min)
    # if IsDebug:
    #     print(f"filtered_zigzag_df list length:{len(filtered_zigzag_df)}\n",filtered_zigzag_df)
              
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
        
    total = check_patterns(patterns_df)
        
    return ohlc_len, zigzag_len, total

#
# ================================================================================#
if __name__ == "__main__":
    print(pd.__version__)

    IsDebug = False
    
    symbol = "SPX"
    #symbol = "MES=F"

    # Define the table name as a string variable
    table_name = "SPX_1m"
    #table_name = "MES=F_1m"
    # Define the SQLite database file directory
    data_dir = "data"

    db_file = os.path.join(data_dir, "stock_bigdata_2019-2023.db")

    # Fee for each trade
    cost = 2.00
    
    # zigzag_1min parameters
    deviation = 0.01  # Percentage
    # Try different deviation values
    #deviation_values = [0.0015, 0.001, 0.0009]
    #deviation_values = [0.0008, 0.0007, 0.0006]
    #deviation_values = [0.0005, 0.0004, 0.0003]
    #deviation_values = [0.00055, 0.0005, 0.00045]
    #deviation_values = [0.00040, 0.00035, 0.00030]
    #deviation_values = [0.00048, 0.00045, 0.00042]
    deviation_values = [0.00049, 0.00048, 0.00047, 0.00046]

    # for deviation in deviation_values:
    #     pivots = peak_valley_pivots(df['Close'].values, deviation, -deviation)
    #     df[f'Pivots_{deviation}'] = pivots
 
    #============================= Training Data ============================================#
    start_date = "2022-01-01"
    end_date = "2022-12-31"

    #total = get_total_earning(start_date, end_date, deviation)

    # Initialize an empty DataFrame with specific columns
    df = pd.DataFrame(columns=['Deviation', 'OHLC_len', 'Zigzag_len', 'Total'])

    # Initialize an empty list to store temporary DataFrames
    temp_dfs = []
    
    for deviation in deviation_values:
        ohlc_len, zigzag_len, total = get_total_earning(start_date, end_date, deviation)
        print(f"Deviation: {deviation}\tOHLC len:{ohlc_len}\t\tZigzag points:{zigzag_len}\tTotal:{total:.2f}")
        
        # Create a temporary DataFrame with the results
        temp_df = pd.DataFrame({
            'Deviation': [deviation],
            'OHLC_len': [ohlc_len],
            'Zigzag_len': [zigzag_len],
            'Total': [total]
        })

        # Append the temporary DataFrame to the list
        temp_dfs.append(temp_df)
    # Concatenate all temporary DataFrames into the main DataFrame
    df = pd.concat(temp_dfs, ignore_index=True)
    
    print(df)
