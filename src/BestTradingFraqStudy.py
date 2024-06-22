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


def check_patterns2(patterns_df):
    # Initialize variables
    in_long_position = False  # Track whether we are in a buy position
    buy_price = 0
    total = 0
    trade_count = 0

    # Loop through the DataFrame and process each row
    for idx, row in patterns_df.iterrows():
        label = row['Label']
        price = row['Price']
        
        if label[1] == 'L':
            if not in_long_position:
                # Buy in at this point
                buy_price = price
                in_long_position = True
                if IsDebug:
                    print(f"Buy at {idx}, Price: {buy_price:.2f}")
            else:
                if IsDebug:
                    print(f"Already in position, ignoring buy signal at {idx}, {label}")
                continue
        
        elif label[1] == 'H':
            if in_long_position:
                # Sell out at this point
                sell_price = price
                profit = sell_price - buy_price - cost
                total += profit
                trade_count += 1
                in_long_position = False
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


def check_patterns(patterns_df, IsDebug = True):
    # Initialize variables
    in_long_position = False  # Track whether we are in a buy position
    in_short_position = False 
    buy_price = 0
    buy_time = None
    sell_time = None
    count_long = 0
    #total_hold_time = pd.Timedelta(0)
    hold_times_long = []  # List to store hold times
    wait_times_long = []

    # Loop through the DataFrame and process each row for LONG position(做多)
    for idx, row in patterns_df.iterrows():
        label = row['Label']
        price = row['Price']
        time = idx  # Access the time from the index directly
        
        if label[1] == 'L':
            if not in_long_position:
                # Buy in at this point
                buy_price = price
                buy_time = time
                if buy_time and sell_time:
                    wait_time_long = buy_time - sell_time
                    wait_times_long.append(wait_time_long)
                in_long_position = True
                if IsDebug:
                    print(f"Buy  at {time}, Price: {buy_price:.2f}")
            else:
                if IsDebug:
                    print(f"Already in position, ignoring long signal at {time}, {label}")
                continue
        
        elif label[1] == 'H':
            if in_long_position:
                # Sell out at this point
                sell_price = price
                sell_time = time
                hold_time = sell_time - buy_time
                hold_times_long.append(hold_time)
                #total_hold_time += hold_time
                profit = sell_price - buy_price - cost
                total_profit_long += profit
                count_long += 1
                in_long_position = False
                if IsDebug:
                    print(f"Sell at {time}, Price: {sell_price:.2f}, Profit: {profit:.2f}, Hold Time: {hold_time}")
            else:
                if IsDebug:
                    print(f"Not in position, ignoring sell signal at {time}, {label}")
                continue
        
        else:
            print(f"Error: Not sure how to process this point at {time}, Label: {label}\n")
 
    # Calculate hold time in minute and wait time
    if count_long > 0:
        # Convert total hold time to minutes
        #total_hold_time_minutes = total_hold_time.total_seconds() / 60
        #avg_hold_time_minutes = total_hold_time_minutes / trade_count
        # Calculate median hold time in minutes
        hold_times_minutes_long = [ht.total_seconds() / 60 for ht in hold_times_long]
        median_hold_time_minutes_long = pd.Series(hold_times_minutes_long).median()
        wait_times_minutes_long = [ht.total_seconds() / 60 for ht in wait_times_long]
        median_wait_time_minutes_long = pd.Series(wait_times_minutes_long).median()
    else:
        avg_hold_time_minutes = 0
        median_hold_time_minutes_long = 0
        median_wait_time_minutes_long = 0
        
    
    # Print total profit/loss,  median hold time and median wait time
    if IsDebug:
        print(f"Total Long Profit/Loss: {total_profit_long:.2f}")
        #print(f"Average Hold Time: {avg_hold_time_minutes:.2f} minutes")
        print(f"Median Hold Time: {median_hold_time_minutes_long:.2f} minutes")
        print(f"Median wait Time: {median_wait_time_minutes_long:.2f} minutes")
    
       
    buy_price = 0
    buy_time = None
    sell_time = None
    total_profile_short = 0
    count_short = 0
    hold_times_short = []  # List to store hold times
    wait_times_short = []        
    # Loop through the DataFrame and process each row for SHORT position (做空)
    for idx, row in patterns_df.iterrows():
        label = row['Label']
        price = row['Price']
        time = idx  # Access the time from the index directly
        
        if label[1] == 'H':
            if not in_short_position:
                # Buy in at this point
                buy_price = price
                buy_time = time
                if buy_time and sell_time:
                    wait_time = buy_time - sell_time
                    wait_times_short.append(wait_time)
                in_short_position = True
                if IsDebug:
                    print(f"Buy  at {time}, Price: {buy_price:.2f}")
            else:
                if IsDebug:
                    print(f"Already in position, ignoring short signal at {time}, {label}")
                continue
        
        elif label[1] == 'L':
            if in_short_position:
                # Sell out at this point
                sell_price = price
                sell_time = time
                hold_time = sell_time - buy_time
                hold_times_short.append(hold_time)
                #total_hold_time += hold_time
                profit = -1 * (sell_price - buy_price) - cost
                total_profile_short += profit
                count_short += 1
                in_short_position = False
                if IsDebug:
                    print(f"Sell at {time}, Price: {sell_price:.2f}, Profit: {profit:.2f}, Hold Time: {hold_time}")
            else:
                if IsDebug:
                    print(f"Not in position, ignoring sell signal at {time}, {label}")
                continue
        
        else:
            print(f"Error: Not sure how to process this point at {time}, Label: {label}\n")
    
    # Calculate hold time in minute and wait time
    if count_short > 0:
        # Convert total hold time to minutes
        #total_hold_time_minutes = total_hold_time.total_seconds() / 60
        #avg_hold_time_minutes = total_hold_time_minutes / trade_count
        # Calculate median hold time in minutes
        hold_times_minutes_short = [ht.total_seconds() / 60 for ht in hold_times_short]
        median_hold_time_minutes_short = pd.Series(hold_times_minutes_short).median()
        wait_times_minutes_short = [ht.total_seconds() / 60 for ht in wait_times_short]
        median_wait_time_minutes_short = pd.Series(wait_times_minutes_short).median()
    else:
        avg_hold_time_minutes = 0
        median_hold_time_minutes_short = 0
        median_wait_time_minutes_short = 0
        
    
    # Print total profit/loss,  median hold time and median wait time
    if IsDebug:
        print(f"Total Short Profit/Loss: {total_profile_short:.2f}")
        #print(f"Average Hold Time: {avg_hold_time_minutes:.2f} minutes")
        print(f"Median Hold Time: {median_hold_time_minutes_short:.2f} minutes")
        print(f"Median wait Time: {median_wait_time_minutes_short:.2f} minutes")
    
    # Create a temporary DataFrame with the results
    temp_df = pd.DataFrame({
        'Deviation': [deviation],
        'OHLC_len': [ohlc_len],
        'Zigzag_cnt': [zigzag_count],
        'Median_hold':[median_hold_time],
        'Median_wait':[median_wait_time],
        'Trade_cnt':[trade_count],
        'Total': [total]

    })
    
    return temp_df

    
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
    zigzag_1min_df = zz.calculate_zigzag(ohlc_1min_df, deviation)
    if IsDebug:
        print(f"zigzag_1min list length:{len(zigzag_1min_df)}\n",zigzag_1min_df)
        
    
    # Plot zigzag_1min
    zz.plot_zigzag(ohlc_1min_df, zigzag_1min_df)

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
   
    #Use 5 minutes Zigzag points filter out some 1 minutes points    
    filtered_zigzag_df = filter_zigzag_rough(zigzag_1min, zigzag_1min)
    zigzag_len = len(filtered_zigzag_df)
    if IsDebug:
        print(f"filtered_zigzag_df list length:{len(filtered_zigzag_df)}\n",filtered_zigzag_df) 
     '''          
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
    #patterns = zz.detect_patterns(filtered_zigzag_df)
    zigzag_len = len(zigzag_1min_df)
    patterns = zz.detect_patterns(zigzag_1min_df)
    #for pattern in patterns:
    #    print(f"Datetime: {pattern[0]}, Point: {pattern[1]}, Label: {pattern[2]}")
    if IsDebug:
        print("Patterns list:\n", patterns)

    patterns_df = zz.convert_list_to_df(patterns)
    if IsDebug:
        # Print to verify DataFrame structure
        print(f"Patterns dataframe length:{len(patterns_df)}\n",patterns_df)  

    zz.plot_patterns(ohlc_1min_df, patterns_df)
        
    #total, median_hold_time, median_wait_time, trade_count = check_patterns(patterns_df)
    
    temp_df = check_patterns(patterns_df)
        
    #return ohlc_len, zigzag_len, total, median_hold_time, median_wait_time, trade_count
    return temp_df

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

    db_file = os.path.join(data_dir, "stock_bigdata_2010-2023.db")

    # Cost for each trade
    cost = 5.00
    
    # zigzag_1min parameters
    #deviation = 0.01  # Percentage
    # Try different deviation values
    #deviation_values = [0.0035, 0.0030, 0.0025, 0.0020]
    #deviation_values = [0.0025, 0.0020, 0.0015, 0.0010]
    #deviation_values = [0.0015, 0.0014, 0.0013, 0.0012, 0.0011]
    #deviation_values = [0.001, 0.0008, 0.0007, 0.0006]
    deviation_values = [0.00055, 0.0005, 0.00045, 0.00040]
    #deviation_values = [0.0005, 0.0004, 0.0003]
    #deviation_values = [0.00040, 0.00035, 0.00030]
    #deviation_values = [0.00048, 0.00045, 0.00042]
    #deviation_values = [0.00049, 0.00048, 0.00047, 0.00046]

    start_date = "2023-01-01"
    end_date = "2023-12-31"

    #total = get_total_earning(start_date, end_date, deviation)

    # Initialize an empty DataFrame with specific columns
    df = pd.DataFrame(columns=['Deviation', 'OHLC_len', 'Zigzag_len', 'Total'])

    # Initialize an empty list to store temporary DataFrames
    temp_dfs = []
    
    for deviation in deviation_values:
        # ohlc_len, zigzag_count, total, median_hold_time, median_wait_time, trade_count = get_total_earning(start_date, end_date, deviation)
        # #print(f"Deviation: {deviation}\tOHLC Len:{ohlc_len}\tZigzag Cnt:{zigzag_count}\tTotal:{total:.2f}\tAvg Hold:{avg_hold_time:.2f}\tTrade Cnt:{trade_count}")
        
        # print(f"Deviation: {deviation}\tOHLC Len: {ohlc_len}\tZigzag Cnt: {zigzag_count}\tTotal: {total:.2f}\t"
        #         f"Median Hold: {median_hold_time:.2f}\tMedian Wait: {median_wait_time:0.2f}\tTrade Cnt: {trade_count}")
        # # Create a temporary DataFrame with the results
        # temp_df = pd.DataFrame({
        #     'Deviation': [deviation],
        #     'OHLC_len': [ohlc_len],
        #     'Zigzag_cnt': [zigzag_count],
        #     'Median_hold':[median_hold_time],
        #     'Median_wait':[median_wait_time],
        #     'Trade_cnt':[trade_count],
        #     'Total': [total]
    
        # })

        temp_df = get_total_earning(start_date, end_date, deviation)
        # Append the temporary DataFrame to the list
        temp_dfs.append(temp_df)
    # Concatenate all temporary DataFrames into the main DataFrame
    df = pd.concat(temp_dfs, ignore_index=True)
    
    print(df)
