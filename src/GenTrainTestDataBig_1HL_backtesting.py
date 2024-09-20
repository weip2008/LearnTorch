# This version read source data from SQLite database tables

import sqlite3
from datetime import datetime
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

def cut_slice(ohlc_df, start_index, end_index):
    # Ensure the start_index and end_index are in the DataFrame index
    if start_index not in ohlc_df.index or end_index not in ohlc_df.index:
        # If either index is not found, return None
        return None
    
    # Get the positional indices of the timestamps
    start_pos = ohlc_df.index.get_loc(start_index)
    end_pos = ohlc_df.index.get_loc(end_index)
    
    # Ensure start_pos is less than or equal to end_pos
    if start_pos > end_pos:
        return None
    
    # Create a copy of the section of the original DataFrame
    # Start from start_pos up to and including end_pos
    section_df = ohlc_df.iloc[start_pos:end_pos + 1].copy()
    section_df.drop(['Volume'], axis=1, inplace=True)
    
    if IsDebug:
        print (section_df)
        
    return section_df



def write_backtesting_data(TradePosition, processing_df, csvfile, first_write):
    # Decide the file mode and whether to write the header
    if first_write:
        mode = 'w'  # Write mode for the first round
        header = True
    else:
        mode = 'a'  # Append mode for subsequent rounds
        header = False
        
    if (TradePosition is TradePosition.SHORT):        
        # result = "0,1," + backtestingdata_str + "\n"
        # if IsDebug:
        #     print(result)
        # # Parse the input string into separate fields
        # #fields = result.split(r',\s*|\)\s*\(', result.strip('[]()'))
        # csvfile.write(result)
        return
    
    if (TradePosition is TradePosition.LONG):
        if IsDebug:
            print(processing_df)
            plot_prices(processing_df)
    
        # Select the first and last row of data
        result_df = pd.concat([processing_df.iloc[[0]], processing_df.iloc[[-1]]])

        # Add the 'Signal' column with 1 for the first row and -1 for the last row
        result_df['Signal'] = [1, -1]
        
        if IsDebug:
            print(result_df)

        # Append the data to the CSV file, using the header only for the first call
        result_df.to_csv(csvfile, mode=mode, header=header, index=True)
        
    return



def generate_training_data(csvfile, tddf_highlow_list, position):
    # Initialize flag to control whether to write header and not append
    first_write = True

    # Iterate over each tuple in tddf_highlow_list
    for i in range(0, len(tddf_highlow_list)):
        processing_df = tddf_highlow_list[i]

        if IsDebug:
            print("\ncurrent processing DataFrame size:", len(processing_df), "\n", processing_df)

        if IsDebug:
            print("\nGenerate backtesting data:")

        # Write backtesting data, managing the first write flag
        write_backtesting_data(position, processing_df, csvfile, first_write)

        # After the first write, set first_write to False to use append mode next time
        first_write = False

    return


def plot_prices(df):
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
    # ax2 = ax1.twinx()
    # ax2.plot(df.index, df['Normalized_Price'], color='red', label='Normalized Price', linestyle='-', marker='x')
    # ax2.set_ylabel('Normalized Price', color='red')
    # ax2.tick_params(axis='y', labelcolor='red')
    # ax2.set_ylim(df['Normalized_Price'].min(), df['Normalized_Price'].max())

    # Add a legend to differentiate the plots
    # lines_1, labels_1 = ax1.get_legend_handles_labels()
    # lines_2, labels_2 = ax2.get_legend_handles_labels()
    # ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')

    # fig.tight_layout()
    # plt.title('Close Price and Normalized Price')
    # plt.show()
    
    # Add a legend to differentiate the plots
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    ax1.legend(lines_1, labels_1, loc='upper left')

    fig.tight_layout()
    plt.title('Close Price Price')
    plt.show()




def check_patterns(ohlc_df, patterns_df, IsDebug = True):
    short_list = []
    long_list = []
    long_profit= []
    short_profit = []
    
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
                    long_profit.append(np.floor(profit))
                    section_df = cut_slice(ohlc_df, buy_time, sell_time)
                        
                    if (section_df is not None):
                        #print(f"Sliced DataFrame:{len(section_df)}\n", section_df)
                        long_list.append(section_df) 
                        
                    in_long_position = False
                    #previous_point = buy_time
                    if IsDebug:
                        print(f"At {time}, LONG sell price: {sell_price:.2f} at {label} point, Profit: {profit:.2f}, Hold Time: {hold_time}")
                
                    continue
                else:
                    # if profit not > 0 or > 5, just drop this L/H pair
                    in_long_position = False
                    previous_point = buy_time
                    if IsDebug:
                        print(f"At {time}, NO sell at price: {sell_price:.2f} at {label} point, Profit: {profit:.2f}, ignore this pair.")         
            else:
                if IsDebug:
                    print(f"At {time}, Not in position, ignore sell signal at {label} point")
                continue        
        else:
            print(f"Error: Not sure how to process this point at {time}, Label: {label}\n")
    
    total_long_profit = sum(long_profit)
    print(total_long_profit)
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
                    short_profit.append(np.floor(profit)) 
                    section_df = cut_slice(ohlc_df, buy_time, sell_time)
                        
                    if (section_df is not None):
                        #print(f"Sliced DataFrame:{len(section_df)}\n", section_df)
                        short_list.append(section_df) 
                    
                    in_short_position = False
                    #previous_point = buy_time
                    if IsDebug:
                        print(f"At {time}, SHORT buy  price: {sell_price:.2f} at {label} point, Profit: {profit:.2f}, Hold Time: {hold_time}")
                    
                    continue
                else:
                    # if profit not > 0 or > 5, just drop this L/H pair
                    in_short_position = False
                    previous_point = buy_time
                    if IsDebug:
                        print(f"At {time}, NO sell at price: {sell_price:.2f} at {label} point, Profit: {profit:.2f}")         
            else:
                if IsDebug:
                    print(f"Not in position, ignoring sell signal at {time}, {label}")
                continue
        
        else:
            print(f"Error: Not sure how to process this point at {time}, Label: {label}\n")
    
    total_short_profit = sum(short_profit)
    print(total_short_profit)
    
    return short_list, long_list


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
    query_result_df = pd.read_sql_query(query_range, conn, params=(query_start, query_end))

    # print("Length of query result is:", len(query_result_df))
    # print("Datatype of query result:", type(query_result_df))
    # print(query_result_df)

    ohlc_df = query_result_df
    ohlc_df['Datetime'] = pd.to_datetime(ohlc_df['Datetime'])
    ohlc_df.set_index('Datetime', inplace=True)
    #ohlc_df.dropna(subset, inplace=True)

    if IsDebug:
        #print("Time elapsed:", time_elapsed, "seconds")
        print("Results dataframe length:", len(ohlc_df))  
        #print("Data read from :", file_path)
        print("Data read from table:", table_name)
        # Print the first few rows of the DataFrame
        print(ohlc_df.head(10))
        print(ohlc_df.tail(10))


    # Calculate ZigZag
    zigzag = zz.calculate_zigzag(ohlc_df, deviation)
    if IsDebug:
        print(f"Zigzag list length:{len(zigzag)}\n",zigzag)

    # Plot ZigZag
    zz.plot_zigzag(ohlc_df, zigzag)

    # zigzag_counts = df['Close'].value_counts()
    # zigzag_value_counts = zigzag_counts[zigzag_counts.index.isin(zigzag)]
    # print("Zigzag value counts:\n", zigzag_value_counts)

    # Filter the original DataFrame using the indices
    # df.loc[zigzag.index]:
    # This expression uses the .loc accessor to select rows from the original DataFrame df.
    # The rows selected are those whose index labels match the index labels of the zigzag DataFrame (or Series).
    # In other words, it filters df to include only the rows where the index (Date) is present in the zigzag index.
    filtered_zigzag_df = ohlc_df.loc[zigzag.index]
    if IsDebug:
        print(f"filtered_zigzag_df list length:{len(filtered_zigzag_df)}\n",filtered_zigzag_df)

    # Detect patterns
    # df[df['Close'].isin(zigzag)] creates a new DataFrame 
    # that contains only the rows from df 
    # where the 'Close' value is in the zigzag list.
    # patterns = detect_patterns(df[df['Close'].isin(zigzag)])
    patterns = zz.detect_patterns(filtered_zigzag_df)

    # if IsDebug:
    #     print("Patterns list:\n", patterns)

    patterns_df = zz.convert_list_to_df(patterns)
    if IsDebug:
        print(f"Patterns dataframe length:{len(patterns_df)}\n",patterns_df)  # Print to verify DataFrame structure

    zz.plot_patterns(ohlc_df, patterns_df)
        
    short_list, long_list = check_patterns(ohlc_df, patterns_df)
    return short_list, long_list



#
# ================================== MAIN ==============================================#
if __name__ == "__main__":
    print(pd.__version__)
    #logging.basicConfig(level=logging.DEBUG, format='%(levelname)s - %(message)s')
    logging.basicConfig(
        level=logging.INFO,  # Set the logging level to INFO or DEBUG
        #format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        format=' %(levelname)s => %(message)s'
        )

    IsDebug = False

    # Series Number for output training/testing data set pairs
    SN = "101"
        
    # ZigZag parameters
    deviation = 0.0010  # Percentage
        
    #symbol = "SPX"
    #symbol = "MES=F"

    # Define the table name as a string variable
    table_name = "SPX_1m"

    # Define the SQLite database file directory
    data_dir = "data"

    db_file = os.path.join(data_dir, "stock_bigdata_2019-2023.db")
    
    # tradecost for each trade
    longtradecost = 1.00
    shorttradecost = 1.00
    
    #============================= Training Data ============================================#
    training_start_date = "2023-01-01"
    training_end_date = "2023-12-31"

    now = datetime.now()
    formatted_now = now.strftime("%Y-%m-%d %H:%M:%S")
    print("Current date and time:", formatted_now)
    
    tddf_short_list, tddf_long_list = gen_highlow_list(training_start_date, training_end_date)
    #if IsDebug:
    print(f"tddf_short_list length:{len(tddf_short_list)}\n")
    print(f"tddf_long_list length:{len(tddf_long_list)}\n")

    td_file = os.path.join(data_dir, f"{table_name}_Backtesting_{SN}.csv")

    generate_training_data(td_file, tddf_long_list, TradePosition.LONG)
    #generate_training_data(tddf_short_list, TradePosition.SHORT)
        

