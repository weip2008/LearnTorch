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


def list_to_string(price_list):
    return ','.join(map(str, price_list))

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


''' 
def gen_list(processing_df):
    price_list = []

    #processing_df['Normalized_Price'] = normalize(processing_df['Close'])
    
    if IsDebug:
        print(processing_df)
        plot_prices(processing_df)
    
    for j in range(0, len(processing_df)):

        #normalized_price_current = processing_df.iloc[j]['Normalized_Price']
        price_current = processing_df.iloc[j]['Close']
        index_current = processing_df.index[j]
        
        #price_list.append((index_current, normalized_price_current))
        #price_list.append((normalized_price_current))
        price_list.append((price_current))

    return price_list
 '''
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

    # Add a legend to differentiate the plots
    lines_1, labels_1 = ax1.get_legend_handles_labels()

    ax1.legend(lines_1 , labels_1 , loc='upper left')

    fig.tight_layout()
    plt.title('Close Price ')
    plt.show()



def check_patterns(ohlc_df, patterns_df, IsDebug = False):
    low_list = []
    high_list = []
    
    # Initialize variables
    in_long_position = False  # Track whether we are in a buy position
    previous_point = None
    buy_time = None
    sell_time = None
    hold_time = None  
    # Loop through the DataFrame and process each row for LONG position(做多)
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
                profit = sell_price - buy_price - cost
                if profit > 0: 
                    if previous_point is not None:
                        section_df = cut_slice(ohlc_df, previous_point, sell_time)                        
                    else:
                        section_df = cut_slice(ohlc_df, buy_time, sell_time)
                        
                    if (section_df is not None):
                        #print(f"Sliced DataFrame:{len(section_df)}\n", section_df)
                        high_list.append(section_df) 
                        
                    in_long_position = False
                    previous_point = buy_time
                    if IsDebug:
                        print(f"At {time}, {label} point, LONG sell price: {sell_price:.2f}, Profit: {profit:.2f}, Hold Time: {hold_time}")
                
                    continue
                else:
                    # if profit not > 0 or > 5, just drop this L/H pair
                    in_long_position = False
                    previous_point = buy_time
                    if IsDebug:
                        print(f"At {time}, {label} point, NO sell at price: {sell_price:.2f}, Profit: {profit:.2f}, ignore this pair.")         
            else:
                if IsDebug:
                    print(f"Not in position, ignoring sell signal at {time}, {label}")
                continue        
        else:
            print(f"Error: Not sure how to process this point at {time}, Label: {label}\n")
    
    if IsDebug:
        print("\n\n=======================================================================\n\n")
        
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
                    print(f"At {time}, SHORT buy  price: {buy_price:.2f} at {label} point")
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
                profit = -1 * (sell_price - buy_price) - cost
                if profit > 0: 
                    if previous_point is not None:
                        section_df = cut_slice(ohlc_df, previous_point, sell_time)                        
                    else:
                        section_df = cut_slice(ohlc_df, buy_time, sell_time)
                        
                    if (section_df is not None):
                        #print(f"Sliced DataFrame:{len(section_df)}\n", section_df)
                        low_list.append(section_df) 
                    
                    in_short_position = False
                    previous_point = buy_time
                    if IsDebug:
                        print(f"At {time}, {label} point, SHORT sell price: {sell_price:.2f}, Profit: {profit:.2f}, Hold Time: {hold_time}")
                    
                    continue
                else:
                    # if profit not > 0 or > 5, just drop this L/H pair
                    in_long_position = False
                    previous_point = buy_time
                    if IsDebug:
                        print(f"At {time}, {label} point, NO sell at price: {sell_price:.2f}, Profit: {profit:.2f}")         
            else:
                if IsDebug:
                    print(f"Not in position, ignoring sell signal at {time}, {label}")
                continue
        
        else:
            print(f"Error: Not sure how to process this point at {time}, Label: {label}\n")

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
    query_result_df = pd.read_sql_query(query_range, conn, params=(query_start, query_end))

    # print("Length of query result is:", len(query_result_df))
    # print("Datatype of query result:", type(query_result_df))
    # print(query_result_df)

    ohlc_df = query_result_df
    ohlc_df['Datetime'] = pd.to_datetime(ohlc_df['Datetime'])
    ohlc_df.set_index('Datetime', inplace=True)

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
        
    low_list, high_list = check_patterns(ohlc_df, patterns_df)
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

    IsDebug = False

    #Trainning Data Length
    # average number of working days in a month is 21.7, based on a five-day workweek
    # so 45 days is total for two months working days
    # 200 days is one year working days
    #tdLen = 30

    # Series Number for output training/testing data set pairs
    SN = "504"
        
    # ZigZag parameters
    deviation = 0.0015  # Percentage
        
    symbol = "SPX"
    #symbol = "MES=F"

    # Define the table name as a string variable
    table_name = "SPX_1m"
    #table_name = "MES=F_1m"
    # Define the SQLite database file directory
    data_dir = "data"

    db_file = os.path.join(data_dir, "stock_bigdata_2019-2023.db")
    
    # Cost for each trade
    cost = 5.00
    
    #============================= Training Data ============================================#
    training_start_date = "2020-01-01"
    training_end_date = "2023-05-31"

    now = datetime.now()
    formatted_now = now.strftime("%Y-%m-%d %H:%M:%S")
    print("Current date and time:", formatted_now)
    
    tddf_low_list, tddf_high_list = gen_highlow_list(training_start_date, training_end_date)
    if IsDebug:
        print(f"tddf_low_list length:{len(tddf_low_list)}\n")
        print(f"tddf_high_list length:{len(tddf_high_list)}\n")

    td_file = os.path.join(data_dir, f"{symbol}_TrainingData_2HL_{SN}.csv")

    with open(td_file, "w") as datafile:
        generate_training_data(tddf_low_list, TradePosition.LONG)
        generate_training_data(tddf_high_list, TradePosition.SHORT)


    #============================= Testing Data ============================================#
    testing_start_date = "2023-06-01"
    testing_end_date = "2023-12-31"
    
    now = datetime.now()
    formatted_now = now.strftime("%Y-%m-%d %H:%M:%S")
    print("Current date and time:", formatted_now)

    tddf_low_list, tddf_high_list = gen_highlow_list(testing_start_date, testing_end_date)

    td_file = os.path.join(data_dir, f"{symbol}_TestingData_2HL_{SN}.csv")

    with open(td_file, "w") as datafile:
        #generate_training_data(patterns_df)
        generate_testing_data(tddf_low_list, TradePosition.LONG)
        generate_testing_data(tddf_high_list, TradePosition.SHORT)

    now = datetime.now()
    formatted_now = now.strftime("%Y-%m-%d %H:%M:%S")
    print("Current date and time:", formatted_now)