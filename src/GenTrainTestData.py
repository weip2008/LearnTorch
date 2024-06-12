# This version read source data from SQLite database tables

import sqlite3
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import os
from ta.utils import dropna
from FilteredHighLowPoints import filter_points
from FilteredHighLowPoints import find_point_index_int
from TimeElapsed import measure_operation_time
from TimeElapsed import read_CSV_file
from enum import Enum
#from zigzagplus1 import calculate_zigzag,detect_patterns
import zigzagplus1 as zz
import CutSlice as ct

class TradePosition(Enum):
    LONG = 1
    HOLD = 0
    SHORT = -1


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
        result = "0,0,1," + trainingdata_str + "\n"
        if IsDebug:
            print(result)
        # Parse the input string into separate fields
        #fields = result.split(r',\s*|\)\s*\(', result.strip('[]()'))
        csvfile.write(result)
        return
    
    if (TradePosition is TradePosition.LONG):
        result = "1,0,0," + trainingdata_str + "\n"
        if IsDebug:
            print(result)
        # Parse the input string into separate fields
        #fields = result.split(r',\s*|\)\s*\(', result.strip('[]()'))
        csvfile.write(result)
        return
    
    if (TradePosition is TradePosition.HOLD):
        result = "0,1,0," + trainingdata_str + "\n"
        if IsDebug:
            print(result)
        # Parse the input string into separate fields
        #fields = result.split(r',\s*|\)\s*\(', result.strip('[]()'))
        csvfile.write(result)
        
    return


def write_testing_data(TradePosition, acceleration_list, csvfile):
    # for testing data, the first number is index of "LONG, HOLD, SHORT" series!
    # so if it's LONG, then it's 0; SHORT is 2;
    
    trainingdata_str = list_to_string(acceleration_list)
   
    if (TradePosition is TradePosition.LONG):
        result = "0," + trainingdata_str + "\n"
        if IsDebug:
            print(result)
    
        csvfile.write(result)
        return
    
    if (TradePosition is TradePosition.HOLD):
        result = "1," + trainingdata_str + "\n"
        if IsDebug:
            print(result)
        
        csvfile.write(result)        
        return
        
    if (TradePosition is TradePosition.SHORT):        
        result = "2," + trainingdata_str + "\n"
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

def check_patterns(ohlc_df, patterns_df, tdLen):
    
    # filtered_low_points_index = filtered_low_points.index.tolist()
    # filtered_high_points_index = filtered_high_points.index.tolist()

    low_list = []
    high_list = []
    # Loop through the DataFrame and find the first item with the second character of 'L'
    for idx, row in patterns_df.iterrows():
        if row['Label'][1] == 'L':
            #print(f"L point found at {idx}, {row['Label']}")
            section_df = ct.cut_slice(ohlc_df, idx, tdLen+1)
            if (section_df is not None):
                #print("\nSliced DataFrame:\n", section_df)
                low_list.append(section_df) 
            continue
        
        if row['Label'][1] == 'H':
            #print(f"H point found at {idx}, {row['Label']}")
            section_df = ct.cut_slice(ohlc_df, idx, tdLen+1)
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
    print("\n\n==========================4===Query==========================\n\n")


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
    print(f"filtered_zigzag_df list length:{len(filtered_zigzag_df)}\n",filtered_zigzag_df)

    # Detect patterns
    # df[df['Close'].isin(zigzag)] creates a new DataFrame 
    # that contains only the rows from df 
    # where the 'Close' value is in the zigzag list.
    # patterns = detect_patterns(df[df['Close'].isin(zigzag)])
    patterns = zz.detect_patterns(filtered_zigzag_df)
    #for pattern in patterns:
    #    print(f"Datetime: {pattern[0]}, Point: {pattern[1]}, Label: {pattern[2]}")
    print("Patterns list:\n", patterns)

    patterns_df = zz.convert_list_to_df(patterns)
    print(f"Patterns dataframe length:{len(patterns_df)}\n",patterns_df)  # Print to verify DataFrame structure

    zz.plot_patterns(ohlc_df, patterns_df)
        
    low_list, high_list = check_patterns(ohlc_df, patterns_df, tdLen)
    return low_list, high_list

#
# ================================================================================#
def main():
    #logging.basicConfig(level=logging.DEBUG, format='%(levelname)s - %(message)s')
    logging.basicConfig(
        level=logging.INFO,  # Set the logging level to DEBUG
        #format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        format=' %(levelname)s => %(message)s'
    )
            
    IsDebug = True
    #WindowLen = 5

    #Trainning data lenth
    # average number of working days in a month is 21.7, based on a five-day workweek
    # so 45 days is total for two months working days
    # 200 days is one year working days
    tdLen = 50

    # Series Number for output training data
    SN = "100"
        
    # ZigZag parameters
    deviation = 0.001  # Percentage
        
    symbol = "SPY"
    #symbol = "MES=F"

    # Define the table name as a string variable
    #table_name = "AAPL_1m"
    table_name = "SPY_1m"
    # Define the SQLite database file
    data_dir = "stockdata"
    db_file = os.path.join(data_dir, "stock_data.db")

    #=========================================================================#
    training_start_date = "2024-04-11"
    training_end_date = "2024-05-26"

    tddf_low_list, tddf_high_list = gen_highlow_list(training_start_date, training_end_date)

    td_file = os.path.join(data_dir, f"{symbol}_TrainingData_{tdLen}_{SN}.csv")

    with open(td_file, "w") as datafile:
        #generate_training_data(patterns_df)
        generate_training_data(tddf_low_list, TradePosition.LONG)
        generate_training_data(tddf_high_list, TradePosition.SHORT)
        #generate_training_data(tddf_hold_list, TradePosition.HOLD)

    #=========================================================================#
    #query_start = "2024-05-20"
    #query_end = "2024-05-26"
    testing_start_date = "2024-05-20"
    testing_end_date = "2024-05-26"

    tddf_low_list, tddf_high_list = gen_highlow_list(testing_start_date, testing_end_date)

    td_file = os.path.join(data_dir, f"{symbol}_TestingData_{tdLen}_{SN}.csv")

    with open(td_file, "w") as datafile:
        #generate_training_data(patterns_df)
        generate_testing_data(tddf_low_list, TradePosition.LONG)
        generate_testing_data(tddf_high_list, TradePosition.SHORT)
        #generate_training_data(tddf_hold_list, TradePosition.HOLD)


if __name__ == "__main__":
    main()