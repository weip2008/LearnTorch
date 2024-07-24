# This version read source data from SQLite database tables

import sqlite3
from datetime import datetime
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from enum import Enum

class TradePosition(Enum):
    LONG = 1
    SHORT = -1

def cut_traintest_slice(ohlc_df, traintest_data_len, target_len):
    total_len = traintest_data_len + target_len
    traintest_data_slices = []
    
    current_end = len(ohlc_df)
    while current_end >= total_len:
        # 1. Count back target_len
        current_end -= target_len+1
        
        # 2. Count back traintest_data_len
        current_start = current_end - traintest_data_len + 1
        
        # Ensure we don't go out of bounds
        if current_start < 0:
            break
        
        # 3. Cut the slice length traintest_data_len + target_lenif
        slice_df = ohlc_df.iloc[current_start:current_end + target_len + 1]
        # if IsDebug:
        #     print("Results dataframe length:", len(slice_df))  
        #     print(slice_df)
        
        traintest_data_slices.append(slice_df)
        
        # Update current_end to the start of the next slice
        current_end = current_start + target_len
    
    return traintest_data_slices


def cut_predect_slice(ohlc_df, traintest_data_len):
    total_len = traintest_data_len + target_len
    traintest_data_slices = []
    
    current_end = len(ohlc_df)
    while current_end >= total_len:
        # 1. Count back target_len
        current_end -= target_len+1
        
        # 2. Count back traintest_data_len
        current_start = current_end - traintest_data_len + 1
        
        # Ensure we don't go out of bounds
        if current_start < 0:
            break
        
        # 3. Cut the slice length traintest_data_len + target_lenif
        slice_df = ohlc_df.iloc[current_start:current_end + target_len + 1]
        # if IsDebug:
        #     print("Results dataframe length:", len(slice_df))  
        #     print(slice_df)
        
        traintest_data_slices.append(slice_df)
        
        # Update current_end to the start of the next slice
        #current_end = current_start + target_len
        current_end = current_end - target_len
    
    
    return traintest_data_slices


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

def convert_df_to_string(df):
    # Extract the "Normalized_Price" column as a list
    normalized_prices = df['Normalized_Price'].tolist()
    
    # Convert the list to a single tuple within a list
    #result_str = str([tuple(normalized_prices)])
    result_str = str(normalized_prices)
    
    return result_str


def generate_traintest_data(tddf_list, type):
    
    filename = 'stockdata/TrainTestDataGenLog_FixLenGRU_'+type+".log"
    # Open a file in write mode
    outputfile = open(filename, 'w')
 
    # Initialize an empty list to store tuples with the "Velocity" column
    tddf_velocity_list = []
    tddf_acceleration_list = []
     
    # Iterate over each tuple in tddf_highlow_list starting from the second tuple
    for i in range(0, len(tddf_list)):
        processing_df = tddf_list[i].copy()
        processing_df['Normalized_Price'] = normalize(processing_df['Close'])
        if IsDebug:
            print("\ncurrent processing DataFrame size:", len(processing_df), "\n", processing_df)        
        
        # Calculate the split point
        split_point = len(processing_df) - target_len

        # Split the DataFrame
        target_df = processing_df.iloc[split_point:].copy()
        processing_df = processing_df.iloc[:split_point].copy()
        if IsDebug:
            print("\ncurrent processing DataFrame size:", len(processing_df), "\n", processing_df)
            print("\ncurrent processing target size:", len(target_df), "\n", target_df)
        
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
        
        write_traintest_data(tddf_acceleration_list, target_df, datafile)
    
    outputfile.close()    
    return


def generate_predict_data(tddf_list, type):
    
    filename = 'stockdata/TestingDataGenLog_FixLenGRU_'+type+".log"
    # Open a file in write mode
    outputfile = open(filename, 'w')
 
    # Initialize an empty list to store tuples with the "Velocity" column
    tddf_velocity_list = []
    tddf_acceleration_list = []
    count = 0 
    # Iterate over each tuple in tddf_highlow_list starting from the second tuple
    for i in range(0, len(tddf_list)):
        processing_df = tddf_list[i].copy()
        processing_df['Normalized_Price'] = normalize(processing_df['Close'])
        if IsDebug:
            print("\ncurrent processing DataFrame size:", len(processing_df), "\n", processing_df)        
        
        # Calculate the split point
        split_point = len(processing_df) - target_len

        # Split the DataFrame
        target_df = processing_df.iloc[split_point:].copy()
        processing_df = processing_df.iloc[:split_point].copy()
        if IsDebug:
            print("\ncurrent processing DataFrame size:", len(processing_df), "\n", processing_df)
            print("\ncurrent processing target size:", len(target_df), "\n", target_df)
        
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
        
        write_traintest_data(tddf_acceleration_list, target_df, datafile)
        count=count+1
        if count> 9:
            break
    
    outputfile.close()    
    return



def calculate_velocity(processing_df):
    velocity_list = []
    
    # Find the lowest price and corresponding volume in the DataFrame
    #lowest_price, corresponding_volume = find_lowest_price_with_volume(processing_df)
    
    # Normalize 'Volume' and 'Price'
    #processing_df['Normalized_Volume'] = normalize(processing_df['Volume'])
    #processing_df['Normalized_Price'] = normalize(processing_df['Close'])
    
    #if IsDebug:
    #    print(processing_df)
    
    for j in range(1, len(processing_df)):
        # Extract Price from the current and previous rows
        #price_current = processing_df.iloc[j]['Close']
        #price_next = processing_df.iloc[j+1]['Close']
        normalized_price_previous = processing_df.iloc[j-1]['Normalized_Price']
        normalized_price_current = processing_df.iloc[j]['Normalized_Price']
        #normalized_price_next = processing_df.iloc[j+1]['Normalized_Price']

        #print("Price_current:", Price_current)
        #print("Price_previous:", Price_previous)
        
        #dY = price_current - price_previous
        dY = normalized_price_current - normalized_price_previous
        #dY = normalized_price_next - normalized_price_current 
        #print("dY:", dY)
        
        # Extract timestamps from the current and previous rows
        index_previous = processing_df.index[j-1]
        index_current = processing_df.index[j]
        #index_next = processing_df.index[j+1]
        #print("index_current:", index_current)
        #print("index_previous:", index_next)
        
        #dT = (index_next - index_current) / pd.Timedelta(minutes=1)  
        #dT = index_current - index_previous 
        #dT = (index_next - index_current) / tdLen
        loc_previous = processing_df.index.get_loc(index_previous)
        loc_current = processing_df.index.get_loc(index_current)
        #loc_next = processing_df.index.get_loc(index_next)

        # Calculate dT based on the difference of locations
        dT = loc_current - loc_previous
        #dT = loc_next - loc_current
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
    for i in range(1, len(velocity_list)):
        # Extract velocity data from the current and next tuples
        #next_tuple = velocity_list[i+1] 
        current_tuple = velocity_list[i]
        previous_tuple = velocity_list[i-1]

        #velocity_next = next_tuple[2]
        velocity_current = current_tuple[2]  # velocity is stored at index 2 in the tuple
        velocity_previous = previous_tuple[2]

        # Calculate the change in velocity
        dV = velocity_current - velocity_previous
        #dV = velocity_next - velocity_current 
        
        #index_current = velocity_list[i].index
        #index_previous = velocity_list[i-1].index
        index_current = i
        #index_next = i+1
        index_previous = i-1
        dT = index_current - index_previous
        #dT = index_next - index_current
        
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



def list_to_string(price_list):
    return ', '.join(map(str, price_list))

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

def gen_list(processing_df):
    price_list = []

    processing_df['Normalized_Price'] = normalize(processing_df['Close'])
    
    if IsDebug:
        print(processing_df)
        plot_prices(processing_df)
    
    for j in range(0, len(processing_df)):

        normalized_price_current = processing_df.iloc[j]['Normalized_Price']
        index_current = processing_df.index[j]
        
        #price_list.append((index_current, normalized_price_current))
        price_list.append((normalized_price_current))

    return price_list

    # Example usage:
    # acceleration_data = calculate_acceleration(velocity_list)

def write_traintest_data( acceleration_list, target_df, csvfile):
    # Initialize an empty string to store the result
    #result = ""
    
    trainingdata_str = list_to_string(acceleration_list)
    target_str = convert_df_to_string(target_df)
    
    # Iterate over each tuple in the acceleration_list
    # for acceleration_tuple in acceleration_list:
    #     # Convert each element of the tuple to a string and concatenate them
    #     result += ",".join(map(str, acceleration_tuple)) 
      
    result = "["+trainingdata_str+"]" + target_str +"\n"
    if IsDebug:
        print(result)

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


def load_data(query_start, query_end):
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
    ohlc_df = ohlc_df.drop(columns=['Volume'])

    if IsDebug:
        #print("Time elapsed:", time_elapsed, "seconds")
        print("Results dataframe length:", len(ohlc_df))  
        #print("Data read from :", file_path)
        print("Data read from table:", table_name)
        # Print the first few rows of the DataFrame
        print(ohlc_df.head(10))
        print(ohlc_df.tail(10))
  
    return ohlc_df

# Normalization function
def normalize(series):
    return (series - series.min()) / (series.max() - series.min())


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
    traintest_data_len = 120
    target_len = 3

    # Series Number for output training/testing data set pairs
    SN = "604"
        
    symbol = "SPX"
    #symbol = "MES=F"

    # Define the table name as a string variable
    table_name = "SPX_1m"
    #table_name = "MES=F_1m"
    # Define the SQLite database file directory
    data_dir = "data"

    db_file = os.path.join(data_dir, "stock_bigdata_2019-2023.db")
  
    #============================= Training Data ============================================#
    training_start_date = "2020-01-01"
    training_end_date = "2023-06-30"

    now = datetime.now()
    formatted_now = now.strftime("%Y-%m-%d %H:%M:%S")
    print("Current date and time:", formatted_now)
    
    ohlc_df = load_data(training_start_date, training_end_date)

    traintest_data_slices_list = cut_traintest_slice(ohlc_df, traintest_data_len+2, target_len)

    td_file = os.path.join(data_dir, f"{symbol}_TrainingData_FixLenGRU_{traintest_data_len}_{SN}.txt")

    with open(td_file, "w") as datafile:
        generate_traintest_data(traintest_data_slices_list, "Train")


#============================= Testing Data ============================================#
    testing_start_date = "2023-07-01"
    testing_end_date = "2023-12-31"

    now = datetime.now()
    formatted_now = now.strftime("%Y-%m-%d %H:%M:%S")
    print("Current date and time:", formatted_now)
    
    ohlc_df = load_data(testing_start_date, testing_end_date)

    testing_data_slices_list = cut_traintest_slice(ohlc_df, traintest_data_len+2, target_len)

    td_file = os.path.join(data_dir, f"{symbol}_TestingData_FixLenGRU_{traintest_data_len}_{SN}.txt")

    with open(td_file, "w") as datafile:
        generate_traintest_data(testing_data_slices_list, "Test")

    #============================= Prediction Data ============================================#
    training_start_date = "2023-06-01"
    training_end_date = "2023-06-30"

    now = datetime.now()
    formatted_now = now.strftime("%Y-%m-%d %H:%M:%S")
    print("Current date and time:", formatted_now)
    
    ohlc_df = load_data(training_start_date, training_end_date)

    traintest_data_slices_list = cut_traintest_slice(ohlc_df, traintest_data_len+2, target_len)

    td_file = os.path.join(data_dir, f"{symbol}_PredictData_FixLenGRU_{traintest_data_len}_{SN}.txt")

    with open(td_file, "w") as datafile:
        generate_predict_data(traintest_data_slices_list,"Predict")