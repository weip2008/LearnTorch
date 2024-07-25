# This version read source data from SQLite database tables

import sqlite3
from datetime import datetime
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from concurrent.futures import ThreadPoolExecutor

def cut_traintest_slice(ohlc_df, traintest_data_len, target_len):
    total_len = traintest_data_len + target_len
    traintest_data_slices = []
    
    current_end = len(ohlc_df)
    while current_end >= total_len:
        # 1. Count back target_len
        current_end -= target_len + 1
        
        # 2. Count back traintest_data_len
        current_start = current_end - traintest_data_len + 1
        
        # Ensure we don't go out of bounds
        if current_start < 0:
            break
        
        # 3. Cut the slice length traintest_data_len + target_len
        slice_df = ohlc_df.iloc[current_start:current_end + target_len + 1]
        
        # Drop the 'Open', 'High', and 'Low' columns
        slice_df = slice_df.drop(columns=['Open', 'High', 'Low'])
        
        # Normalize the 'Close' column
        slice_df['Normalized_Price'] = normalize(slice_df['Close'])
        
        # Add 'Velocity' and 'Acceleration' columns
        slice_df['Velocity'] = slice_df['Normalized_Price'].diff()
        slice_df['Acceleration'] = slice_df['Velocity'].diff()
        
        # Drop rows with NaN values
        slice_df = slice_df.dropna()
        
        traintest_data_slices.append(slice_df)
        
        # Update current_end to the start of the next slice
        current_end = current_start + target_len
    
    return traintest_data_slices


def process_chunk(ohlc_chunk, traintest_data_len, target_len):
    return cut_traintest_slice(ohlc_chunk, traintest_data_len, target_len)

def parallel_cut_traintest_slice(ohlc_df, traintest_data_len, target_len, num_threads=5):
    # Split the DataFrame into chunks
    chunk_size = len(ohlc_df) // num_threads
    chunks = [ohlc_df[i:i + chunk_size] for i in range(0, len(ohlc_df), chunk_size)]
    
    traintest_data_slices = []
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(process_chunk, chunk, traintest_data_len, target_len) for chunk in chunks]
        for future in futures:
            traintest_data_slices.extend(future.result())
    
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


def convert_df_to_string(df):
    # Extract the "Normalized_Price" column as a list
    normalized_prices = df['Normalized_Price'].tolist()
    
    # Convert the list to a single tuple within a list
    #result_str = str([tuple(normalized_prices)])
    result_str = str(normalized_prices)
    
    return result_str


# Function to convert DataFrame slice to string format with numeric date and time
def df_slice_to_string_with_numeric_datetime(df):
    result = [
        (convert_to_day_and_time(row.name)[0], convert_to_day_and_time(row.name)[1], row['Normalized_Price'], row['Velocity'], row['Acceleration'])
        for _, row in df.iterrows()
    ]
    return str(result)


def generate_traintest_file(tddf_list, datatype):
    
    td_file = os.path.join(data_dir, f"{symbol}_{datatype}Data_FixLenGRU_{traintest_data_len}_{SN}.txt")

    with open(td_file, "w") as datafile:
    # Iterate over each tuple in tddf_highlow_list starting from the second tuple
        for i in range(0, len(tddf_list)):
            processing_df = tddf_list[i].copy()
            #processing_df['Normalized_Price'] = normalize(processing_df['Close'])
            if IsDebug:
                print("\ncurrent processing DataFrame size:", len(processing_df), "\n", processing_df)        
            
            # Calculate the split point
            split_point = len(processing_df) - target_len

            # Split the DataFrame
            target_df = processing_df.iloc[split_point:].copy()
            traintestdata_df = processing_df.iloc[:split_point].copy()
            if IsDebug:
                print("\ncurrent processing DataFrame size:", len(traintestdata_df), "\n", traintestdata_df)
                print("\ncurrent processing target size:", len(target_df), "\n", target_df)
        
            #write_traintest_data(traintestdata_df, target_df, datafile)
            trainingdata_str = df_slice_to_string_with_numeric_datetime(traintestdata_df)
            #print(trainingdata_str)
            target_str = convert_df_to_string(target_df)
            result_string = trainingdata_str + target_str +"\n"
            if IsDebug:
                print(result_string)

            datafile.write(result_string)
        
    return


def generate_predict_data(tddf_list, datatype):
    count = 0
    td_file = os.path.join(data_dir, f"{symbol}_{datatype}Data_FixLenGRU_{traintest_data_len}_{SN}.txt")

    with open(td_file, "w") as datafile:
    # Iterate over each tuple in tddf_highlow_list starting from the second tuple
        for i in range(0, len(tddf_list)):
            processing_df = tddf_list[i].copy()
            #processing_df['Normalized_Price'] = normalize(processing_df['Close'])
            if IsDebug:
                print("\ncurrent processing DataFrame size:", len(processing_df), "\n", processing_df)        
            
            # Calculate the split point
            split_point = len(processing_df) - target_len

            # Split the DataFrame
            target_df = processing_df.iloc[split_point:].copy()
            traintestdata_df = processing_df.iloc[:split_point].copy()
            if IsDebug:
                print("\ncurrent processing DataFrame size:", len(traintestdata_df), "\n", traintestdata_df)
                print("\ncurrent processing target size:", len(target_df), "\n", target_df)
        
            #write_traintest_data(traintestdata_df, target_df, datafile)
            trainingdata_str = df_slice_to_string_with_numeric_datetime(traintestdata_df)
            #print(trainingdata_str)
            target_str = convert_df_to_string(target_df)
            result_string = trainingdata_str + target_str +"\n"
            if IsDebug:
                print(result_string)

            datafile.write(result_string)
            count=count+1
            if count> 9:
                break

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

    # Add a legend to differentiate the plots
    lines_1, labels_1 = ax1.get_legend_handles_labels()

    ax1.legend(lines_1 , labels_1 , loc='upper left')

    fig.tight_layout()
    plt.title('Close Price ')
    plt.show()



    
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

def process_data(start_date, end_date, datatype):
    print(f"---------------------------{datatype}----------------------------------")
    print("1. Load data")
    now = datetime.now()
    formatted_now = now.strftime("%Y-%m-%d %H:%M:%S")
    print("Current date and time:", formatted_now)
    
    ohlc_df = load_data(start_date, end_date)
    print(f"Length of ohlc_df: {len(ohlc_df)}")
    

    now = datetime.now()
    formatted_now = now.strftime("%Y-%m-%d %H:%M:%S")
    print("Current date and time:", formatted_now)

    print("2. Cut slices")    
    # Use multi-threading to cut the DataFrame into slices
    slices_list  = parallel_cut_traintest_slice(ohlc_df, traintest_data_len+2, target_len, num_threads=5)

    # Get the length of the slices list
    slices_length = len(slices_list)

    # Print the length of the slices list
    print(f"Number of slices: {slices_length}")
    
    now = datetime.now()
    formatted_now = now.strftime("%Y-%m-%d %H:%M:%S")
    print("Current date and time:", formatted_now)
    
    print("3. Generate training/testing data file")
    #td_file = os.path.join(data_dir, f"{symbol}_{datatype}Data_FixLenGRU_{traintest_data_len}_{SN}.txt")

    #with open(td_file, "w") as datafile:
    generate_traintest_file(slices_list , datatype)

    print("4. Finish")
    now = datetime.now()
    formatted_now = now.strftime("%Y-%m-%d %H:%M:%S")
    print("Current date and time:", formatted_now)

    print(f"---------------------------{datatype}----------------------------------")
    return
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
    SN = "620"
        
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

    process_data(training_start_date, training_end_date, "Training")
    
    #============================= Testing Data ============================================#
    testing_start_date = "2023-07-01"
    testing_end_date = "2023-12-31"

    process_data(testing_start_date, testing_end_date, "Testing")

    #============================= Prediction Data ============================================#
    predict_start_date = "2023-06-01"
    predict_end_date = "2023-06-30"
    
    process_data(predict_start_date, predict_end_date, "Predict")
