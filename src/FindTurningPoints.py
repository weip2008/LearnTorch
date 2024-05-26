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


class TradePosition(Enum):
    Long = 1
    Short = 0


def find_selected_points1(ohlc_df, comparison_operator):

    # Find points based on the comparison operator
    # shift(1) is move backward(previous) one position
    points = ohlc_df[comparison_operator(ohlc_df['Price'], ohlc_df['Price'].shift(1)) & 
                comparison_operator(ohlc_df['Price'], ohlc_df['Price'].shift(2)) & 
                comparison_operator(ohlc_df['Price'], ohlc_df['Price'].shift(3)) & 
                comparison_operator(ohlc_df['Price'], ohlc_df['Price'].shift(-1)) & 
                comparison_operator(ohlc_df['Price'], ohlc_df['Price'].shift(-2))]

    # Select points based on the comparison operator
    selected_points = points[comparison_operator(points['Price'], points['Price'].shift(1)) & 
                             comparison_operator(points['Price'], points['Price'].shift(2)) & 
#                             comparison_operator(points['Price'], points['Price'].shift(3)) & 
                             comparison_operator(points['Price'], points['Price'].shift(-1))]

    return selected_points

def find_selected_points2(ohlc_df, comparison_operator):
    
    # Find points based on the comparison operator
    # shift(1) is move backward(previous) one position
    points = ohlc_df[comparison_operator(ohlc_df['Price'], ohlc_df['Price'].shift(1)) & 
                comparison_operator(ohlc_df['Price'], ohlc_df['Price'].shift(2)) & 
#                comparison_operator(ohlc_df['Price'], ohlc_df['Price'].shift(3)) & 
                comparison_operator(ohlc_df['Price'], ohlc_df['Price'].shift(-1)) & 
                comparison_operator(ohlc_df['Price'], ohlc_df['Price'].shift(-2))]

    # Select points based on the comparison operator
    selected_points = points[comparison_operator(points['Price'], points['Price'].shift(1)) & 
                             comparison_operator(points['Price'], points['Price'].shift(2)) & 
                             comparison_operator(points['Price'], points['Price'].shift(-1)) & 
                             comparison_operator(points['Price'], points['Price'].shift(-2))]

    return selected_points


   
def cut_slices(ohlc_df, selected_points_index, window_len):
    '''
    In this function the start index has -1 and the end index +1 is because in next step
    calculate voloctiy, the result voloctiy list will shrink at two ends;
    in next step calculate acceleration, the list will shrink at two ends again.
    therefore, the list should two more elements at both ends
    '''
    tddf_list = []
    for index in selected_points_index:
        # Backup WindowLen positions and extract the slice of data
        l1 = ohlc_df.index[0]
        # pd.Timedelta is a pandas function used to represent a duration 
        # or a difference between two dates or times. 
        #td =  pd.Timedelta(minutes=(window_len + 1))
        l2 = index - (window_len+1) 
        start_index = max(l1, l2)
        #start_index = index - (window_len+1)  
        
        # Adjust end index to include one more element
        end_index = index+1
        
        # By passing minutes=1 as an argument, 
        # create a Timedelta object that represents a duration of 1 minute.
        # end_index = index + pd.Timedelta(minutes=1)  
        
        # Create a copy of the section of the original DataFrame
        # start from start_index up to but not including end_index!
        #section_df = ohlc_df.loc[start_index:end_index].copy()  
        section_df = ohlc_df[start_index:end_index].copy()
        # if IsDebug:
        #     print("section_df length:", len(section_df))
         
        # Log the modified DataFrame
        #logging.debug("\nSliced DataFrame:\n%s", section_df)

        # Drop unwanted columns
        section_df.drop(['Open', 'High', 'Low', 'Close', 'AdjClose'], axis=1, inplace=True)  
        # Reset the index and drop the old index
        section_df.reset_index(drop=True, inplace=True)
        logging.debug("\nSliced DataFrame:\n%s", section_df)
        
        tddf_list.append(section_df)  # Append the section DataFrame to the tdohlc_df_list
        
    #logging.DEBUG(tddf_list)
    
    return tddf_list

''' def list_to_string(acceleration_list):
    # Example usage:
    # acceleration_str = list_to_string(acceleration_list)
    # print(acceleration_str)

    # Convert each tuple to a string with parentheses and join them with newline characters
    return ' '.join(['(' + ','.join(map(str, acceleration_tuple)) + '),' for acceleration_tuple in acceleration_list])
 '''

def list_to_string(acceleration_list):
    # Convert each tuple to a string with parentheses and join them with newline characters
    #return ' '.join(['(' + ','.join(map(str, acceleration_tuple)) + '),' for acceleration_tuple in acceleration_list])
    #return ' '.join('(' + ','.join(map(str, acceleration_tuple)) + '),' for acceleration_tuple in acceleration_list)    
    #return ','.join(','.join(map(str, acceleration_tuple)) for acceleration_tuple in acceleration_list)
    return ','.join(','.join(f'{value:.4f}' for value in acceleration_tuple) for acceleration_tuple in acceleration_list)


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


def calculate_velocity(processing_element):
    velocity_list = []
    for j in range(1, len(processing_element)):
        # Extract Price from the current and previous rows
        #price_current = processing_element[j]['Price']
        #price_previous = processing_element[j - 1]['Price']
        price_current = processing_element.iloc[j]['Price']
        price_previous = processing_element.iloc[j - 1]['Price']

        #print("Price_current:", Price_current)
        #print("Price_previous:", Price_previous)
        
        dY = price_current - price_previous
        #print("dY:", dY)
        
        # Extract timestamps from the current and previous rows
        index_current = processing_element.index[j]
        index_previous = processing_element.index[j - 1]
        #print("index_current:", index_current)
        #print("index_previous:", index_previous)
        
        #dT = (index_current - index_previous) / pd.Timedelta(minutes=1)  
        dT = index_current - index_previous 
        #print("dT:", dT)
        
        # Calculate the velocity (dY/dT)
        velocity = dY / dT
        #print("velocity:", velocity)
        
        datetime_current = processing_element.iloc[j]['Datetime']
        volume_current = processing_element.iloc[j]['Volume']
        # Append the tuple with the "Velocity" column to tdohlc_df_high_velocity_list
        velocity_list.append((datetime_current, price_current, volume_current, velocity))

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
        # Extract velocity data from the current and previous tuples
        current_tuple = velocity_list[i]
        previous_tuple = velocity_list[i - 1]

        velocity_current = current_tuple[3]  # velocity is stored at index 2 in the tuple
        velocity_previous = previous_tuple[3]

        # Calculate the change in velocity
        #dV = abs(velocity_current) - abs(velocity_previous)
        dV = velocity_current - velocity_previous

        ''' 
        # Convert timestamp strings to datetime objects
        # then alculate the change in time (dT) in minutes
        # But when using different time unit, need change code
        index_current = pd.to_datetime(current_tuple[0])
        index_previous = pd.to_datetime(previous_tuple[0])
        dT = (index_current - index_previous) / pd.Timedelta(minutes=1)  # Convert to minutes
        '''
        #index_current = velocity_list[i].index
        #index_previous = velocity_list[i-1].index
        index_current = i
        index_previous = i-1
        dT = index_current - index_previous
        
        # Calculate acceleration (dV/dT)
        acceleration = dV / dT
        
        current_time = pd.to_datetime(current_tuple[0])
        #current_volume = current_tuple[2]
        #day_of_week_numeric, time_float = convert_to_day_and_time(index_current)
        day_of_week_numeric, time_float = convert_to_day_and_time(current_time)

        # Append the tuple with the "Acceleration" column to acceleration_list
        #acceleration_list.append((index_current, current_tuple[1], velocity_current, acceleration))
        acceleration_list.append((day_of_week_numeric, time_float, 
                                  current_tuple[1],  current_tuple[2], current_tuple[3], 
                                  acceleration))

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
    
    if (TradePosition is TradePosition.Short):        
        result = "0,1," + trainingdata_str + "\n"
        if IsDebug:
            print(result)
        # Parse the input string into separate fields
        #fields = result.split(r',\s*|\)\s*\(', result.strip('[]()'))
        csvfile.write(result)
    else:
        result = "1,0," + trainingdata_str + "\n"
        if IsDebug:
            print(result)
        # Parse the input string into separate fields
        #fields = result.split(r',\s*|\)\s*\(', result.strip('[]()'))
        csvfile.write(result)
    
    return

def generate_training_data(tddf_highlow_list, position):
    
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
            
        write_training_data(position, tddf_acceleration_list, datafile)
        
    return

#logging.basicConfig(level=logging.DEBUG, format='%(levelname)s - %(message)s')
logging.basicConfig(
    level=logging.DEBUG,  # Set the logging level to DEBUG
    #format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    format=' %(levelname)s => %(message)s'
)
        
IsDebug = True
#WindowLen = 5

#Trainning data lenth
tdLen = 200

# Series Number for output training data
SN = "08"
           
symbol = "SPY"
#symbol = "MES=F"

# Define the table name as a string variable
#table_name = "AAPL_1m"
table_name = "SPY_1m"
# Define the SQLite database file
data_dir = "stockdata"
db_file = os.path.join(data_dir, "stock_data.db")

# Define the query date range
query_start = "2024-04-11"
#query_end = "2024-04-19"
query_end = "2024-05-19"

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

if IsDebug:
    #print("Time elapsed:", time_elapsed, "seconds")
    print("Results dataframe length:", len(ohlc_df))  
    #print("Data read from :", file_path)
    print("Data read from table:", table_name)
    # Print the first few rows of the DataFrame
    print(ohlc_df.head(10))
    print(ohlc_df.tail(10))


#logging.debug("Time elapsed: %s seconds", time_elapsed)
logging.debug("Results dataframe length: %d", len(ohlc_df))
logging.debug("Data read from table: %s", table_name)
# Log the first few rows of the DataFrame
logging.debug("First 10 rows of the DataFrame:\n%s", ohlc_df.head(10))
logging.debug("Last 10 rows of the DataFrame:\n%s", ohlc_df.tail(10))


# Clean NaN values
ohlc_df.dropna()

# Calculate the rolling average over a 3-day window for the 'Adj Close' column
rolling_avg = ohlc_df['AdjClose'].rolling(window=3).mean()

# Shift the rolling average by one position to align with the second position (index = 1)
rolling_avg = rolling_avg.shift(-1)

# Append the rolling average as a new column named 'Price' to the original DataFrame
ohlc_df['Price'] = rolling_avg.round(2)
print(ohlc_df.head(3))
print(ohlc_df.tail(3))

 # finding high points
selected_high_points = find_selected_points1(ohlc_df, lambda x, y: x > y)

# finding low points
selected_low_points = find_selected_points1(ohlc_df, lambda x, y: x < y)

if IsDebug:
    print("\n1st round selected high points:\n", selected_high_points.head(20))
    print(type(selected_high_points))
    print("\n1st round selected low points:\n", selected_low_points.head(20))
    print(type(selected_low_points))

# Call the filtering function
filtered_low_points, filtered_high_points = filter_points(selected_low_points, selected_high_points)

if IsDebug:
    # Output filtered_low_points and filtered_high_points
    print("\nFiltered Low Points:")
    print(filtered_low_points)
    print("\nFiltered High Points:")
    print(filtered_high_points)


# Example usage of find_point_index_int
''' print("\nIndex and location of filtered points in selected_low_points:")
for i, filtered_point in filtered_low_points.iterrows():
    original_index, location = find_point_index_int(filtered_point, selected_low_points)
    if original_index is not None:
        print(f"Filtered point at index {i} corresponds to original index {original_index} \
                and location No. {location} in selected_low_points.")
    else:
        print(f"Filtered point at index {i} not found in selected_low_points.")
    
         '''

# Plot original data
plt.figure(figsize=(10, 6))
#plt.plot(ohlc_df.index, ohlc_df['Adj Close'], color='blue', label='Original Data')
plt.plot(ohlc_df.index, ohlc_df['Price'], color='blue', label='3-Point Avg Data')

# Plot selected points
plt.scatter(selected_high_points.index, selected_high_points['Price'], 
            color='green', label='Selected High Points', marker='o')

plt.scatter(selected_low_points.index, selected_low_points['Price'], 
            color='red', label='Selected Low Points', marker='o')

            
# Add legend and labels
plt.legend()
plt.title(symbol+' Original Data with Selected Points')
plt.xlabel('Date')
plt.ylabel('Price')

# Rotate date labels for better readability
plt.xticks(rotation=45)

# Show plot
plt.tight_layout()
plt.show()

selected_low_points_index = selected_low_points.index.tolist()
selected_high_points_index = selected_high_points.index.tolist()

filtered_low_points_index = filtered_low_points.index.tolist()
filtered_high_points_index = filtered_high_points.index.tolist()

tddf_low_list = cut_slices(ohlc_df, filtered_low_points_index, tdLen)
#print("First couple of Traning Data Dataframe Low points list:\n")
#print(tddf_low_list[:5])
tddf_high_list = cut_slices(ohlc_df, filtered_high_points_index, tdLen)

#formatted_strings = convert_list_to_string(tddf_low_list)

# Print out the first 5 formatted strings
#for formatted_str in formatted_strings[:5]:
#    print(formatted_str)
    
#tddf_low_list = find_highest_points(bbohlc_df)

if IsDebug:
    print("\n\n\nLength of original DataFrames:", len(ohlc_df))
    print("Length of Low Trainning Data DataFrames:", len(tddf_low_list))               
    print("Contents of Trainning Data DataFrmes:")
    print(tddf_low_list[:3])    # first 3
    print(".................")
    print(tddf_low_list[-3:])   # last 3
    print("========================================\n\n\n")


    print("Length of High Trainning Data DataFrames:", len(tddf_high_list))               
    print("Contents of Trainning Data DataFrmes:")
    print(tddf_high_list[:3])    # first 3
    print(".................")
    print(tddf_high_list[-3:])   # last 3
    print("========================================\n\n\n")

# Open a CSV file in write mode

td_file = os.path.join(data_dir, f"{symbol}_TraningData_{tdLen}_{SN}.csv")

with open(td_file, "w") as datafile:
    generate_training_data(tddf_low_list, TradePosition.Long)
    generate_training_data(tddf_high_list, TradePosition.Short)


