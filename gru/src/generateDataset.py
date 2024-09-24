# This version read source data from SQLite database tables
#
# [1,0] stand for sell
# [0,1] stand for buy
# [0,1,1,0] test data means 4 set of [sell, buy, buy, sell]

from datetime import datetime
from datetime import timedelta
import logging
import pandas as pd
import numpy as np
import os
from enum import Enum
import statistics

import zigzagplus1 as zz
from gru import Logger
from config import Config, execution_time
from utilities import DataSource, convert_to_day_and_time,normalize

class TradePosition(Enum):
    LONG = 1
    SHORT = -1

class DataProcessor:
    traintest_data_len = None
    def __init__(self, training=True):
        traintest_data_len = int(DataSource.config.traintest_data_len)
        self.ds = DataSource()
        self.query_start, self.query_end= DataSource.config.training_start_date, DataSource.config.training_end_date
        if not training:
            self.query_start, self.query_end= DataSource.config.testing_start_date,DataSource.config.testing_end_date

        self.ds.queryDB(self.query_start, self.query_end)
        self.df = self.ds.getDataFrameFromDB()
        self.gen_zigzag_patterns()

        if (DataProcessor.traintest_data_len is None):
            DataProcessor.traintest_data_len = self.check_patterns_length()
        tddf_long_list= self.check_long_patterns(DataProcessor.traintest_data_len)
        tddf_short_list = self.check_short_patterns(DataProcessor.traintest_data_len)
        if training:
            self.generateTrain(tddf_short_list, tddf_long_list, DataProcessor.traintest_data_len)
        else:
            self.generateTest(tddf_short_list, tddf_long_list, DataProcessor.traintest_data_len)

    def gen_zigzag_patterns(self):
        zigzag = self.ds.getZigzag()
        log.debug(f"Zigzag list length:{len(zigzag)}\n{zigzag}")

        # Filter the original DataFrame using the indices
        # df.loc[zigzag.index]:
        # This expression uses the .loc accessor to select rows from the original DataFrame df.
        # The rows selected are those whose index labels match the index labels of the zigzag DataFrame (or Series).
        # In other words, it filters df to include only the rows where the index (Date) is present in the zigzag index.
        filtered_zigzag_df = self.df.loc[zigzag.index]
        log.debug(f"filtered_zigzag_df list length:{len(filtered_zigzag_df)}\n{filtered_zigzag_df}")

        # Detect patterns
        # df[df['Close'].isin(zigzag)] creates a new DataFrame 
        # that contains only the rows from df 
        # where the 'Close' value is in the zigzag list.
        # patterns = detect_patterns(df[df['Close'].isin(zigzag)])
        patterns = zz.detect_patterns(filtered_zigzag_df)

        self.patterns_df = zz.convert_list_to_df(patterns)
        log.debug(f"Patterns dataframe length:{len(self.patterns_df)}\n{self.patterns_df}")  # Print to verify DataFrame structure

    def check_patterns_length(self):
        traintest_data_len = int(config.traintest_data_len)
        longtradecost = float(config.longtradecost)
        shorttradecost = float(config.shorttradecost)
        
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
        for idx, row in self.patterns_df.iterrows():
            label = row['Label']
            price = row['Price']
            time = idx  # Access the time from the index directly
            
            start_pos = self.df.index.get_loc(idx)
            if start_pos < traintest_data_len:
                continue
            
            if label[1] == 'L':
                if not in_long_position:
                    # Buy in at this point
                    buy_price = price
                    buy_time = idx
                    in_long_position = True
                    log.debug(f"At {time}, LONG buy  price: {buy_price:.2f} at {label} point")
                else:
                    log.debug(f"At {time}, already in long position, ignoring signal {label} at price: {buy_price:.2f}")
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
                        log.debug(f"At {time}, LONG sell price: {sell_price:.2f} at {label} point, Profit: {profit:.2f}, Hold Time: {hold_time}")
                    
                        continue
                    else:
                        # if profit not > 0, just drop this L/H pair
                        in_long_position = False
                        log.debug(f"At {time}, NO sell at price: {sell_price:.2f} at {label} point, Profit: {profit:.2f}, ignore this pair.")         
                else:
                    log.debug(f"At {time}, Not in position, ignore sell signal at {label} point")
                    continue        
            else:
                log.info(f"Error: Not sure how to process this point at {time}, Label: {label}\n")

        # Mean hold time
        mean_hold_time = statistics.mean(holdtime_list)

        # Median hold time
        median_hold_time = statistics.median(holdtime_list)

        # Standard deviation
        std_dev_hold_time = statistics.stdev(holdtime_list)

        log.info("Long:")
        log.info(f"Mean Hold Time: {mean_hold_time}")
        log.info(f"Median Hold Time: {median_hold_time}")
        log.info(f"Standard Deviation: {std_dev_hold_time}")
        
        long_hold_time = int(np.ceil(mean_hold_time))
        
        log.debug("\n\n=======================================================================\n\n")
            
        # essentially, "selling high and buying low" is the strategy for a short position.    
        in_short_position = False
        previous_point = None
        buy_time = None
        sell_time = None      
        hold_time = None  
        # Loop through the DataFrame and process each row for SHORT position (做空)
        for idx, row in self.patterns_df.iterrows():
            label = row['Label']
            price = row['Price']
            time = idx  # Access the time from the index directly
            
            start_pos = self.df.index.get_loc(idx)
            if start_pos < traintest_data_len:
                continue
            
            if label[1] == 'H':
                if not in_short_position:
                    # Buy in at this point
                    buy_price = price
                    buy_time = time
                    in_short_position = True
                    log.debug(f"At {time}, SHORT sell price: {buy_price:.2f} at {label} point")
                else:
                    in_short_position = False
                    log.debug(f"At {time}, already in SHORT position, ignoring signal {label} at price: {buy_price:.2f}. Ignore this pair.")
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
                        in_short_position = False
                        log.debug(f"At {time}, SHORT buy  price: {sell_price:.2f} at {label} point, Profit: {profit:.2f}, Hold Time: {hold_time}")
                        
                        continue
                    else:
                        # if profit not > 0 , just drop this L/H pair
                        in_short_position = False
                        log.debug(f"At {time}, NO sell at price: {sell_price:.2f} at {label} point, Profit: {profit:.2f}")         
                else:
                    log.debug(f"At {time}, Not in position, ignore sell signal at {label} point")
                    continue
            
            else:
                log.info(f"Error: Not sure how to process this point at {time}, Label: {label}\n")

        # Mean hold time
        mean_hold_time = statistics.mean(holdtime_list)

        # Median hold time
        median_hold_time = statistics.median(holdtime_list)

        # Standard deviation
        std_dev_hold_time = statistics.stdev(holdtime_list)

        log.info("Short:")
        log.info(f"Mean Hold Time: {mean_hold_time}")
        log.info(f"Median Hold Time: {median_hold_time}")
        log.info(f"Standard Deviation: {std_dev_hold_time}")
        
        short_hold_time = int(np.ceil(mean_hold_time))
        
        avg_hold_time = int((short_hold_time + long_hold_time) / 2 )
        log.info(f"Avg mean Hold Time: {avg_hold_time}")
        
        return avg_hold_time

    def check_long_patterns(self, long_traintest_data_len):
        traintest_data_len = int(config.traintest_data_len)
        longtradecost = float(config.longtradecost)

        long_list = []
        
        # Initialize variables
        in_long_position = False  # Track whether we are in a buy position
        #previous_point = None
        buy_time = None
        sell_time = None
        hold_time = None  
        # Loop through the DataFrame and process each row for LONG position(做多)
        # essentially, "buying low and selling high" is the strategy for a long position.    
        for idx, row in self.patterns_df.iterrows():
            label = row['Label']
            price = row['Price']
            time = idx  # Access the time from the index directly
            
            start_pos = self.df.index.get_loc(idx)
            if start_pos < traintest_data_len:
                continue
            
            if label[1] == 'L':
                if not in_long_position:
                    # Buy in at this point
                    buy_price = price
                    buy_time = idx
                    in_long_position = True
                    log.debug(f"At {time}, LONG buy  price: {buy_price:.2f} at {label} point")
                else:
                    log.debug(f"At {time}, already in long position, ignoring signal {label} at price: {buy_price:.2f}")
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
                        section_df = cut_slice(self.df, sell_time, long_traintest_data_len)                    
                            
                        if (section_df is not None):
                            log.debug(f"Sliced DataFrame:{len(section_df)}\n {section_df}")
                            long_list.append(section_df) 
                            
                        in_long_position = False
                        log.debug(f"At {time}, LONG sell price: {sell_price:.2f} at {label} point, Profit: {profit:.2f}, Hold Time: {hold_time}")
                    
                        continue
                    else:
                        # if profit not > 0, just drop this L/H pair
                        in_long_position = False
                        log.debug(f"At {time}, NO sell at price: {sell_price:.2f} at {label} point, Profit: {profit:.2f}, ignore this pair.")         
                else:
                    log.debug(f"At {time}, Not in position, ignore sell signal at {label} point")
                    continue        
            else:
                log.error(f"Error: Not sure how to process this point at {time}, Label: {label}\n")
        
        return long_list

    def check_short_patterns(self, short_traintest_data_len):
        traintest_data_len = int(config.traintest_data_len)
        shorttradecost = float(config.shorttradecost)
        short_list = []
            
        # essentially, "selling high and buying low" is the strategy for a short position.    
        in_short_position = False
        previous_point = None
        buy_time = None
        sell_time = None      
        hold_time = None  
        # Loop through the DataFrame and process each row for SHORT position (做空)
        for idx, row in self.patterns_df.iterrows():
            label = row['Label']
            price = row['Price']
            time = idx  # Access the time from the index directly
            
            start_pos = self.df.index.get_loc(idx)
            if start_pos < traintest_data_len:
                continue
            
            if label[1] == 'H':
                if not in_short_position:
                    # Buy in at this point
                    buy_price = price
                    buy_time = time
                    in_short_position = True
                    log.debug(f"At {time}, SHORT sell price: {buy_price:.2f} at {label} point")
                else:
                    in_short_position = False
                    log.debug(f"At {time}, already in SHORT position, ignoring signal {label} at price: {buy_price:.2f}. Ignore this pair.")
                    continue
            
            elif label[1] == 'L':
                if in_short_position:
                    # Sell out at this point
                    sell_price = price
                    sell_time = idx
                    hold_time = sell_time - buy_time                   
                    profit = -1 * (sell_price - buy_price) - shorttradecost
                    if profit > 0: 
                        section_df = cut_slice(self.df, sell_time, short_traintest_data_len)
                        #section_df = cut_slice(ohlc_df, buy_time, sell_time)
                            
                        if (section_df is not None):
                            log.debug(f"Sliced DataFrame:{len(section_df)}\n {section_df}")
                            short_list.append(section_df) 
                        
                        in_short_position = False
                        log.debug(f"At {time}, SHORT buy  price: {sell_price:.2f} at {label} point, Profit: {profit:.2f}, Hold Time: {hold_time}")
                        
                        continue
                    else:
                        # if profit not > 0 , just drop this L/H pair
                        in_short_position = False
                        log.debug(f"At {time}, NO sell at price: {sell_price:.2f} at {label} point, Profit: {profit:.2f}")         
                else:
                    log.debug(f"At {time}, Not in position, ignore sell signal at {label} point")
                    continue
            
            else:
                log.error(f"Error: Not sure how to process this point at {time}, Label: {label}\n")

        return short_list

    def generateTrain(self, tddf_short_list,tddf_long_list,traintest_data_len):
        SN = DataSource.config.sn
        data_dir = DataSource.config.data_dir
        table_name = DataSource.config.table_name
        td_file = os.path.join(data_dir, \
        f"{table_name}_TrainingData_HL_{traintest_data_len}_{SN}.txt")
    
        log.info(td_file)

        with open(td_file, "w") as datafile:
            generate_training_data(tddf_short_list, TradePosition.LONG, datafile)
            generate_training_data(tddf_long_list, TradePosition.SHORT, datafile)

    def generateTest(self, tddf_short_list,tddf_long_list,traintest_data_len):
        SN = DataSource.config.sn
        data_dir = DataSource.config.data_dir
        table_name = DataSource.config.table_name
        td_file = os.path.join(data_dir, \
            f"{table_name}_TestingData_HL_{traintest_data_len}_{SN}.txt")
        
        log.info(td_file)

        with open(td_file, "w") as datafile:
            generate_testing_data(tddf_short_list, TradePosition.LONG, datafile)
            generate_testing_data(tddf_long_list, TradePosition.SHORT, datafile)


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

def gen_list(processing_df):
    price_list = []

    processing_df['Normalized_Price'] = normalize(processing_df['Close'])
    
    log.debug(processing_df)
    
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
        log.debug(result)
        csvfile.write(result)
        return
    
    if (TradePosition is TradePosition.LONG):
        #result = "1," + trainingdata_str + "\n"
        result = f"{TradePosition.LONG.value}," + trainingdata_str + "\n"
        log.debug(result)
        csvfile.write(result)

    return

def write_testing_data(TradePosition, trainingdata_str, csvfile):
    # for testing data, the first number is index of "LONG, SHORT" series!
    # so if it's LONG, SHORT is 1;
    
    #trainingdata_str = list_to_string(data_list)
   
    if (TradePosition is TradePosition.LONG):
        result = "1," + trainingdata_str + "\n"
        log.debug(result)
    
        csvfile.write(result)
        return

        
    if (TradePosition is TradePosition.SHORT):        
        result = "-1," + trainingdata_str + "\n"
        log.debug(result)
        
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

def generate_training_data(tddf_highlow_list, position, datafile):
    
    filename = 'gru/log/TrainingDataGenLog_'+ str(position)+".log"
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
        log.debug(training_data_string)
        
                
        log.debug("\nGenerate training/testing data:")
        
        # Write lengths to the file in the desired format
        outputfile.write(
            f"{len(processing_df)}\n"
        ) 
        
        write_traintest_data(position, training_data_string, datafile)
    
        #print(i)
    outputfile.close()    
    return

def generate_testing_data(tddf_highlow_list, position,datafile):
    
    filename = 'gru/log/TestingDataGenLog_'+ str(position)+".log"
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
        log.debug(testing_data_string)
                        
        log.debug("\nGenerate training/testing data:")
        
        # Write lengths to the file in the desired format
        outputfile.write(
            f"{len(processing_df)}\n"
        ) 
        
        write_traintest_data(position, testing_data_string, datafile)
    
        #print(i)
    outputfile.close()    


@execution_time
def main():
    DataProcessor(training=True)
    DataProcessor(training=False)
    DataSource.conn.close()
    log.info("================================ Done")

# ================================================================================#
if __name__ == "__main__":
    log = Logger('gru/log/gru.log')
    log.info(f'sqlite version: {pd.__version__}')

    config = Config('gru/src/config.ini')

    main()
