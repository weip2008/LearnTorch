# This version read source data from SQLite database tables
#
# [1,0,0] stand for short(sell)
# [0,1,0] stand for hold
# [0,0,1] stand for long(buy)
# [0,1,1,0,2,2] test data means 4 set of [sell, hold, hold, sell, buy, buy]

import os
import torch
import pandas as pd
import numpy as np
import statistics
from enum import Enum
import matplotlib.pyplot as plt
from torch.utils.data import Dataset

import zigzagplus1 as zz
from logger import Logger
from config import Config, execution_time
from utilities import DataSource, convert_to_day_and_time,normalize,smooth_sma

class TradePosition(Enum):
    LONG = 1
    SHORT = -1

class Trade:
    def __init__(self, open_price, open_time, trade_cost, label):
        self.open_price, self.open_time = open_price, open_time
        self.trade_cost = trade_cost
        log.debug(f"At {open_time}, LONG buy  price: {open_price:.2f} at {label} point")


    def hold_minutes(self, close_price, close_time, label):
        hold_time = 0
        profit = self.profit(close_price, label)
        if (profit > 0):
            hold_time = (close_time - self.open_time).total_seconds() / 60
            log.debug(f"At {close_time}, LONG sell price: {close_price:.2f} at {label} point, Profit: {profit:.2f}, Hold Time: {hold_time}")

        return hold_time

    def profit(self, close, label):
        profit = close - self.open_price - self.trade_cost # 低买/高卖 (做多Long)
        if label[1] == 'L':
            profit = self.open_price - close - self.trade_cost # 高卖/低买 （做空Short）
        return profit
    
    def cut_slice(self, df, close_price, close_time, label, slice_length):
        if (self.profit(close_price, label)>0):
            section_df = cut_slice(df, close_time, slice_length)  
            return section_df
        return None

class StockDataset(Dataset):
    def __init__(self, data_list, num_cols=8, transform=None):
        self.data_list = data_list  # list of dictionaries with "feature" and "target" keys
        self.num_cols = num_cols  # number of values per group (e.g., 8)
        self.transform = transform

    def __len__(self):
        return len(self.data_list)  # number of dictionaries (instances)

    def __getitem__(self, idx):
        # Get the dictionary for the current instance
        data_dict = self.data_list[idx]

        # Extract the features from the dictionary (list of groups, each with 8 values)
        features = data_dict["feature"]  # assuming a 2D list (groups, 8 values each)
        features = np.array(features).astype(float)  # convert to numpy array
        
        # Ensure each group has 8 values (num_cols)
        assert features.shape[1] == self.num_cols, f"Each group should have {self.num_cols} values"

        # Extract the target from the dictionary
        target = data_dict["target"]  # target can be scalar or array

        # Convert to torch tensors
        features = torch.tensor(features, dtype=torch.float32)  # features tensor (n_groups, num_cols)
        target = torch.tensor(target, dtype=torch.float32)  # target tensor (scalar or multi-label)

        if self.transform:
            features = self.transform(features)

        return features, target
    
    def get_shapes(self):
        features, target = self.__getitem__(0)
        return features.shape, target.shape
    
    def get_shapes(self):
        features, target = self.__getitem__(0)
        return features.shape, target.shape
        
class DataProcessor:
    slice_length = 76
    def __init__(self, training=True):
        self.target_map = {'short':[1.,0.,0.], 'hold':[0.,1.,0.], 'long':[0.,0.,1.]}
        if not training:
            self.target_map = {'short':[0], 'hold':[1], 'long':[2]}

        self.df = self.getDataFrame(training)
        self.ds.getZigzag()
        self.ds.getHoldZigzag()
        long_list, short_list, hold_list = self.ds.slice()
        self.normalize(long_list,short_list,hold_list)
        self.write(long_list,short_list,hold_list,training)
        # self.write2file(long_list,short_list,hold_list, training)
        log.info(f"DataProcessor for {'Training' if training else 'Testing'} ========================================= Done.\n")

    def normalize(self, long_list, short_list, hold_list):
        from concurrent.futures import ThreadPoolExecutor

        def normalize_column(df, exclude_cols):
            # Separate columns to exclude from normalization
            exclude_columns = df[exclude_cols]
            
            # Select numeric columns excluding those in `exclude_cols`
            numeric_cols = df.drop(columns=exclude_cols).select_dtypes(include='number')
            
            # Normalize only the remaining numeric columns
            normalized_numeric_cols = (numeric_cols - numeric_cols.min()) / (numeric_cols.max() - numeric_cols.min())
            
            # Concatenate excluded columns with normalized numeric columns
            return pd.concat([exclude_columns, normalized_numeric_cols], axis=1)

        def normalize_data_list(data_list, exclude_cols):
            for i in range(len(data_list)):
                df = data_list[i].copy()
                df = normalize_column(df, exclude_cols)  # Normalize while excluding specific columns
                data_list[i] = df  # Update the original DataFrame in the list

        def normalize_parallel(exclude_cols, *lists):
            with ThreadPoolExecutor() as executor:
                executor.map(lambda data_list: normalize_data_list(data_list, exclude_cols), lists)

        normalize_parallel(["Datetime","MACDh_12_26_9"], long_list, short_list, hold_list)

    def write(self, long_list, short_list, hold_list, training=True):
        filepath = config.training_file_path
        if not training:
            filepath = config.testing_file_path

        dict_list = self.buildDictionaryList(long_list, short_list, hold_list, training)
        # Create dataset
        dataset = StockDataset(data_list=dict_list)
        torch.save(dataset, filepath)
        log.info(f"Dataset has been saved to {filepath}.")

    def write2file(self, long_list, short_list, hold_list, training=True):
        filepath = config.training_file_path
        if not training:
            filepath = config.testing_file_path

        # Open the file for writing
        with open(filepath, 'w') as f:
            self.writeList2File(f, long_list, 'long')
            self.writeList2File(f, short_list, 'short')
            self.writeList2File(f, hold_list, 'hold')
        log.info(f"Dataset has been saved to {filepath}.")

    def date2minutes(self, df):
        tmp = pd.DataFrame(df)
        tmp['Datetime'] = pd.to_datetime(tmp['Datetime'])

        # Extract weekday (as an integer where Monday=0, Sunday=6)
        tmp['Weekday'] = tmp['Datetime'].dt.weekday  # Or use df['Datetime'].dt.day_name() for names

        # Convert time to total minutes (hours * 60 + minutes)
        tmp['Time_in_minutes'] = tmp['Datetime'].dt.hour * 60 + tmp['Datetime'].dt.minute

        # Drop the original 'Datetime' column
        tmp.drop(columns=['Datetime'], inplace=True)
        tmp.drop(columns=['Close'], inplace=True)
        return tmp

    def writeList2File(self, f, list, type):
        for df in list:
            # Flatten the DataFrame values and create a new list starting with '1,0,0'
            flattened_data = self.target_map[type] + df.values.flatten().tolist()
            
            # Convert the list to a comma-separated string
            line = ','.join(map(str, flattened_data))
            
            # Write the string to the file followed by a newline
            f.write(line + '\n')
        
    def zhouhao(self, training=True):

        self.gen_zigzag_patterns()
        DataProcessor.slice_length = self.estimateSliceLength() # 得到切片长度

        # tddf_long_list, tddf_short_list = self.create_data_list(DataProcessor.slice_length)

        # if training:
        #     self.generateTrain(tddf_short_list, tddf_long_list, DataProcessor.slice_length)
        # else:
        #     self.generateTest(tddf_short_list, tddf_long_list, DataProcessor.slice_length)

    def buildDictionaryList(self, long_list, short_list, hold_list, training=True):
        log.info(f"{'Training' if training else 'Testing'} data:")
        log.info(f"long points: {len(long_list)}\nshort points: {len(short_list)}\nhold points: {len(hold_list)}")
        combined_list = []
        slice_len = int(config.slice_length) + 1

        # Helper function to process each list with corresponding label
        def process_list(data_list, label):
            list_dict = []
            for df in data_list:
                df = self.date2minutes(df)

                # Flatten the DataFrame values and create a feature list
                flattened_data = df.values.flatten().tolist()

                # Split flattened data into groups of 8 for the feature
                feature = [flattened_data[i:i+8] for i in range(0, len(flattened_data), 8)]

                # Create a dictionary with "feature" and "target"
                data_dict = {
                    "feature": feature,  # 2D list of (groups, 8 values each)
                    "target": label  # target label
                }

                # Append the dictionary to the list
                list_dict.append(data_dict)
            return list_dict

        # Process each list with corresponding label
        combined_list.extend(process_list(short_list, self.target_map['short']))  # For short_list
        combined_list.extend(process_list(long_list, self.target_map['long']))  # For long_list
        combined_list.extend(process_list(hold_list, self.target_map['hold']))  # For hold_list

        return combined_list

    def getDataFrame(self, training):
        self.ds = DataSource()
        self.query_start, self.query_end= DataSource.config.training_start_date, DataSource.config.training_end_date
        if not training:
            self.query_start, self.query_end= DataSource.config.testing_start_date,DataSource.config.testing_end_date

        self.ds.queryDB(self.query_start, self.query_end)
        return self.ds.getDataFrameFromDB()

    def gen_zigzag_patterns(self):
        deviation = float(config.deviation)
        self.df.set_index("Datetime", inplace=True)
        zigzag = zz.calculate_zigzag(self.df, deviation)
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

    def statistics(self, holdtime_list, type):
        # Mean hold time
        mean_hold_time = statistics.mean(holdtime_list)

        # Median hold time
        median_hold_time = statistics.median(holdtime_list)

        # Standard deviation
        std_dev_hold_time = statistics.stdev(holdtime_list)

        log.info(type)
        log.info(f"Mean Hold Time: {mean_hold_time}")
        log.info(f"Median Hold Time: {median_hold_time}")
        log.info(f"Standard Deviation: {std_dev_hold_time}")
        
        long_hold_time = int(np.ceil(mean_hold_time))
        return long_hold_time
    
    def estimateSliceLength(self):
        slice_length = int(config.slice_length)
        longtradecost = float(config.longtradecost)
        shorttradecost = float(config.shorttradecost)
        
        # Initialize variables
        at_long_position = False  # Track whether we are in a long (buy) position
        in_short_position = False  # Track whether we are in a short (sell) position
        long_holdtime_list = []
        short_holdtime_list = []
        
        # Loop through the DataFrame and process each row for both LONG and SHORT positions
        for time, row in self.patterns_df.iterrows():
            label = row['Label']
            price = row['Price']
            
            start_pos = self.df.index.get_loc(time)
            if start_pos < slice_length:
                continue

            if label[1] == 'L':
                if not at_long_position:
                    # Open a long position
                    long_trade = Trade(price, time, longtradecost, label)
                    at_long_position = True
                if in_short_position:
                    # Close the short position
                    short_hold_time = short_trade.hold_minutes(price, time, label)
                    if short_hold_time > 0: 
                        short_holdtime_list.append(short_hold_time)
                    in_short_position = False

            if label[1] == 'H':
                if not in_short_position:
                    # Open a short position
                    short_trade = Trade(price, time, shorttradecost, label)
                    in_short_position = True
                if at_long_position:
                    # Close the long position
                    long_hold_time = long_trade.hold_minutes(price, time, label)
                    if long_hold_time > 0: 
                        long_holdtime_list.append(long_hold_time)
                    at_long_position = False

        # Calculate hold times for both long and short positions
        long_hold_time = self.statistics(long_holdtime_list, "Long")
        short_hold_time = self.statistics(short_holdtime_list, "Short")

        # Calculate the average hold time
        avg_hold_time = int((long_hold_time + short_hold_time) / 2)
        log.info(f"Avg mean Hold Time: {avg_hold_time}")
        
        return avg_hold_time
   
    def create_data_list(self, slice_length):
        longtradecost = float(config.longtradecost)
        shorttradecost = float(config.shorttradecost)

        long_list = []
        short_list = []

        # Initialize variables
        in_long_position = False  # Track whether we are in a buy position for long trades
        in_short_position = False  # Track whether we are in a sell position for short trades

        # Loop through the DataFrame and process each row for both LONG and SHORT positions
        for time, row in self.patterns_df.iterrows():
            label = row['Label']
            price = row['Price']

            start_pos = self.df.index.get_loc(time)
            if start_pos < slice_length:
                continue

            # Check for LONG positions (buy low, sell high)
            if label[1] == 'L':
                if not in_long_position:
                    # Enter long position
                    long_trade = Trade(price, time, longtradecost, label)
                    in_long_position = True
                # Exit short position if in one
                if in_short_position:
                    section_df = short_trade.cut_slice(self.df, price, time, label, slice_length)
                    if section_df is not None:
                        log.debug(f"Sliced DataFrame for SHORT: {len(section_df)}\n {section_df}")
                        short_list.append(section_df)
                    in_short_position = False

            # Check for SHORT positions (sell high, buy low)
            if label[1] == 'H':
                if not in_short_position:
                    # Enter short position
                    short_trade = Trade(price, time, shorttradecost, label)
                    in_short_position = True
                # Exit long position if in one
                if in_long_position:
                    section_df = long_trade.cut_slice(self.df, price, time, label, slice_length)
                    if section_df is not None:
                        log.debug(f"Sliced DataFrame for LONG: {len(section_df)}\n {section_df}")
                        long_list.append(section_df)
                    in_long_position = False

        return long_list, short_list

    def generateTrain(self, tddf_short_list,tddf_long_list,slice_length):
        SN = DataSource.config.sn
        data_dir = DataSource.config.data_dir
        table_name = DataSource.config.table_name
        td_file = os.path.join(data_dir, \
        f"{table_name}_TrainingData_HL_{slice_length}_{SN}.txt")
    
        log.info(td_file)

        with open(td_file, "w") as datafile:
            generate_training_data(tddf_short_list, TradePosition.LONG, datafile)
            generate_training_data(tddf_long_list, TradePosition.SHORT, datafile)

    def generateTest(self, tddf_short_list,tddf_long_list,slice_length):
        SN = DataSource.config.sn
        data_dir = DataSource.config.data_dir
        table_name = DataSource.config.table_name
        td_file = os.path.join(data_dir, \
            f"{table_name}_TestingData_HL_{slice_length}_{SN}.txt")
        
        log.info(td_file)

        with open(td_file, "w") as datafile:
            generate_testing_data(tddf_short_list, TradePosition.LONG, datafile)
            generate_testing_data(tddf_long_list, TradePosition.SHORT, datafile)

def cut_slice(ohlc_df, end_index, slice_length):
    # Ensure the start_index and end_index are in the DataFrame index
    if end_index not in ohlc_df.index:
    #if start_index not in ohlc_df.index or end_index not in ohlc_df.index:
        # If either index is not found, return None
        return None
    
    # Get the positional indices of the timestamps
    end_pos = ohlc_df.index.get_loc(end_index)  
    #start_pos = ohlc_df.index.get_loc(start_index)
    start_pos = end_pos - slice_length 
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
    if (len(section_df) < (slice_length+2)):   
        return None
   
    # section_df.drop(['Open', 'High', 'Low', 'Volume'], axis=1, inplace=True)
    
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

def plot(yLabel="Close", zero_line=False):
    df = DataSource().queryDB(config.training_start_date, config.training_end_date,False).getDataFrameFromDB()
    # Plot the price list
    plt.plot(df[yLabel])
    if zero_line:
        plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Time Index')
    plt.ylabel(yLabel)
    plt.title(f'Plot for DataFrame')
    return plt

def plotMACD_RSI():
    plot("STOCHRSIk_70_70_35_35")
    plot("STOCHRSId_70_70_35_35")
    # plot("MACD_12_26_9")
    # plot("MACDs_12_26_9")
    # plt = plot("MACDh_12_26_9", True)
    plt.show()

def plotIndex():
    ds = DataSource()
    query_start, query_end = config.training_start_date, config.training_end_date
    ds.queryDB(query_start, query_end, False)
    df = ds.getDataFrameFromDB()
    
    # Plot original "Close" prices
    plt.plot(df["Close"], label="Original Close Price")
    
    # Plot smoothed "Close" prices
    plt.plot(df["Close_SMA_9"], label="9-Point Smooth (SMA)", color="orange")
    
    plt.xlabel("Index")
    plt.ylabel("Close Price")
    plt.legend()
    plt.show()

def plotZigzag():
    # Example usage
    query_start, query_end= DataSource.config.training_start_date, DataSource.config.training_end_date
    ds = DataSource()
    ds.queryDB(query_start,query_end, False).getDataFrameFromDB()  
    ds.getZigzag()
    ds.getHoldZigzag()

    ds.plot_zigzag()

def plotSlice(index, column):
    filepath = config.training_file_path
    training_dataset = torch.load(filepath)
    print(f"Total of {len(training_dataset)} rows.")
    features, targets = training_dataset[index]
    data = features[:, column-1]
    macdh = features[:,0]
    macd = features[:,4]
    macds = features[:,5]

    # Plotting the second column
    plt.plot(data.numpy(), label='Close SMA 9')
    plt.plot(macdh.numpy()*0.1,label="Histogram")
    plt.plot(macd.numpy(),label="MACD")
    plt.plot(macds.numpy(),label="Singnal")
    plt.axhline(y=0, color='r', linestyle='--', linewidth=2)
    plt.title(f'{column} Column of row {index}')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.show()

@execution_time
def main():
    DataProcessor()
    DataProcessor(training=False)
    DataSource.conn.close()
    log.info("main() ================================ Done")

@execution_time
def slice():
    query_start, query_end = DataSource.config.training_start_date, DataSource.config.training_end_date
    ds = DataSource()
    ds.queryDB(query_start, query_end, False).getDataFrameFromDB()
    ds.getZigzag()
    ds.getHoldZigzag()
    
    # Initialize lists for long and short positions
    long_list, short_list, hold_list = ds.slice()
    
    return long_list, short_list, hold_list

def estimateSliceLength():
    dp = DataProcessor()
    dp.zhouhao()

if __name__ == "__main__":
    log = Logger('gru/log/gru.log', logger_name='data')
    log.info(f'sqlite version: {pd.__version__}')

    config = Config('gru/src/config.ini')

    funcs = {1:main, 2:plotMACD_RSI, 3:plotIndex, 4:plotZigzag, 5:slice, 6:plot, 7:estimateSliceLength, 8:plotSlice}

    funcs[1]()

    # long,short,hold = funcs[5]()
    # print(f'long list length: {len(long)}; \nshort list length: {len(short)}\nhold list length: {len(hold)}')

    # plt = funcs[6](yLabel = "Close")
    # plt.show()
