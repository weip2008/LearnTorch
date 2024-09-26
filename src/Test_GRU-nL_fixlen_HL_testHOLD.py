import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import time
from datetime import datetime
import numpy as np
import random
import sqlite3
from datetime import datetime
from datetime import timedelta
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from enum import Enum
import zigzagplus1 as zz


IsDebug = True

#testing_file_path  = 'data/SPX_1m_TestingData_HL_80_400.txt'
testing_file_path  = 'data/SPX_1m_PredictData_HL_54_750.txt'
model_save_path = 'models/GRU_model_with_LH_fixlen_data_800.pth'
sample_size = 20

testing_start_date = "2024-05-27"
testing_end_date = "2024-09-22"

# ZigZag parameters
deviation = 0.0010  # Percentage

#Trainning Data Length
# average number of working days in a month is 21.7, based on a five-day workweek
# so 45 days is total for two months working days
# 200 days is one year working days
traintest_data_len = 54    


# Define the table name as a string variable
table_name = "SPX_1m"
#table_name = "MES=F_1m"
# Define the SQLite database file directory
data_dir = "data"

db_file = os.path.join(data_dir, "stock_data_2024.db")

# tradecost for each trade
longtradecost = 1.00
shorttradecost = 1.00


class TradePosition(Enum):
    LONG = 1
    HOLD = 0
    SHORT = -1


def gen_zigzag_patterns(query_start, query_end):
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
        print(f"Patterns dataframe length:{len(patterns_df)}\n",patterns_df.head(20))  # Print to verify DataFrame structure

    zz.plot_patterns(ohlc_df, patterns_df)

    return ohlc_df, patterns_df
    

def generate_hold_list(ohlc_df, patterns_df):
    hold_df = pd.DataFrame(columns=ohlc_df.columns)  # Empty DataFrame to store results
    
    for i in range(0, len(patterns_df) - 1, 2):  # Step by 2 to get pairs
        # Get the first and second datetime indices
        dt1 = patterns_df.index[i]
        dt2 = patterns_df.index[i+1]
        
        # Calculate the middle datetime
        mid_dt = dt1 + (dt2 - dt1) / 2
        
        # Get the row from ohlc_df corresponding to the middle datetime
        if mid_dt in ohlc_df.index:
            row = ohlc_df.loc[[mid_dt]]  # Get row as DataFrame for proper concatenation
            if hold_df.empty:
                hold_df = row  # Initialize hold_df with the first valid row
            else:
                hold_df = pd.concat([hold_df, row], axis=0)  # Concatenate non-empty DataFrames
    
    return hold_df


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
   
    section_df.drop(['Open', 'High', 'Low', 'AdjClose', 'Volume'], axis=1, inplace=True)
    
    return section_df

def load_testing_data(training_file_path):
    data = []
    signals = []

    with open(training_file_path, 'r') as file:
        for line in file:
            # Split the line into data and target parts
            signals_part, data_part = line.strip().split(',[')
            
            signal = int(signals_part.strip())
            signals.append(signal)
            
            # Add the beginning bracket to the data part and opening bracket to the target part
            data_part = '[' + data_part
            
            # Convert the string representations to actual lists
            data_row = eval(data_part)
            
            # Append to the respective lists
            data.append(data_row)
            #targets.append(target_row[0])  # Ensure target_row is a 1D array
    
    # Convert lists to numpy arrays
    data_np = np.array(data)
    signals_np = np.array(signals).reshape(-1, 1)  # Reshape to (6883, 1)
    #signals_np = np.array(signals)
    
    return data_np, signals_np

def cut_hold_list(ohlc_df, patterns_df, data_len):
    hold_list = []

    for idx, row in patterns_df.iterrows():        
        section_df = cut_slice(ohlc_df, idx, data_len)                                            
        if (section_df is not None):
            hold_list.append(section_df) 
            
    return hold_list




# Normalization function
def normalize(series):
    return (series - series.min()) / (series.max() - series.min())

    
def convert_to_day_and_time(timestamp):
    # Get the day of the week (Monday=0, Sunday=6)
    day_of_week_numeric = timestamp.weekday() + 1

    # Convert the timestamp to a datetime object (to handle timezone)
    dt = timestamp.to_pydatetime()

    # Calculate the time in float format
    time_float = dt.hour + dt.minute / 60 + dt.second / 3600

    return day_of_week_numeric, time_float


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

def generate_holding_data(tddf_highlow_list, IsDebug=False):
    data = []
    signals = []
 
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
        if IsDebug:
            print(testing_data_string)
        
        # Convert the string representations to actual lists
        data_row = eval(testing_data_string)
        
        # Append to the respective lists
        data.append(data_row)
        
        signal = int(0)
        signals.append(signal)

    data_np = np.array(data)
    signals_np = np.array(signals).reshape(-1, 1)  # Reshape to (6883, 1)
    #signals_np = np.array(signals)
    
    return data_np, signals_np



print(f"Current date and time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"1. Load testing data from {table_name} table")

ohlc_df, patterns_df = gen_zigzag_patterns(testing_start_date, testing_end_date)

hold_df = generate_hold_list(ohlc_df, patterns_df)
if IsDebug:
        print(f"hold_list dataframe length:{len(hold_df)}\n",hold_df.head(20))
        
hold_list = cut_hold_list(ohlc_df, hold_df, traintest_data_len)

if IsDebug:
        print(f"hold_list dataframe length:{len(hold_list)}\n")

# Example usage
print(f"Current date and time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


testing_data, testing_signals = generate_holding_data(hold_list)

print("Data shape:", testing_data.shape)
print("Targets shape:", testing_signals.shape)


# Custom dataset class for loading signals and data
class TimeSeriesDataset(Dataset):
    def __init__(self, data, signals):
        self.data = data
        self.signals = signals

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.signals[idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

# Instantiate the dataset
print("2. Define dataset and dataloader")
print(f"Current date and time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
test_dataset = TimeSeriesDataset(testing_data, testing_signals)

# Create DataLoader for batching
batch_size = 32  # You can change the batch size as needed

# Test dataloader with shuffling
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Define the GRU model with 2 layers
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.gru(x, h0)
        out = self.fc(out[:, -1, :])
        return out

# Instantiate the model, define the loss function and the optimizer
print("3. Instantiate the model, define the loss function and the optimize")
print(f"Current date and time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Define hyperparameters
input_size = 5  # Number of features in each tuple
hidden_size = 64  # Number of features in the hidden state
output_size = 1  # Number of output features (signal)
num_layers = 5    # Number of GRU layers


# Instantiate the model
print(f"Number of layers: {num_layers}")
model = GRUModel(input_size, hidden_size, output_size, num_layers)

# Load the saved model state
print(f"3. Load trained model from {model_save_path}")
checkpoint = torch.load(model_save_path)
model.load_state_dict(checkpoint['model_state_dict'])
#model.eval()  # Set the model to evaluation mode


# Function to categorize the model output
def categorize_output(output):
    if 0.6 <= output <= 1.3:
        return 1.0
    elif -1.3 <= output <= -0.6:
        return -1.0
    else:
        return 0.0

# Function to get the model output for a single input row
def get_model_output(single_input):
    single_input_tensor = torch.tensor(single_input, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():  # No need for gradients during testing
        test_output = model(single_input_tensor)
    return test_output.item()  # Return the single output as a scalar


# Randomly select 10 rows from testing data
random_indices = random.sample(range(len(testing_data)), sample_size)
random_datas = testing_data[random_indices]
random_targets = testing_signals[random_indices]


# Print the output for each selected row
print("Randomly selected 10 rows and their corresponding outputs:")
for i in range(sample_size):
    test_data = random_datas[i]
    test_target = random_targets[i].item()  # Get the actual target value
    
    # Call get_model_output to get the predicted output
    test_output = get_model_output(test_data)
    
    # Call categorize_output to categorize the predicted output
    categorized_output = categorize_output(test_output)
    
    # Print the test output, categorized output, and test target
    print(f"Test Output: {test_output:.4f} => Categorized Output: {categorized_output}, \tTarget: {test_target}")


