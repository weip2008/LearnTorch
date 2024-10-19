"""
This version read source data from SQLite database tables
and write dataset to file which name is defined in config.ini
"""
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import xgboost as xgb
import shap

from logger import Logger
from config import Config, execution_time
from utilities import DataSource, normalize,smooth_sma

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

class DataProcessor:
    slice_length = 76
    def __init__(self, training=True):
        self.target_map = {'short':[1.,0.,0.], 'hold':[0.,1.,0.], 'long':[0.,0.,1.]}
        if not training:
            self.target_map = {'short':[0], 'hold':[1], 'long':[2]}
        self.training = training

    def main(self):
        self.df = self.getDataFrame()
        self.ds.getZigzag()
        self.ds.getHoldZigzag()
        long_list, short_list, hold_list = self.ds.slice()
        self.write(long_list,short_list,hold_list,self.training)
        # self.write2file(long_list,short_list,hold_list, self.training)
        log.info(f"DataProcessor for {'Training' if self.training else 'Testing'} ========================================= Done.\n")

    def write(self, long_list, short_list, hold_list, training=True):
        filepath = config.training_file_path
        if not training:
            filepath = config.testing_file_path

        num_cols = int(config.num_cols)
        dict_list = self.buildDictionaryList(long_list, short_list, hold_list, num_cols, training)
        dataset = StockDataset(dict_list, num_cols)
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

    def writeList2File(self, f, list, type):
        for df in list:
            # Flatten the DataFrame values and create a new list starting with '1,0,0'
            flattened_data = self.target_map[type] + df.values.flatten().tolist()
            
            # Convert the list to a comma-separated string
            line = ','.join(map(str, flattened_data))
            
            # Write the string to the file followed by a newline
            f.write(line + '\n')
        
    def buildDictionaryList(self, long_list, short_list, hold_list, num_cols, training=True):
        log.info(f"{'Training' if training else 'Testing'} data:")
        log.info(f"long points: {len(long_list)}\nshort points: {len(short_list)}\nhold points: {len(hold_list)}")
        combined_list = []
        slice_len = int(config.slice_length) + 1

        # Helper function to process each list with corresponding label
        def process_list(data_list, label):
            list_dict = []
            for df in data_list:
                # Flatten the DataFrame values and create a feature list
                flattened_data = df.values.flatten().tolist()

                # Split flattened data into groups of 8 for the feature
                feature = [flattened_data[i:i+num_cols] for i in range(0, len(flattened_data), num_cols)]

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

    def getDataFrame(self):
        self.ds = DataSource()
        self.query_start, self.query_end= DataSource.config.training_start_date, DataSource.config.training_end_date
        if not self.training:
            self.query_start, self.query_end= DataSource.config.testing_start_date,DataSource.config.testing_end_date

        self.ds.queryDB(self.query_start, self.query_end)
        # self.ds.df.drop(columns=["Volume","Datetime"], inplace=True) 
        self.ds.df.drop(columns=["Volume","Datetime","Open","High","Low"], inplace=True) 
        return self.ds.getDataFrameFromDB()

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
    # plot("STOCHRSIk_70_70_35_35")
    # plot("STOCHRSId_70_70_35_35")
    plot("MACD_12_26_9")
    plot("MACDs_12_26_9")
    plt = plot("MACDh_12_26_9", True)
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
    macdh = features[:,3]
    macd = features[:,2]
    macds = features[:,4]

    # Plotting the second column
    plt.plot(data.numpy(), label='EMA')
    plt.plot(macdh.numpy(),label="Histogram")
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
    DataProcessor().main()
    DataProcessor(training=False).main()
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

def features():
    dp = DataProcessor()
    df = dp.getDataFrame()
    # Encode categorical features if necessary
    data = pd.get_dummies(df, drop_first=True)  # This will convert categorical variables to dummy variables

    # Ensure to use the correct target column name
    y = data['Close']  # Use the actual column name for the stock price
    X = data.drop(columns=['Close', 'Close_SMA_9'])  # Drop the target column from features
    # X = normalize_column(X, ['MACDh_12_26_9','VOLATILITY'])
    # X['MACDh_12_26_9'] = X['MACDh_12_26_9']*5

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the XGBoost model
    model = xgb.XGBRegressor(objective='reg:squarederror')
    model.fit(X_train, y_train)

    # Create a SHAP explainer
    explainer = shap.Explainer(model)

    # Compute SHAP values for the test set
    shap_values = explainer(X_test)
    # Assuming 'shap_values' is the SHAP explanation object
    # Extract SHAP values and feature names
    shap_values_matrix = shap_values.values  # Get the SHAP values matrix
    feature_names = shap_values.feature_names  # Get feature names

    # Compute the mean absolute SHAP values for each feature
    mean_abs_shap_values = np.abs(shap_values_matrix).mean(axis=0)

    # Get the indices of the top 3 features
    top_3_indices = np.argsort(mean_abs_shap_values)[-3:]

    # Extract the top 3 feature names
    top_3_features = np.array(feature_names)[top_3_indices]

    # Print the top 3 most important features
    print("Top 3 Features:", top_3_features)
    
    # Visualize the SHAP values
    shap.summary_plot(shap_values, X_test)

    # Assuming you have trained a tree-based model (e.g., XGBoost) and have data (X_test)
    explainer = shap.TreeExplainer(model)

    # Compute SHAP interaction values
    shap_interaction_values = explainer.shap_interaction_values(X_test)

    # Plot an interaction summary plot (useful for global interaction insights)
    shap.summary_plot(shap_interaction_values, X_test)

    # Plot a dependence plot with interaction
    # For example: See how feature 0 interacts with feature 1
    shap.dependence_plot(0, shap_values.values, X_test, interaction_index=1)

def saveSHAP(file, shap_values):
    import pickle
    # Save shap_values to a file
    with open(file, 'wb') as f:
        pickle.dump(shap_values, f)
    
if __name__ == "__main__":
    log = Logger('gru/log/gru.log', logger_name='data')
    log.info(f'sqlite version: {pd.__version__}')

    config = Config('gru/src/config.ini')

    funcs = {1:main, 2:plotMACD_RSI, 3:plotIndex, 4:plotZigzag, 5:slice, 6:plot, 7:plotSlice, 8:features}

    funcs[1]()

    # funcs[7](860, 8) # 1~401=buy; 402-802=sell; 803~1308=hold

    # long,short,hold = funcs[5]()
    # print(f'long list length: {len(long)}; \nshort list length: {len(short)}\nhold list length: {len(hold)}')

    # plt = funcs[6](yLabel = "Close")
    # plt.show()
