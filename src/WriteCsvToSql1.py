import os
import time
import pandas as pd
import sqlite3
from datetime import datetime

def measure_operation_time(operation_func, *args, **kwargs):
    """
    Measure the time taken by a given operation function.
    
    Parameters:
        operation_func (function): The function to measure.
        *args: Positional arguments to pass to the operation function.
        **kwargs: Keyword arguments to pass to the operation function.
        
    Returns:
        tuple: A tuple containing the time elapsed in seconds and the result(s) of the operation function.
    """
    start_time = time.time()
    results = operation_func(*args, **kwargs)
    end_time = time.time()
    time_elapsed = end_time - start_time
    return time_elapsed, results


def read_CSV_file(csv_file_path):
    # Read data from the CSV file into a DataFrame with specified column names
    df = pd.read_csv(csv_file_path, header=None, names=["Datetime", "Open", "High", "Low", "Close", "Volume"])
    
    # Convert the 'Datetime' column to the desired format
    df['Datetime'] = df['Datetime'].apply(lambda x: datetime.strptime(x, "%Y%m%d %H%M%S").strftime("%Y-%m-%d %H:%M:%S"))
    
    return df


# Define the path to the CSV file
#csv_file_path = os.path.join("stockdata", "DAT_ASCII_SPXUSD_M1_2010.csv")
#csv_file_path = os.path.join("stockdata", "DAT_ASCII_SPXUSD_M1_2011.csv")
csv_file_path = os.path.join("stockdata", "DAT_ASCII_SPXUSD_M1_2012.csv")

# Define the table name as a string variable
table_name = "SPY_1m"

# Measure the time to read data from the CSV file into a DataFrame
time_elapsed, ohlcv = measure_operation_time(read_CSV_file, csv_file_path)
print("Time elapsed:", time_elapsed, "seconds")

print(ohlcv)
print("\n\n=============================1===Create/Insert=======================\n\n")

# Define the SQLite database file
db_file = os.path.join("stockdata", "stock__bigdata.db")

# Connect to the SQLite database
conn = sqlite3.connect(db_file)
cursor = conn.cursor()

# Create the table using the string variable
create_table_query = f'''
CREATE TABLE IF NOT EXISTS {table_name} (
    Datetime TEXT PRIMARY KEY,
    Open REAL,
    High REAL,
    Low REAL,
    Close REAL,
    Volume INTEGER
)
'''
cursor.execute(create_table_query)

# Insert data into the table using the string variable
for index, row in ohlcv.iterrows():
    insert_query = f'''
    INSERT OR REPLACE INTO {table_name} (Datetime, Open, High, Low, Close, Volume)
    VALUES (?, ?, ?, ?, ?, ?)
    '''
    cursor.execute(insert_query, (row['Datetime'], row['Open'], row['High'], row['Low'], row['Close'], row['Volume']))

# Commit the changes
conn.commit()


print("\n\n=============================2===Query/Count=======================\n\n")

count_query = f'''
SELECT COUNT(*) FROM {table_name}
'''
cursor.execute(count_query)
count_result = cursor.fetchone()[0]
print(f"Number of records in {table_name} table:", count_result)


# Close the connection
conn.close()