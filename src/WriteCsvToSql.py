import sqlite3
import pandas as pd
import time
import os



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
    # Read data from the CSV file into a DataFrame
    df = pd.read_csv(csv_file_path)
    return df


# Define the path to the CSV file
#csv_file_path = os.path.join("data", "AAPL_2024-05-06_2024-05-12_1m.csv")
#csv_file_path = os.path.join("data", "AAPL_2024-04-12_2024-05-19_1m.csv")
#sv_file_path = os.path.join("data", "AAPL_2024-05-20_2024-05-26_1m.csv")
#csv_file_path = os.path.join("data\\SPY_2024-04-11_2024-05-26_1m.csv")
#csv_file_path = os.path.join("data\\SPY_2024-04-11_2024-05-26_1m.csv")

#csv_file_path = os.path.join("stockdata\\MES=F_max_2024-05-27_2024-06-03_1m.csv")
#csv_file_path = os.path.join("stockdata\\MES=F_max_2024-05-28_2024-06-04_1m.csv")
#csv_file_path = os.path.join("stockdata\\MES=F_max_2024-05-31_2024-06-07_1m.csv")
#csv_file_path = os.path.join("stockdata\\MES=F_max_2024-06-09_2024-06-14_1m.csv")
#csv_file_path = os.path.join("stockdata\\MES=F_max_2024-06-16_2024-06-21_1m.csv")
#csv_file_path = os.path.join("stockdata\\MES=F_max_2024-06-23_2024-06-28_1m.csv")
#csv_file_path = os.path.join("stockdata\\MES=F_max_2024-06-30_2024-07-07_1m.csv")
#csv_file_path = os.path.join("stockdata\\MES=F_max_2024-07-07_2024-07-12_1m.csv")
#csv_file_path = os.path.join("stockdata\\MES=F_max_2024-07-15_2024-07-22_1m.csv")
#csv_file_path = os.path.join("stockdata\\MES=F_max_2024-07-21_2024-07-26_1m.csv")
#csv_file_path = os.path.join("stockdata\\MES=F_max_2024-07-28_2024-08-02_1m.csv")
#csv_file_path = os.path.join("stockdata\\MES=F_max_2024-08-04_2024-08-09_1m.csv")
#csv_file_path = os.path.join("stockdata\\MES=F_max_2024-08-11_2024-08-16_1m.csv")
#csv_file_path = os.path.join("stockdata\\MES=F_max_2024-08-18_2024-08-23_1m.csv")
#csv_file_path = os.path.join("stockdata\\MES=F_max_2024-08-25_2024-08-30_1m.csv")
#csv_file_path = os.path.join("stockdata\\MES=F_max_2024-09-01_2024-09-06_1m.csv")
#csv_file_path = os.path.join("stockdata\\MES=F_max_2024-09-08_2024-09-13_1m.csv")
#csv_file_path = os.path.join("stockdata\\MES=F_max_2024-09-15_2024-09-22_1m.csv")
#csv_file_path = os.path.join("stockdata\\MES=F_max_2024-09-22_2024-09-27_1m.csv")
csv_file_path = os.path.join("stockdata\MES=F_max_2024-09-29_2024-10-04_1m.csv")

# Define the table name as a string variable
#table_name = "AAPL_1D"
#table_name = "AAPL_1m"
table_name = "SPX_1m"

# Read data from the CSV file into a DataFrame
#ohlcv = pd.read_csv(csv_file_path)
# Corrected usage: pass function object and its arguments separately
time_elapsed, ohlcv = measure_operation_time(read_CSV_file, csv_file_path)
print("Time elapsed:", time_elapsed, "seconds")

# Ensure the 'Datetime' column is in ISO 8601 format with timezone information
#ohlcv['Datetime'] = pd.to_datetime(ohlcv['Datetime']).dt.strftime('%Y-%m-%d %H:%M:%S%z')
print(ohlcv)
print("\n=============================1===Create/Insert=======================\n")

# Define the SQLite database file
db_file = os.path.join("data", "stock_data_2024.db")

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
    AdjClose REAL,
    Volume INTEGER
)
'''
cursor.execute(create_table_query)

print("\n=============================2===Query/Count=======================\n")


count_query = f'''
SELECT COUNT(*) FROM {table_name}
'''
cursor.execute(count_query)
count_result = cursor.fetchone()[0]
#print("\nNumber of records in AAPL_1m table:", count_result)
print(f"\nNumber of records in {table_name} table:", count_result)


print("\n=============================3===Insert=======================\n")
# Insert data into the table
# for index, row in ohlcv.iterrows():
#     cursor.execute('''
#         INSERT OR REPLACE INTO AAPL_1m (Datetime, Open, High, Low, Close, AdjClose, Volume)
#         VALUES (?, ?, ?, ?, ?, ?, ?)
#     ''', (row['Datetime'], row['Open'], row['High'], row['Low'], row['Close'], row['Adj Close'], row['Volume']))

# Insert data into the table using the string variable
for index, row in ohlcv.iterrows():
    insert_query = f'''
    INSERT OR REPLACE INTO {table_name} (Datetime, Open, High, Low, Close, AdjClose, Volume)
    VALUES (?, ?, ?, ?, ?, ?, ?)
    '''
    cursor.execute(insert_query,(row['Datetime'], row['Open'], row['High'], row['Low'], row['Close'], row['Adj Close'], row['Volume']))

# Commit the changes
conn.commit()

# Delete the ohlcv variable to save memory
del ohlcv

print("\n=============================4===Count=======================\n")


count_query = f'''
SELECT COUNT(*) FROM {table_name}
'''
cursor.execute(count_query)
count_result = cursor.fetchone()[0]
#print("\nNumber of records in AAPL_1m table:", count_result)
print(f"\nNumber of records in {table_name} table:", count_result)


# print("\n\n==========================3===Query==========================\n\n")


# # Query the data for May 6th, 2024, at 10 AM
# query_10am = f'''
# SELECT * FROM {table_name}
# WHERE Datetime LIKE '2024-05-06 09:5%'
# '''
# cursor.execute(query_10am)
# rows_10am = cursor.fetchall()

# print("Datatype of rows_10am:", type(rows_10am))
# print("Data for May 6th, 2024, at 10 AM:")
# for row in rows_10am:
#     print(row)


# print("\n\n==========================4===Query==========================\n\n")

# # Define the query date range
# query_start = "2024-05-06"
# query_end = "2024-05-12"

# # Query the data between May 6th, 2024, and May 12th, 2024
# # query_range = '''
# # SELECT * FROM AAPL_1m
# # WHERE Datetime BETWEEN ? AND ?
# # '''
# query_range = f'''
# SELECT * FROM {table_name}
# WHERE Datetime BETWEEN ? AND ?
# '''
# # Save the query result into a DataFrame object named query_result_df
# query_result_df = pd.read_sql_query(query_range, conn, params=(query_start, query_end))

# print("Length of query result is:", len(query_result_df))
# print("Datatype of query result:", type(query_result_df))
# print(query_result_df)

# Close the connection
conn.close()

