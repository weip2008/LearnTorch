import pandas as pd
import sqlite3
import os

symbol = "SPX"

# Define the table name as a string variable
table_name_1m = "SPX_1m"
table_name_5m = "SPX_5m"

# Define the SQLite database file directory
data_dir = "data"
#db_file = os.path.join(data_dir, "stock_bigdata_2010-2023.db")
db_file = os.path.join(data_dir, "stock_bigdata_2019-2023.db")

# Define the start and end dates for the query
start_date = "2010-11-14"
end_date = "2023-12-30"

# Load data from the SQLite database
conn = sqlite3.connect(db_file)
cursor = conn.cursor()

# Query the data between the start_date and end_date
query_range = f'''
SELECT * FROM {table_name_1m}
WHERE Datetime BETWEEN ? AND ?
'''
query_result_df = pd.read_sql_query(query_range, conn, params=(start_date, end_date))

# Process the data
ohlc_df = query_result_df
ohlc_df['Datetime'] = pd.to_datetime(ohlc_df['Datetime'])
ohlc_df.set_index('Datetime', inplace=True)
ohlc_df = ohlc_df.drop(columns=['Volume'])

print(ohlc_df.head(10))
print(ohlc_df.tail(10))
print("==================================================")

# Resample to 5-minute intervals
ohlc_5min_df = ohlc_df.resample('5min').agg({
    'Open': 'first',
    'High': 'max',
    'Low': 'min',
    'Close': 'last'
})

# Reset the index if you need the Datetime column as a column instead of the index
ohlc_5min_df = ohlc_5min_df.reset_index()


# Print the first 10 and last 10 rows
print(ohlc_5min_df.head(10))
print(ohlc_5min_df.tail(10))


# Create a new table SPX_5m
create_table_query = f'''
CREATE TABLE IF NOT EXISTS {table_name_5m} (
    Datetime TEXT PRIMARY KEY,
    Open REAL,
    High REAL,
    Low REAL,
    Close REAL
)
'''
cursor.execute(create_table_query)

# Insert the data into the new table
ohlc_5min_df.to_sql(table_name_5m, conn, if_exists='replace', index=False)

# Commit the transaction and close the connection
conn.commit()
conn.close()
