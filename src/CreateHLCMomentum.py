import sqlite3
import pandas as pd
import os
from datetime import datetime

now = datetime.now()
formatted_now = now.strftime("%Y-%m-%d %H:%M:%S")
print("Current date and time:", formatted_now)
    
# Define the table name and database file path
table_name = "SPX_1m"
data_dir = "data"  # Ensure this directory exists and the db_file is correctly placed
db_file = os.path.join(data_dir, "stock_bigdata_2019-2023.db")

# Ensure the data directory exists
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

# Function to load data from the SQLite database
def load_data(query_start, query_end):
    conn = sqlite3.connect(db_file)
    query_range = f'''
    SELECT * FROM {table_name}
    WHERE Datetime BETWEEN ? AND ?
    '''
    query_result_df = pd.read_sql_query(query_range, conn, params=(query_start, query_end))
    query_result_df['Datetime'] = pd.to_datetime(query_result_df['Datetime'])
    query_result_df.set_index('Datetime', inplace=True)
    ohlc_df = query_result_df.drop(columns=['Volume'])
    conn.close()
    return ohlc_df

# Function to create an index on the Datetime column if it doesn't exist
def create_index():
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_{table_name}_datetime ON {table_name} (Datetime)")
    conn.commit()
    conn.close()

# Function to create Close_momentum table
def create_close_momentum():
    start_date = "2019-01-01"
    end_date = "2023-12-31"
    ohlc_df = load_data(start_date, end_date)

    ohlc_df['Velocity'] = ohlc_df['Close'].diff()
    ohlc_df['Acceleration'] = ohlc_df['Velocity'].diff()
    ohlc_df.dropna(inplace=True)
    momentum_df = ohlc_df[['Velocity', 'Acceleration']]
    momentum_df.reset_index(inplace=True)

    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()

    cursor.execute("DROP TABLE IF EXISTS Close_momentum")
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS Close_momentum (
        Datetime TEXT PRIMARY KEY,
        Velocity REAL,
        Acceleration REAL
    )
    ''')
    conn.commit()

    momentum_df.to_sql('Close_momentum', conn, if_exists='replace', index=False)
    cursor.execute('''
    CREATE INDEX IF NOT EXISTS idx_Close_momentum_datetime ON Close_momentum (Datetime)
    ''')
    conn.commit()
    conn.close()

# Function to create High_momentum table
def create_high_momentum():
    start_date = "2019-01-01"
    end_date = "2023-12-31"
    ohlc_df = load_data(start_date, end_date)

    ohlc_df['Velocity'] = ohlc_df['High'].diff()
    ohlc_df['Acceleration'] = ohlc_df['Velocity'].diff()
    ohlc_df.dropna(inplace=True)
    momentum_df = ohlc_df[['Velocity', 'Acceleration']]
    momentum_df.reset_index(inplace=True)

    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()

    cursor.execute("DROP TABLE IF EXISTS High_momentum")
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS High_momentum (
        Datetime TEXT PRIMARY KEY,
        Velocity REAL,
        Acceleration REAL
    )
    ''')
    conn.commit()

    momentum_df.to_sql('High_momentum', conn, if_exists='replace', index=False)
    cursor.execute('''
    CREATE INDEX IF NOT EXISTS idx_High_momentum_datetime ON High_momentum (Datetime)
    ''')
    conn.commit()
    conn.close()

# Function to create Low_momentum table
def create_low_momentum():
    start_date = "2019-01-01"
    end_date = "2023-12-31"
    ohlc_df = load_data(start_date, end_date)

    ohlc_df['Velocity'] = ohlc_df['Low'].diff()
    ohlc_df['Acceleration'] = ohlc_df['Velocity'].diff()
    ohlc_df.dropna(inplace=True)
    momentum_df = ohlc_df[['Velocity', 'Acceleration']]
    momentum_df.reset_index(inplace=True)

    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()

    cursor.execute("DROP TABLE IF EXISTS Low_momentum")
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS Low_momentum (
        Datetime TEXT PRIMARY KEY,
        Velocity REAL,
        Acceleration REAL
    )
    ''')
    conn.commit()

    momentum_df.to_sql('Low_momentum', conn, if_exists='replace', index=False)
    cursor.execute('''
    CREATE INDEX IF NOT EXISTS idx_Low_momentum_datetime ON Low_momentum (Datetime)
    ''')
    conn.commit()
    conn.close()

# Create index on SPX_1m table
create_index()

# Create the Close_momentum, High_momentum, and Low_momentum tables
create_close_momentum()
create_high_momentum()
create_low_momentum()

now = datetime.now()
formatted_now = now.strftime("%Y-%m-%d %H:%M:%S")
print("Current date and time:", formatted_now)
    
    