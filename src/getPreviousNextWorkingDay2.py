import sqlite3
from datetime import datetime

# Function to find the previous working day based on actual data in the database
def get_previous_working_day(date_str, table_name, db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Query to get the latest date before the given date
    query = f"""
    SELECT MAX(Datetime) FROM {table_name} 
    WHERE Datetime < '{date_str} 00:00:00'
    """
    cursor.execute(query)
    result = cursor.fetchone()
    conn.close()
    
    previous_working_day = result[0]
    if previous_working_day:
        return previous_working_day.split(" ")[0]  # Return only the date part
    else:
        return None


# Function to find the next working day based on actual data in the database
def get_next_working_day(date_str, table_name, db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Query to get the earliest date after the given date
    query = f"""
    SELECT MIN(Datetime) FROM {table_name} 
    WHERE Datetime > '{date_str} 23:59:59'
    """
    cursor.execute(query)
    result = cursor.fetchone()
    conn.close()
    
    next_working_day = result[0]
    if next_working_day:
        return next_working_day.split(" ")[0]  # Return only the date part
    else:
        return None

# Function to get data for a specific date from the database
def get_data_for_date(date_str, table_name, db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    query = f"SELECT * FROM {table_name} WHERE Datetime LIKE '{date_str}%'"
    cursor.execute(query)
    rows = cursor.fetchall()
    conn.close()
    return rows

# Given date
date_str = "2024-05-20"
table_name = "SPY_1m"
db_path="stockdata/stock_data.db"
previous_working_day = get_previous_working_day(date_str, table_name, db_path)
if previous_working_day:
    print(f"Previous working day for {date_str} is {previous_working_day}")

    # Fetch data for the previous working day
    # data = get_data_for_date(previous_working_day, table_name, db_path)
    # for row in data:
    #     print(row)
    # print("Total length:", len(data))
else:
    print(f"No previous working day found for {date_str}")

next_working_day = get_next_working_day(date_str, table_name, db_path)
if previous_working_day:
    print(f"Next working day for {date_str} is {next_working_day}")

    # Fetch data for the next working day
    # data = get_data_for_date(next_working_day, table_name, db_path)
    # for row in data:
    #     print(row)
    # print("Total length:", len(data))
else:
    print(f"No previous working day found for {date_str}")

