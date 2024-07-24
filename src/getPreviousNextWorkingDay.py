import sqlite3
from datetime import datetime, timedelta

# Define a list of holidays (as strings in the format YYYY-MM-DD)
holidays = [
    "2024-01-01",  # New Year's Day
    "2024-01-15",  # Martin Luther King Jr. Day
    "2024-02-19",  # Presidents' Day
    "2024-04-01",  # Good Friday
    "2024-05-27",  # Memorial Day
    "2024-06-19",  # Juneteenth
    "2024-07-04",  # Independence Day
    "2024-09-02",  # Labor Day
    "2024-11-28",  # Thanksgiving Day
    "2024-12-25",  # Christmas Day
]

# Define a function to find the previous working day
def get_previous_working_day(date_str):
    given_date = datetime.strptime(date_str, "%Y-%m-%d")
    previous_working_day = given_date
    
    while True:
        previous_working_day -= timedelta(days=1)
        
        # Check if the previous day is a weekend
        if previous_working_day.weekday() >= 5:
            continue
        
        # Check if the previous day is a holiday
        if previous_working_day.strftime("%Y-%m-%d") in holidays:
            continue
        
        break
    
    return previous_working_day.strftime("%Y-%m-%d")


# Define a function to find the next working day
def get_next_working_day(date_str):
    given_date = datetime.strptime(date_str, "%Y-%m-%d")
    next_working_day = given_date
    
    while True:
        next_working_day += timedelta(days=1)
        
        # Check if the next day is a weekend
        if next_working_day.weekday() >= 5:
            continue
        
        # Check if the next day is a holiday
        if next_working_day.strftime("%Y-%m-%d") in holidays:
            continue
        
        break
    
    return next_working_day.strftime("%Y-%m-%d")

# Function to get data for a specific date from the database
def get_data_for_date(date_str, db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    query = f"SELECT * FROM SPY_1m WHERE Datetime LIKE '{date_str}%'"
    cursor.execute(query)
    rows = cursor.fetchall()
    conn.close()
    return rows

# Given database
db_path="stockdata/stock_data.db"
# Given date
date_str = "2024-07-19"
previous_working_day = get_previous_working_day(date_str)
print(f"Previous working day for {date_str} is {previous_working_day}")

next_working_day = get_next_working_day(date_str)
print(f"Previous working day for {date_str} is {next_working_day}")


# Fetch data for the previous working day
# data = get_data_for_date(previous_working_day)
# for row in data:
#     print(row)
