import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import pandas as pd
import sqlite3

from sqlalchemy import create_engine, inspect

def tablenamesfrom(engine):

    """
    Returns a list of table names in the database
    """
    # Use inspector to get table names
    return inspect(engine).get_table_names()

    # inspector = inspect(engine)
    # table_names = inspector.get_table_names()
    # return table_names

    # print("Tables in the database:")
    # for table_name in table_names:
    #     print(table_name)

def fetch_data_from_db(db_path: str, query: str) -> pd.DataFrame:
    """
    Fetch data from an SQLite database and return it as a pandas DataFrame.

    Parameters:
    - db_path (str): Path to the SQLite database file.
    - query (str): SQL query to execute.

    Returns:
    - pd.DataFrame: DataFrame containing the query results.
    """
    try:
        # Create a connection object
        conn = sqlite3.connect(db_path)

        # Read the data into a pandas DataFrame
        df = pd.read_sql(query, conn)

        # Display the first few rows of the DataFrame
        print(df.head())

        return df

    except sqlite3.DatabaseError as e:
        print(f"Database error: {e}")
        return pd.DataFrame()
    except Exception as e:
        print(f"An error occurred: {e}")
        return pd.DataFrame()
    finally:
        # Ensure the connection is closed
        conn.close()
