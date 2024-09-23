import sqlite3
import os
import pandas as pd

import zigzagplus1 as zz
from gru import Logger
from config import Config

class DataSource:
    config = Config("gru/src/config.ini")
    log = Logger("gru/log/gru.log")
    conn = None

    def __init__(self):
        sqliteDB = DataSource.config.sqlite_db
        data_dir = DataSource.config.data_dir

        # Connect to the SQLite database
        db_file = os.path.join(data_dir, sqliteDB)
        # Connect to the SQLite database
        DataSource.conn = sqlite3.connect(db_file)
        cursor = DataSource.conn.cursor()

    def queryDB(self, query_start, query_end):
        table_name = DataSource.config.table_name
        # Query the data between May 6th, 2024, and May 12th, 2024
        query_range = f'''
        SELECT * FROM {table_name}
        WHERE Datetime BETWEEN ? AND ?
        '''

        # Save the query result into a DataFrame object named query_result_df
        query_result_df = pd.read_sql_query(query_range, DataSource.conn, params=(query_start, query_end))
    
        self.df = query_result_df
        self.df['Datetime'] = pd.to_datetime(self.df['Datetime'])
        self.df.set_index('Datetime', inplace=True)

        DataSource.log.debug(f"Results dataframe length: {len(self.df)}")  
        DataSource.log.debug(f"Data read from table: {table_name}")
        DataSource.log.debug(self.df.head(10))
        DataSource.log.debug(self.df.tail(10))   


    def getDataFrameFromDB(self):
        return self.df

    def plotDataFrame(self):
        self.zigzag = zz.calculate_zigzag(self.df, float(DataSource.config.deviation))
        
        # Plot ZigZag
        zz.plot_zigzag(self.df, self.zigzag)

    def plotPaterns(self):
        filtered_zigzag_df = self.df.loc[self.zigzag.index]
        DataSource.log.debug(f"filtered_zigzag_df list length:{len(filtered_zigzag_df)}\n{filtered_zigzag_df}")

        # Detect patterns
        # df[df['Close'].isin(zigzag)] creates a new DataFrame 
        # that contains only the rows from df 
        # where the 'Close' value is in the zigzag list.
        # patterns = detect_patterns(df[df['Close'].isin(zigzag)])
        patterns = zz.detect_patterns(filtered_zigzag_df)

        patterns_df = zz.convert_list_to_df(patterns)
        zz.plot_patterns(self.df, patterns_df)

if __name__ == "__main__":
    # plot training zigzag
    train_ds = DataSource()
    query_start, query_end= DataSource.config.training_start_date, DataSource.config.training_end_date
    train_ds.queryDB(query_start, query_end)
    train_ds.plotDataFrame()
    train_ds.plotPaterns()

    # plot testing zigzag
    test_ds = DataSource()
    query_start, query_end= DataSource.config.testing_start_date,DataSource.config.testing_end_date
    test_ds.queryDB(query_start, query_end)
    test_ds.plotDataFrame()
    test_ds.plotPaterns()
 
    DataSource.conn.close()
