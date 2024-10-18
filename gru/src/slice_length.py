import statistics
import numpy as np

from utilities import DataSource
import zigzagplus1 as zz
from logger import Logger
from config import Config

class Trade:
    def __init__(self, open_price, open_time, trade_cost, label):
        self.open_price, self.open_time = open_price, open_time
        self.trade_cost = trade_cost
        log.debug(f"At {open_time}, LONG buy  price: {open_price:.2f} at {label} point")


    def hold_minutes(self, close_price, close_time, label):
        hold_time = 0
        profit = self.profit(close_price, label)
        if (profit > 0):
            hold_time = (close_time - self.open_time).total_seconds() / 60
            log.debug(f"At {close_time}, LONG sell price: {close_price:.2f} at {label} point, Profit: {profit:.2f}, Hold Time: {hold_time}")

        return hold_time

    def profit(self, close, label):
        profit = close - self.open_price - self.trade_cost # 低买/高卖 (做多Long)
        if label[1] == 'L':
            profit = self.open_price - close - self.trade_cost # 高卖/低买 （做空Short）
        return profit
    
    def cut_slice(self, df, close_price, close_time, label, slice_length):
        if (self.profit(close_price, label)>0):
            section_df = cut_slice(df, close_time, slice_length)  
            return section_df
        return None

class SliceLength:
    """
    this class read stock data from SQLite Database, and find peaks/troughs patterns in the data, 
    and estimate median hold time between peaks and troughs that could be used as slice length.
    """
    def __init__(self):
        query_start, query_end= DataSource.config.training_start_date, DataSource.config.training_end_date
        self.df = DataSource().queryDB(query_start,query_end).getDataFrameFromDB()
        self.gen_zigzag_patterns()
        self.estimateSliceLength()

    def estimateSliceLength(self):
        """
        Calculate Mean Hold Time, Median Hold Time for long/short stock actions, 
        get Avg mean hold time which may be used for slice length in modeling.
        """
        slice_length = int(config.slice_length)
        longtradecost = float(config.longtradecost)
        shorttradecost = float(config.shorttradecost)
        
        # Initialize variables
        at_long_position = False  # Track whether we are in a long (buy) position
        in_short_position = False  # Track whether we are in a short (sell) position
        long_holdtime_list = []
        short_holdtime_list = []
        
        # Loop through the DataFrame and process each row for both LONG and SHORT positions
        for time, row in self.patterns_df.iterrows():
            label = row['Label']
            price = row['Price']
            
            start_pos = self.df.index.get_loc(time)
            if start_pos < slice_length:
                continue

            if label[1] == 'L':
                if not at_long_position:
                    # Open a long position
                    long_trade = Trade(price, time, longtradecost, label)
                    at_long_position = True
                if in_short_position:
                    # Close the short position
                    short_hold_time = short_trade.hold_minutes(price, time, label)
                    if short_hold_time > 0: 
                        short_holdtime_list.append(short_hold_time)
                    in_short_position = False

            if label[1] == 'H':
                if not in_short_position:
                    # Open a short position
                    short_trade = Trade(price, time, shorttradecost, label)
                    in_short_position = True
                if at_long_position:
                    # Close the long position
                    long_hold_time = long_trade.hold_minutes(price, time, label)
                    if long_hold_time > 0: 
                        long_holdtime_list.append(long_hold_time)
                    at_long_position = False

        # Calculate hold times for both long and short positions
        long_hold_time = self.statistics(long_holdtime_list, "Long")
        short_hold_time = self.statistics(short_holdtime_list, "Short")

        # Calculate the average hold time
        avg_hold_time = int((long_hold_time + short_hold_time) / 2)
        log.info(f"Avg mean Hold Time: {avg_hold_time}")
        
        return avg_hold_time

    def gen_zigzag_patterns(self):
        """
        Generate zigzag patters, and store the patter data into self.patterns_df as DataFrame showing below:

                    self.patterns_df
                                    Price Label
            Datetime                           
            2023-01-03 07:12:00  3848.366    HL
            2023-01-03 08:02:00  3861.628    LH
            2023-01-03 08:09:00  3849.622    HL
            2023-01-03 09:17:00  3866.619    HH
            2023-01-03 09:31:00  3853.616    HL
            ...                       ...   ...
            2023-12-22 09:53:00  4772.937    HH
            2023-12-22 14:31:00  4737.452    HL
            2023-12-29 03:29:00  4795.067    HH
            2023-12-29 12:23:00  4750.997    HL
            2023-12-29 16:08:00  4767.817    LH

        """
        deviation = float(config.deviation)
        self.df.set_index("Datetime", inplace=True)
        zigzag = zz.calculate_zigzag(self.df, deviation)
        log.debug(f"Zigzag list length:{len(zigzag)}\n{zigzag}")

        # Filter the original DataFrame using the indices
        # df.loc[zigzag.index]:
        # This expression uses the .loc accessor to select rows from the original DataFrame df.
        # The rows selected are those whose index labels match the index labels of the zigzag DataFrame (or Series).
        # In other words, it filters df to include only the rows where the index (Date) is present in the zigzag index.
        filtered_zigzag_df = self.df.loc[zigzag.index]
        log.debug(f"filtered_zigzag_df list length:{len(filtered_zigzag_df)}\n{filtered_zigzag_df}")

        # Detect patterns
        # df[df['Close'].isin(zigzag)] creates a new DataFrame 
        # that contains only the rows from df 
        # where the 'Close' value is in the zigzag list.
        # patterns = detect_patterns(df[df['Close'].isin(zigzag)])
        patterns = zz.detect_patterns(filtered_zigzag_df)

        self.patterns_df = zz.convert_list_to_df(patterns)
        log.debug(f"Patterns dataframe length:{len(self.patterns_df)}\n{self.patterns_df}")  # Print to verify DataFrame structure

    def statistics(self, holdtime_list, type):
        # Mean hold time
        mean_hold_time = statistics.mean(holdtime_list)

        # Median hold time
        median_hold_time = statistics.median(holdtime_list)

        # Standard deviation
        std_dev_hold_time = statistics.stdev(holdtime_list)

        log.info(type)
        log.info(f"Mean Hold Time: {mean_hold_time}")
        log.info(f"Median Hold Time: {median_hold_time}")
        log.info(f"Standard Deviation: {std_dev_hold_time}")
        
        long_hold_time = int(np.ceil(mean_hold_time))
        return long_hold_time

def convert_to_day_and_time(timestamp):
    # Get the day of the week (Monday=0, Sunday=6)
    day_of_week_numeric = timestamp.weekday() + 1

    # Convert the timestamp to a datetime object (to handle timezone)
    dt = timestamp.to_pydatetime()

    # Calculate the time in float format
    time_float = dt.hour + dt.minute / 60 + dt.second / 3600

    return day_of_week_numeric, time_float
   
if __name__ == '__main__':
    log = Logger('gru/log/gru.log', logger_name='data')
    config = Config('gru/src/config.ini')

    SliceLength()
