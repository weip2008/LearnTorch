from tvDatafeed import TvDatafeed, Interval
from datetime import datetime

username = 'YourTradingViewUsername'
password = 'YourTradingViewPassword'

# tv = TvDatafeed(username, password)

# without user name and password
tv = TvDatafeed()

# add storage directory
# import os
# if not os.path.exists('stockdata'):
#     os.makedirs('stockdata')

# futures NQ and ES as list
futures = ["NQ1!", "ES1!"]
# periods enum and short string list 
periods = [[Interval.in_1_minute, "1M"], 
            [Interval.in_5_minute, "5M"],
            [Interval.in_30_minute, "30M"],
            [Interval.in_4_hour, "4H"],
            [Interval.in_daily,"1D"]]
# print(periods[3][1])
# this design make it more flexible to add more symbol and interval.
# merge them to new list
merged = [[future] + period for future in futures for period in periods]
# print(merged)

# prepare csv file name and append list 
for sublist in merged:
    today = datetime.now().strftime("%Y-%m-%d")
    csv_file_name = f'{sublist[0]}_{today}_{sublist[2]}.csv'
    #print(csv_file_name)
    sublist.append(csv_file_name)

#print(merged)

# nq_data = tv.get_hist(symbol='NQ1!', exchange='CME_MINI', interval=Interval.in_1_minute, n_bars=100000)
# print(nq_data)
# invoke get history data from tradingview and save it to csv files
for sublist in merged:
    nq_data = tv.get_hist(symbol=sublist[0], exchange='CME_MINI', interval=sublist[1], n_bars=100000)
    nq_data.to_csv(f'stockdata/{sublist[3]}')

# nq_data.to_csv("nq_1m.csv")
