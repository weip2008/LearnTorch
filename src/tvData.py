from tvDatafeed import TvDatafeed, Interval

username = 'YourTradingViewUsername'
password = 'YourTradingViewPassword'

# tv = TvDatafeed(username, password)
# without user name and password
tv = TvDatafeed()

# (symbol: str, exchange: str = 'NSE', interval: Interval = Interval.in_daily, n_bars: int = 10, fut_contract: int | None = None, extended_session: bool = False) -> DataFrame)



# Test with a known working symbol
test_data = tv.get_hist(symbol='NDX', exchange='NASDAQ', interval=Interval.in_1_minute, n_bars=5)
print(test_data)

# print(tv) # <tvDatafeed.main.TvDatafeed object at 0x7fc1d6307c10>

#print(seis)

# index
# nifty_index_data = tv.get_hist(symbol='BA',exchange='NSE',interval=Interval.in_1_hour,n_bars=1000)

# futures continuous contract
nq_data = tv.get_hist(symbol='NQ', exchange='CME', interval=Interval.in_1_minute, n_bars=5, fut_contract=1)
print(nq_data)
# crudeoil
# crudeoil_data = tv.get_hist(symbol='CRUDEOIL',exchange='MCX',interval=Interval.in_1_hour,n_bars=5000,fut_contract=1)

# downloading data for extended market hours
# extended_price_data = tv.get_hist(symbol="EICHERMOT",exchange="NSE",interval=Interval.in_1_hour,n_bars=500, extended_session=False)
# print(extended_price_data)
