from tvDatafeed import TvDatafeed, Interval

username = 'YourTradingViewUsername'
password = 'YourTradingViewPassword'

# tv = TvDatafeed(username, password)

# without user name and password
tv = TvDatafeed()

# print(tv) # <tvDatafeed.main.TvDatafeed object at 0x7fc1d6307c10>

# (symbol: str, exchange: str = 'NSE', interval: Interval = Interval.in_daily, n_bars: int = 10, fut_contract: int | None = None, extended_session: bool = False) -> DataFrame)

# Test with a known working symbol
#test_data = tv.get_hist(symbol='NDX', exchange='NASDAQ', interval=Interval.c, n_bars=5)
#print(test_data)

# futures continuous contract
# nq_data = tv.get_hist(symbol='NQ', exchange='CME', interval=Interval.in_1_minute, n_bars=5, fut_contract=1)
nq_data = tv.get_hist(symbol='NQ1!', exchange='CME_MINI', interval=Interval.in_1_minute, n_bars=100000)
print(nq_data)

# downloading data for extended market hours
# extended_price_data = tv.get_hist(symbol="EICHERMOT",exchange="NSE",interval=Interval.in_1_hour,n_bars=500, extended_session=False)
# print(extended_price_data)
