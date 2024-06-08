''' 
Explanation of code:
1. Using "max" Period: This ensures we get the maximum available historical data for better analysis.
2. Lowered Thresholds: The thresholds for the MACD histogram are set to 0.2 and -0.2 to capture more signals.
3. Relaxed RSI Conditions: The RSI conditions are adjusted to be less strict (buy when RSI < 40 and sell when RSI > 60)
    to allow more signals.
4. Data Checks: Print statements for stock_hist.head() and stock_hist.tail() are added to ensure that the historical data
    is correctly loaded and spans a sufficient period.
These changes should help generate more signals and ensure the strategy is not overly restrictive, 
leading to a more profitable and accurate trading strategy. If the results are still not satisfactory, 
further tuning of the parameters and conditions may be necessary based on the specific characteristics of the market
and the asset being traded.
 '''
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# Define the ticker symbol and interval
ticker_symbol = "MES=F"
t_interval = "1d"

# Fetch historical data
stock_data = yf.Ticker(ticker_symbol)
stock_hist = stock_data.history(period="max", interval=t_interval, auto_adjust=False)

# Drop the 'Dividends' and 'Stock Splits' columns
if 'Dividends' in stock_hist.columns:
    stock_hist = stock_hist.drop(columns=['Dividends'])
if 'Stock Splits' in stock_hist.columns:
    stock_hist = stock_hist.drop(columns=['Stock Splits'])   
            
# Check data availability
print(stock_hist.head())
print(stock_hist.tail())

# Calculate MACD and Signal Line
short_ema = stock_hist['Close'].ewm(span=12, adjust=False).mean()
long_ema = stock_hist['Close'].ewm(span=26, adjust=False).mean()
macd = short_ema - long_ema
signal = macd.ewm(span=9, adjust=False).mean()
macd_hist = macd - signal
print("MACD:\n", macd_hist)

# Calculate Relative Strength Index (RSI)
delta = stock_hist['Close'].diff(1)
gain = delta.where(delta > 0, 0)
loss = -delta.where(delta < 0, 0)
avg_gain = gain.rolling(window=14).mean()
avg_loss = loss.rolling(window=14).mean()
rs = avg_gain / avg_loss
rsi = 100 - (100 / (1 + rs))
print("RSI:\n", rsi)

# Calculate a long-term moving average for trend confirmation
long_ma = stock_hist['Close'].rolling(window=50).mean()

# Create a DataFrame to store signals
signals = pd.DataFrame(index=stock_hist.index)
signals['MACD'] = macd
signals['Signal Line'] = signal
signals['MACD Histogram'] = macd_hist
signals['RSI'] = rsi
signals['Long MA'] = long_ma
print("Combined criteria:\n")
print(signals.head())
print(signals.tail())


# Define dynamic thresholds for MACD histogram
buy_threshold = 0.05
sell_threshold = -0.05

# Generate buy/sell signals based on combined criteria
signals['Buy Signal'] = (
    (signals['MACD Histogram'] > buy_threshold) & 
    (signals['MACD Histogram'].shift(1) <= buy_threshold) &
    (signals['RSI'] < 40) 
#    (stock_hist['Close'] > signals['Long MA'])
)
signals['Sell Signal'] = (
    (signals['MACD Histogram'] < sell_threshold) & 
    (signals['MACD Histogram'].shift(1) >= sell_threshold) &
    (signals['RSI'] > 60) 
#    (stock_hist['Close'] < signals['Long MA'])
)

# Apply holding period filter (e.g., 5 minutes)
holding_period = 5
buy_signal_int = signals['Buy Signal'].astype(int)
sell_signal_int = signals['Sell Signal'].astype(int)
signals['Buy Signal'] = \
    (signals['Buy Signal']) & (~buy_signal_int.shift().rolling(window=holding_period).max().fillna(0).astype(bool))
signals['Sell Signal'] = \
    (signals['Sell Signal']) & (~sell_signal_int.shift().rolling(window=holding_period).max().fillna(0).astype(bool))

# Plotting
plt.figure(figsize=(14, 10))

# Plot MACD and Signal Line
plt.subplot(3, 1, 1)
plt.plot(signals.index, signals['MACD'], label='MACD', color='blue')
plt.plot(signals.index, signals['Signal Line'], label='Signal Line', color='red')
plt.legend(loc='upper left')
plt.title(f"MACD and Signal Line for {ticker_symbol}")

# Plot MACD Histogram
plt.subplot(3, 1, 2)
plt.bar(signals.index, signals['MACD Histogram'], color='gray', label='MACD Histogram')
plt.legend(loc='upper left')

# Plot RSI
plt.subplot(3, 1, 3)
plt.plot(signals.index, signals['RSI'], label='RSI', color='purple')
plt.axhline(40, linestyle='--', alpha=0.5, color='green')
plt.axhline(60, linestyle='--', alpha=0.5, color='red')
plt.legend(loc='upper left')

# Mark buy/sell points on the MACD plot
buy_signals = signals[signals['Buy Signal']]
sell_signals = signals[signals['Sell Signal']]
plt.subplot(3, 1, 1)
plt.scatter(buy_signals.index, signals.loc[buy_signals.index, 'MACD'], marker='^', color='green', label='Buy Signal', alpha=1)
plt.scatter(sell_signals.index, signals.loc[sell_signals.index, 'MACD'], marker='v', color='red', label='Sell Signal', alpha=1)
plt.legend(loc='upper left')

plt.tight_layout()
plt.show()

# Print buy/sell points list
buy_sell_points = pd.concat([buy_signals, sell_signals])
# Sort by index
buy_sell_points = buy_sell_points.sort_index()
buy_sell_points_list = buy_sell_points[['Buy Signal', 'Sell Signal']]
print("\nBuy/Sell Points:")
print(buy_sell_points_list)
