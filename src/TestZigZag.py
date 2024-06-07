import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def true_range(high, low, close):
    return np.maximum(high - low, np.maximum(abs(high - close.shift(1)), abs(low - close.shift(1))))

def wilders_average(data, period):
    return data.ewm(alpha=1/period, adjust=False).mean()

def calculate_zigzag(df, percentage_reversal=5.0, absolute_reversal=0.0, atr_length=5, atr_reversal=1.5, tick_reversal=0):
    assert percentage_reversal >= 0, "'percentage reversal' must not be negative"
    assert absolute_reversal >= 0, "'absolute reversal' must not be negative"
    assert atr_reversal >= 0, "'atr reversal' must not be negative"
    assert tick_reversal >= 0, "'ticks' must not be negative"
    assert percentage_reversal != 0 or absolute_reversal != 0 or atr_reversal != 0 or tick_reversal != 0, "Either 'percentage reversal' or 'absolute reversal' or 'atr reversal' or 'tick reversal' must not be zero"

    if 'TickSize' not in df.columns:
        df['TickSize'] = 0.0085  # Set a default value for TickSize if not present

    abs_reversal = absolute_reversal if absolute_reversal != 0 else tick_reversal * df['TickSize'].iloc[0]

    if atr_reversal != 0:
        atr = wilders_average(true_range(df['High'], df['Low'], df['Close']), atr_length)
        hl_pivot = percentage_reversal / 100 + atr / df['Close'] * atr_reversal
    else:
        hl_pivot = percentage_reversal / 100

    df['state'] = 'init'
    df['max_priceH'] = df['High']
    df['min_priceL'] = df['Low']
    df['new_max'] = True
    df['new_min'] = True

    for i in range(1, len(df)):
        prev_state = df.iloc[i-1]['state']
        prev_maxH = df.iloc[i-1]['max_priceH']
        prev_minL = df.iloc[i-1]['min_priceL']
        current_high = df.iloc[i]['High']
        current_low = df.iloc[i]['Low']

        if prev_state == 'init':
            df.at[df.index[i], 'state'] = 'undefined'
        elif prev_state == 'undefined':
            if current_high >= prev_maxH:
                df.at[df.index[i], 'state'] = 'uptrend'
                df.at[df.index[i], 'new_max'] = True
                df.at[df.index[i], 'new_min'] = False
            elif current_low <= prev_minL:
                df.at[df.index[i], 'state'] = 'downtrend'
                df.at[df.index[i], 'new_max'] = False
                df.at[df.index[i], 'new_min'] = True
            else:
                df.at[df.index[i], 'new_max'] = df.at[df.index[i], 'new_min'] = False
        elif prev_state == 'uptrend':
            if current_low <= prev_maxH - prev_maxH * hl_pivot.iloc[i] - abs_reversal:
                df.at[df.index[i], 'state'] = 'downtrend'
                df.at[df.index[i], 'new_max'] = False
                df.at[df.index[i], 'new_min'] = True
            else:
                df.at[df.index[i], 'new_max'] = current_high >= prev_maxH
        elif prev_state == 'downtrend':
            if current_high >= prev_minL + prev_minL * hl_pivot.iloc[i] + abs_reversal:
                df.at[df.index[i], 'state'] = 'uptrend'
                df.at[df.index[i], 'new_max'] = True
                df.at[df.index[i], 'new_min'] = False
            else:
                df.at[df.index[i], 'new_min'] = current_low <= prev_minL

        if df.at[df.index[i], 'new_max']:
            df.at[df.index[i], 'max_priceH'] = current_high
        else:
            df.at[df.index[i], 'max_priceH'] = prev_maxH

        if df.at[df.index[i], 'new_min']:
            df.at[df.index[i], 'min_priceL'] = current_low
        else:
            df.at[df.index[i], 'min_priceL'] = prev_minL

    bar_number = np.arange(len(df))
    bar_count = len(df)

    def get_last_point(row, df, offset):
        if row['high_point'] and offset > 1:
            for i in range(1, offset):
                if df.iloc[len(df)-i]['new_max'] or (i == offset - 1 and df.iloc[len(df)-i]['High'] == df.iloc[len(df)-offset]['High']):
                    return np.nan
                else:
                    return df.iloc[len(df)-offset]['High']
        elif row['low_point'] and offset > 1:
            for i in range(1, offset):
                if df.iloc[len(df)-i]['new_min'] or (i == offset - 1 and df.iloc[len(df)-i]['Low'] == df.iloc[len(df)-offset]['Low']):
                    return np.nan
                else:
                    return df.iloc[len(df)-offset]['Low']
        else:
            return np.nan

    df['new_state'] = df['state'].shift(1) != df['state']
    df['offset'] = bar_count - bar_number

    df['high_point'] = (df['state'] == 'uptrend') & (df['High'] == df['max_priceH'])
    df['low_point'] = (df['state'] == 'downtrend') & (df['Low'] == df['min_priceL'])

    df['lastH'] = df.apply(lambda x: get_last_point(x, df, x['offset']), axis=1)
    df['lastL'] = df.apply(lambda x: get_last_point(x, df, x['offset']), axis=1)

    df['ZZ'] = np.nan
    df['ZZ'] = np.where(bar_number == 0, 
                        df.apply(lambda x: df['Low'].iloc[::-1][df['state'].iloc[::-1] == 'uptrend'].values[0] if 'uptrend' in df['state'].values[::-1] else df['High'].iloc[::-1][df['state'].values[::-1] == 'downtrend'].values[0] if 'downtrend' in df['state'].values[::-1] else np.nan, axis=1), 
                        df['ZZ'])
    
    df['ZZ'] = np.where(bar_number == bar_count-1, 
                        np.where((df['high_point']) | ((df['state'] == 'downtrend') & (df['Low'] > df['min_priceL'])), df['High'], 
                                 np.where((df['low_point']) | ((df['state'] == 'uptrend') & (df['High'] < df['max_priceH'])), df['Low'], np.nan)), 
                        df['ZZ'])
    
    df['ZZ'] = np.where(~df['lastH'].isna(), df['lastH'], 
                        np.where(~df['lastL'].isna(), df['lastL'], df['ZZ']))

    return df

# Example usage:
# Create a sample dataframe
data = {
    'High': [5244.00, 5244.50, 5246.00, 5245.75, 5246.00],
    'Low': [5243.00, 5243.50, 5244.25, 5245.25, 5245.50],
    'Close': [5243.75, 5244.25, 5245.50, 5245.50, 5245.75]
}
df = pd.DataFrame(data, index=pd.to_datetime(['2024-05-30 23:24:00-04:00', '2024-05-30 23:25:00-04:00', '2024-05-30 23:26:00-04:00', '2024-05-30 23:27:00-04:00', '2024-05-30 23:28:00-04:00']))
df['TickSize'] = 0.01  # Set a constant tick size for example

zz_df = calculate_zigzag(df)
print(zz_df[['High', 'Low', 'Close', 'ZZ']])


# Example usage:
ticker_symbols = [ "MES=F" ]
# Data interval
t_interval="1m"

# Fetch the historical data from the first day it started trading
stock_data = yf.Ticker("MES=F")
#stock_hist = stock_data.history(period="max", auto_adjust=False)
df = stock_data.history(period="max", interval=t_interval, auto_adjust=False)
# Drop the 'Dividends' and 'Stock Splits' columns
if 'Dividends' in df.columns:
    df = df.drop(columns=['Dividends'])
if 'Stock Splits' in df.columns:
    df = df.drop(columns=['Stock Splits'])
if 'Adj Close' in df.columns:
    df = df.drop(columns=['Adj Close'])    
                           
                   
print(df)


df['TickSize'] = 0.0082 # Set a constant tick size for example

zz_df = calculate_zigzag(df)
print(zz_df[['High', 'Low', 'Close', 'ZZ']])

# Plotting the results
plt.figure(figsize=(14, 7))
plt.plot(df.index, df['Close'], label='Close', color='blue')
plt.scatter(zz_df.index, zz_df['ZZ'], label='ZigZag', color='red')
plt.title('Close Price and ZigZag Indicator')
plt.xlabel('Datetime')
plt.ylabel('Price')
plt.legend()
plt.show()