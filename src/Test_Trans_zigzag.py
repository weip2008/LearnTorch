import pandas as pd
import numpy as np
import yfinance as yf

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

    abs_reversal = absolute_reversal if absolute_reversal != 0 else tick_reversal * df['TickSize']

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
        prev_state = df.at[i-1, 'state']
        prev_maxH = df.at[i-1, 'max_priceH']
        prev_minL = df.at[i-1, 'min_priceL']

        if prev_state == 'init':
            df.at[i, 'state'] = 'undefined'
        elif prev_state == 'undefined':
            if df.at[i, 'High'] >= prev_maxH:
                df.at[i, 'state'] = 'uptrend'
                df.at[i, 'new_max'] = True
                df.at[i, 'new_min'] = False
            elif df.at[i, 'Low'] <= prev_minL:
                df.at[i, 'state'] = 'downtrend'
                df.at[i, 'new_max'] = False
                df.at[i, 'new_min'] = True
            else:
                df.at[i, 'new_max'] = df.at[i, 'new_min'] = False
        elif prev_state == 'uptrend':
            if df.at[i, 'Low'] <= prev_maxH - prev_maxH * hl_pivot - abs_reversal:
                df.at[i, 'state'] = 'downtrend'
                df.at[i, 'new_max'] = False
                df.at[i, 'new_min'] = True
            else:
                df.at[i, 'new_max'] = df.at[i, 'High'] >= prev_maxH
        elif prev_state == 'downtrend':
            if df.at[i, 'High'] >= prev_minL + prev_minL * hl_pivot + abs_reversal:
                df.at[i, 'state'] = 'uptrend'
                df.at[i, 'new_max'] = True
                df.at[i, 'new_min'] = False
            else:
                df.at[i, 'new_min'] = df.at[i, 'Low'] <= prev_minL

        if df.at[i, 'new_max']:
            df.at[i, 'max_priceH'] = df.at[i, 'High']
        else:
            df.at[i, 'max_priceH'] = prev_maxH

        if df.at[i, 'new_min']:
            df.at[i, 'min_priceL'] = df.at[i, 'Low']
        else:
            df.at[i, 'min_priceL'] = prev_minL

    bar_number = np.arange(len(df))
    bar_count = len(df)

    def get_last_point(df, high_point, low_point, offset, new_state, new_max, new_min):
        if high_point and offset > 1:
            for i in range(1, offset):
                if new_max.iloc[-i] or (i == offset - 1 and df['High'].iloc[-i] == df['High'].iloc[-offset]):
                    return np.nan
                else:
                    return df['High'].iloc[-offset]
        elif low_point and offset > 1:
            for i in range(1, offset):
                if new_min.iloc[-i] or (i == offset - 1 and df['Low'].iloc[-i] == df['Low'].iloc[-offset]):
                    return np.nan
                else:
                    return df['Low'].iloc[-offset]
        else:
            return np.nan

    df['new_state'] = df['state'].shift(1) != df['state']
    offset = bar_count - bar_number

    df['high_point'] = (df['state'] == 'uptrend') & (df['High'] == df['max_priceH'])
    df['low_point'] = (df['state'] == 'downtrend') & (df['Low'] == df['min_priceL'])

    df['lastH'] = df.apply(lambda x: get_last_point(df, x['high_point'], x['low_point'], offset, df['new_state'], df['new_max'], df['new_min']), axis=1)
    df['lastL'] = df.apply(lambda x: get_last_point(df, x['high_point'], x['low_point'], offset, df['new_state'], df['new_max'], df['new_min']), axis=1)

    df['ZZ'] = np.nan
    df['ZZ'] = np.where(bar_number == 0, 
                        df.apply(lambda x: df['Low'].iloc[::-1][df['state'].iloc[::-1] == 'uptrend'].values[0] if 'uptrend' in df['state'].values[::-1] else df['High'].iloc[::-1][df['state'].iloc[::-1] == 'downtrend'].values[0] if 'downtrend' in df['state'].values[::-1] else np.nan, axis=1), 
                        df['ZZ'])
    
    df['ZZ'] = np.where(bar_number == bar_count-1, 
                        np.where((df['high_point']) | ((df['state'] == 'downtrend') & (df['Low'] > df['min_priceL'])), df['High'], 
                                 np.where((df['low_point']) | ((df['state'] == 'uptrend') & (df['High'] < df['max_priceH'])), df['Low'], np.nan)), 
                        df['ZZ'])
    
    df['ZZ'] = np.where(~df['lastH'].isna(), df['lastH'], 
                        np.where(~df['lastL'].isna(), df['lastL'], df['ZZ']))

    return df

# Example usage:
# df = pd.DataFrame({'High': ..., 'Low': ..., 'Close': ...})
# df['TickSize'] = 0.01  # example tick size
# zz_df = calculate_zigzag(df)
# print(zz_df[['High', 'Low', 'Close', 'ZZ']])


# Example usage:
ticker_symbols = [ "MES=F" ]
# Data interval
t_interval="1m"


# Fetch the historical data from the first day it started trading
stock_data = yf.Ticker("MES=F")
#stock_hist = stock_data.history(period="max", auto_adjust=False)
df = stock_data.history(period="max", interval=t_interval, auto_adjust=False)

#df = pd.DataFrame({'high': ..., 'low': ..., 'close': ...})
df['tick_size'] = 0.01  # example tick size
zz_df = calculate_zigzag(df)
print(zz_df[['High', 'Low', 'Close', 'ZZ']])

