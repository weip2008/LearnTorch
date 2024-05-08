import numpy as np

def wilders_average(data, length):
    return np.mean(data[-length:])

def tick_size():
    return 0.01  # Assuming tick size is 0.01 for this example

def convert_thinkscript_to_python(price_high, price_low, percentage_reversal, absolute_reversal, atr_length, atr_reversal, tick_reversal):
    assert percentage_reversal >= 0, f"'percentage reversal' must not be negative: {percentage_reversal}"
    assert absolute_reversal >= 0, f"'absolute reversal' must not be negative: {absolute_reversal}"
    assert atr_reversal >= 0, f"'atr reversal' must not be negative: {atr_reversal}"
    assert tick_reversal >= 0, f"'ticks' must not be negative: {tick_reversal}"
    assert any([percentage_reversal != 0, absolute_reversal != 0, atr_reversal != 0, tick_reversal != 0]), "Either 'percentage reversal' or 'absolute reversal' or 'atr reversal' or 'tick reversal' must not be zero"

    abs_reversal = absolute_reversal if absolute_reversal != 0 else tick_reversal * tick_size()

    if atr_reversal != 0:
        hl_pivot = percentage_reversal / 100 + wilders_average(np.abs(price_high - price_low), atr_length) / price_low * atr_reversal
    else:
        hl_pivot = percentage_reversal / 100

    state = {'init': 0, 'undefined': 1, 'uptrend': 2, 'downtrend': 3}
    max_price_high = None
    min_price_low = None
    new_max = None
    new_min = None
    prev_max_high = None
    prev_min_low = None

    for i in range(len(price_high)):
        if state.get(i - 1) == state.get('init'):
            max_price_high[i] = price_high[i]
            min_price_low[i] = price_low[i]
            new_max[i] = True
            new_min[i] = True
            state[i] = state.get('undefined')
        elif state.get(i - 1) == state.get('undefined'):
            if price_high[i] >= prev_max_high:
                state[i] = state.get('uptrend')
                max_price_high[i] = price_high[i]
                min_price_low[i] = prev_min_low
                new_max[i] = True
                new_min[i] = False
            elif price_low[i] <= prev_min_low:
                state[i] = state.get('downtrend')
                max_price_high[i] = prev_max_high
                min_price_low[i] = price_low[i]
                new_max[i] = False
                new_min[i] = True
            else:
                state[i] = state.get('undefined')
                max_price_high[i] = prev_max_high
                min_price_low[i] = prev_min_low
                new_max[i] = False
                new_min[i] = False
        elif state.get(i - 1) == state.get('uptrend'):
            if price_low[i] <= prev_max_high - prev_max_high * hl_pivot - abs_reversal:
                state[i] = state.get('downtrend')
                max_price_high[i] = prev_max_high
                min_price_low[i] = price_low[i]
                new_max[i] = False
                new_min[i] = True
            else:
                state[i] = state.get('uptrend')
                if price_high[i] >= prev_max_high:
                    max_price_high[i] = price_high[i]
                    new_max[i] = True
                else:
                    max_price_high[i] = prev_max_high
                    new_max[i] = False
                min_price_low[i] = prev_min_low
                new_min[i] = False
        else:
            if price_high[i] >= prev_min_low + prev_min_low * hl_pivot + abs_reversal:
                state[i] = state.get('uptrend')
                max_price_high[i] = price_high[i]
                min_price_low[i] = prev_min_low
                new_max[i] = True
                new_min[i] = False
            else:
                state[i] = state.get('downtrend')
                max_price_high[i] = prev_max_high
                new_max[i] = False
                if price_low[i] <= prev_min_low:
                    min_price_low[i] = price_low[i]
                    new_min[i] = True
                else:
                    min_price_low[i] = prev_min_low
                    new_min[i] = False

    return max_price_high, min_price_low

# Example usage
price_high = [100, 110, 105, 115, 120]
price_low = [90, 95, 100, 105, 110]
percentage_reversal = 5.0
absolute_reversal = 0.0
atr_length = 5
atr_reversal = 1.5
tick_reversal = 0

max_price_high, min_price_low = convert_thinkscript_to_python(price_high, price_low, percentage_reversal, absolute_reversal, atr_length, atr_reversal, tick_reversal)
print("Max Price High:", max_price_high)
print("Min Price Low:", min_price_low)
