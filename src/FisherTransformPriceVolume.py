import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import yfinance as yf
import vectorbt as vbt
from sklearn.model_selection import ParameterGrid

# Function to calculate Price Volume Trend (PVT)
def calculate_pvt(df):
    pvt = [0]  # Initial PVT value
    for i in range(1, len(df)):
        pvt_value = pvt[-1] + ((df['Close'][i] - df['Close'][i-1]) / df['Close'][i-1]) * df['Volume'][i]
        pvt.append(pvt_value)
    return pd.Series(pvt, index=df.index)

# Function to calculate Fisher Transform and its Signal line
def calculate_fisher_transform(df, period=10):
    high_rolling = df['High'].rolling(window=period).max()
    low_rolling = df['Low'].rolling(window=period).min()

    X = 2 * ((df['Close'] - low_rolling) / (high_rolling - low_rolling) - 0.5)
    fisher = 0.5 * np.log((1 + X) / (1 - X))
    fisher_signal = fisher.ewm(span=9).mean()

    return fisher, fisher_signal

# Define the stock symbol and time period
symbol = 'TSLA'
start_date = '2019-01-01'
end_date = '2025-01-01'

# Download the data
df = yf.download(symbol, start=start_date, end=end_date)
#df.columns = ['Close', 'High', 'Low', 'Open', 'Volume']
df = df[['Close', 'High', 'Low', 'Open', 'Volume']]
df.ffill(inplace=True)

# Define the range for the optimization
shift_values = range(1, 31)  # PVT shift from 1 to 30
fisher_period_values = range(5, 31)  # Fisher period from 5 to 30

# Generate parameter grid
param_grid = {
    'shift': shift_values,
    'fisher_period': fisher_period_values
}
grid = ParameterGrid(param_grid)

# Store results for all combinations
results = []

# Optimize PVT shift and Fisher period
for params in grid:
    shift_value = params['shift']
    fisher_period = params['fisher_period']

    # Calculate Price Volume Trend (PVT)
    df['PVT'] = calculate_pvt(df)

    # Calculate Fisher Transform and Signal line
    df['Fisher'], df['Fisher_Signal'] = calculate_fisher_transform(df, period=fisher_period)

    # Define Entry and Exit signals based on PVT and Fisher Transform
    df['Entry'] = (df['PVT'] > df['PVT'].shift(shift_value)) & (df['Fisher'] > df['Fisher_Signal'])
    df['Exit'] = (df['PVT'] < df['PVT'].shift(shift_value)) & (df['Fisher'] < df['Fisher_Signal'])

    # Filter data for the test period (2020-2025)
    df_test = df[(df.index.year >= 2020) & (df.index.year <= 2025)]

    # Backtest using vectorbt
    portfolio = vbt.Portfolio.from_signals(
        close=df_test['Close'],
        entries=df_test['Entry'],
        exits=df_test['Exit'],
        init_cash=100_000,
        fees=0.001
    )

    # Store the result
    results.append({
        'shift': shift_value,
        'fisher_period': fisher_period,
        'performance': portfolio.stats()['Total Return [%]']
    })

# Convert results to DataFrame
results_df = pd.DataFrame(results)

# Pivot table for heatmap
heatmap_data = results_df.pivot(index='fisher_period', columns='shift', values='performance')

# Plot heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(heatmap_data, annot=False, fmt=".1f", cmap="coolwarm", cbar_kws={'label': 'Total Return [%]'})
plt.title("Heatmap of Total Return by Shift and Fisher Period")
plt.xlabel("Shift")
plt.ylabel("Fisher Period")
plt.show()

# Find the best parameters based on Total Return
best_params = results_df.loc[results_df['performance'].idxmax()]

print("Best parameters:")
print(f"Shift: {best_params['shift']}")
print(f"Fisher Period: {best_params['fisher_period']}")
print(f"Total Return: {best_params['performance']}")

# Calculate Price Volume Trend (PVT)
df['PVT'] = calculate_pvt(df)

# Calculate Fisher Transform and Signal line
df['Fisher'], df['Fisher_Signal'] = calculate_fisher_transform(df, period=30)

# Define Entry and Exit signals based on PVT and Fisher Transform
df['Entry'] = (df['PVT'] > df['PVT'].shift(26)) & (df['Fisher'] > df['Fisher_Signal'])
df['Exit'] = (df['PVT'] < df['PVT'].shift(26)) & (df['Fisher'] < df['Fisher_Signal'])

# Filter data for the test period (2020-2025)
df = df[(df.index.year >= 2020) & (df.index.year <= 2025)]

# Backtest using vectorbt
portfolio = vbt.Portfolio.from_signals(
    close=df['Close'],
    entries=df['Entry'],
    exits=df['Exit'],
    init_cash=100_000,
    fees=0.001
)

# Display performance metrics
print(portfolio.stats())

# Plot equity curve
portfolio.plot().show()
