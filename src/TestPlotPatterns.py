import pandas as pd
import matplotlib.pyplot as plt

# Example detect_patterns function
def detect_patterns(df):
    # Sample implementation for illustration
    # The actual function should return a list of dictionaries with 'Datetime', 'Point', and 'Label' keys
    return [
        {'Datetime': '2024-06-07 16:25:00-04:00', 'Point': 5352.25, 'Label': 'LL'},
        {'Datetime': '2024-06-07 16:26:00-04:00', 'Point': 5352.25, 'Label': 'LH'},
        {'Datetime': '2024-06-07 16:32:00-04:00', 'Point': 5351.25, 'Label': 'LH'},
        {'Datetime': '2024-06-07 16:33:00-04:00', 'Point': 5351.25, 'Label': 'LL'},
        {'Datetime': '2024-06-07 16:35:00-04:00', 'Point': 5352.5, 'Label': 'HL'},
        {'Datetime': '2024-06-07 16:36:00-04:00', 'Point': 5352.75, 'Label': 'HH'},
        {'Datetime': '2024-06-07 16:38:00-04:00', 'Point': 5352.25, 'Label': 'LH'},
        {'Datetime': '2024-06-07 16:39:00-04:00', 'Point': 5352.25, 'Label': 'LL'},
        {'Datetime': '2024-06-07 16:40:00-04:00', 'Point': 5351.5, 'Label': 'LH'},
        {'Datetime': '2024-06-07 16:41:00-04:00', 'Point': 5351.5, 'Label': 'LL'},
        {'Datetime': '2024-06-07 16:43:00-04:00', 'Point': 5351.5, 'Label': 'LH'},
        {'Datetime': '2024-06-07 16:46:00-04:00', 'Point': 5352.25, 'Label': 'HL'}
    ]

# Simulate data for df and zigzag
data = {
    'Datetime': pd.date_range(start='2024-06-07 16:00:00', periods=12, freq='T'),
    'Close': [5352.25, 5352.25, 5351.25, 5351.25, 5352.5, 5352.75, 5352.25, 5352.25, 5351.5, 5351.5, 5351.5, 5352.25]
}
df = pd.DataFrame(data)
zigzag = df['Close'].tolist()  # Sample zigzag list for illustration

# Detect patterns
patterns = detect_patterns(df[df['Close'].isin(zigzag)])
print(patterns)  # Print to check structure

# Convert list to DataFrame if not empty
if patterns:
    patterns_df = pd.DataFrame(patterns)
    if 'Datetime' in patterns_df.columns:
        patterns_df['Datetime'] = pd.to_datetime(patterns_df['Datetime'])
        patterns_df.set_index('Datetime', inplace=True)
        
        # Plotting to visualize the result
        fig, ax = plt.subplots(figsize=(14, 7))

        ax.plot(df['Datetime'], df['Close'], label='Close Price')

        # Use .loc for indexing
        ax.scatter(patterns_df.loc[patterns_df['Label'] == 'HH'].index, patterns_df.loc[patterns_df['Label'] == 'HH', 'Point'], color='green', label='HH', marker='^', alpha=1)
        ax.scatter(patterns_df.loc[patterns_df['Label'] == 'LL'].index, patterns_df.loc[patterns_df['Label'] == 'LL', 'Point'], color='red', label='LL', marker='v', alpha=1)
        ax.scatter(patterns_df.loc[(patterns_df['Label'] == 'LH') | (patterns_df['Label'] == 'HL')].index, patterns_df.loc[(patterns_df['Label'] == 'LH') | (patterns_df['Label'] == 'HL'), 'Point'], color='black', label='LH/HL', marker='o', alpha=1)

        ax.set_title('Close Price with Detected Patterns')
        ax.set_xlabel('Datetime')
        ax.set_ylabel('Close Price')
        ax.legend()

        plt.show()
    else:
        print("Error: 'Datetime' column not found in patterns_df")
else:
    print("No patterns detected")
