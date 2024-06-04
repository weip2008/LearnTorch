import pandas as pd

# Define the ticker name
ticker_name = "SPY"

# List the CSV files
csv_files = [
    f"stockdata/{ticker_name}_2024-04-11_2024-04-15_1m.csv",
    f"stockdata/{ticker_name}_2024-04-15_2024-04-21_1m.csv",
    f"stockdata/{ticker_name}_2024-04-22_2024-04-28_1m.csv",
    f"stockdata/{ticker_name}_2024-04-29_2024-05-05_1m.csv",
    f"stockdata/{ticker_name}_2024-05-06_2024-05-12_1m.csv",
    f"stockdata/{ticker_name}_2024-05-13_2024-05-19_1m.csv",
    f"stockdata/{ticker_name}_2024-05-20_2024-05-26_1m.csv",
    f"stockdata/{ticker_name}_2024-05-27_2024-06-01_1m.csv"
]

# Read each CSV file into a stockdataFrame and concatenate
dfs = []
for file in csv_files:
    df = pd.read_csv(file)
    dfs.append(df)

# Concatenate all stockdataFrames into one
concatenated_df = pd.concat(dfs, ignore_index=True)

# Save the concatenated stockdataFrame to a new CSV file
output_file = f"stockdata/{ticker_name}_2024-04-11_2024-06-01_1m.csv"
concatenated_df.to_csv(output_file, index=False)

print(f"Concatenated stockdata saved to {output_file}")