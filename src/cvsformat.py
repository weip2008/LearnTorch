import pandas as pd

# Assuming input_data and output_data are already defined as described earlier

# Reshape the input data to (num_groups, 5000) for CSV format
input_data_reshaped = input_data.view(num_groups, -1)

# Create a DataFrame with input and output data
df = pd.DataFrame(input_data_reshaped.numpy())
df['output1'] = output_data[:, 0].numpy()
df['output2'] = output_data[:, 1].numpy()

# Save the DataFrame to a CSV file
df.to_csv('dataset.csv', index=False)
