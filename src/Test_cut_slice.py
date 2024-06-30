import pandas as pd

def cut_slice(ohlc_df, start_index, end_index):
    # Ensure the start_index and end_index are in the DataFrame index
    if start_index not in ohlc_df.index or end_index not in ohlc_df.index:
        # If either index is not found, return None
        return None
    
    # Get the positional indices of the timestamps
    start_pos = ohlc_df.index.get_loc(start_index)
    end_pos = ohlc_df.index.get_loc(end_index)
    
    # Ensure start_pos is less than or equal to end_pos
    if start_pos > end_pos:
        return None
    
    # Create a copy of the section of the original DataFrame
    # Start from start_pos up to and including end_pos
    section_df = ohlc_df.iloc[start_pos:end_pos + 1].copy()
    section_df.drop(['Open', 'High', 'Low', 'Volume'], axis=1, inplace=True)
    
    return section_df

if __name__ == "__main__":
    # Provided data
    data = {
        'Open': [4779.636, 4781.136, 4782.836, 4781.648, 4780.145, 4781.233, 4781.639, 4783.751, 4783.851],
        'High': [4785.539, 4784.099, 4784.242, 4782.151, 4781.699, 4782.699, 4783.899, 4784.354, 4784.199],
        'Low': [4779.636, 4780.133, 4781.342, 4780.142, 4779.836, 4781.133, 4781.639, 4782.842, 4783.351],
        'Close': [4781.251, 4782.736, 4781.348, 4780.242, 4781.633, 4781.854, 4783.899, 4783.633, 4783.845],
        'Volume': [0, 0, 0, 0, 0, 0, 0, 0, 0]
    }
    index = [
        '2022-01-02 18:00:00', '2022-01-02 18:01:00', '2022-01-02 18:02:00', 
        '2022-01-02 18:03:00', '2022-01-02 18:04:00', '2022-01-02 18:05:00', 
        '2022-01-02 18:06:00', '2022-01-02 18:07:00', '2022-01-02 18:08:00'
    ]
    index = pd.to_datetime(index)

    ohlc_df = pd.DataFrame(data, index=index)

    # Example usage
    start_index = pd.Timestamp('2022-01-02 18:03:00')
    end_index = pd.Timestamp('2022-01-02 18:07:00')

    result_df = cut_slice(ohlc_df, start_index, end_index)
    print(result_df)
