import pandas as pd

def cut_slice(ohlc_df, datetime_index, window_len):
    # Convert the datetime_index to a positional index
    try:
        index = ohlc_df.index.get_loc(datetime_index)
    except KeyError:
        # If the datetime_index is not found in the DataFrame index, return None
        return None
    
    start_index = index - window_len
    # If we don't have enough long data series for this slice, ignore it
    if start_index < 0:
        return None
    
    # Adjust end index to include the last element
    end_index = index + 1
    
    # Create a copy of the section of the original DataFrame
    # Start from start_index up to but not including end_index!
    section_df = ohlc_df.iloc[start_index:end_index].copy()
    section_df.drop(['Open', 'High', 'Low', 'AdjClose', 'Volume' ], axis=1, inplace=True) 
    return section_df

if __name__ == "__main__":
    # Sample data for ohlc_df
    data = {
        'Open': [100, 101, 102, 103, 104],
        'High': [110, 111, 112, 113, 114],
        'Low': [90, 91, 92, 93, 94],
        'Close': [105, 106, 107, 108, 109],
        'AdjClose': [105, 106, 107, 108, 109],
        'Volume': [105, 106, 107, 108, 109]
    }
    index = pd.to_datetime(['2024-04-11 09:58:00', '2024-04-11 10:16:00', '2024-04-11 10:45:00', '2024-04-11 10:59:00', '2024-04-11 11:22:00'])
    ohlc_df = pd.DataFrame(data, index=index)
    print(ohlc_df)
    
    # Example usage
    datetime_index = pd.to_datetime('2024-04-11 10:59:00')
    window_len = 2
    slice_df = cut_slice(ohlc_df, datetime_index, window_len+1)
    print(slice_df)
