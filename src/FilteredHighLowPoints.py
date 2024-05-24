import pandas as pd
import matplotlib.pyplot as plt

def filter_points(selected_low_points, selected_high_points):
    # Initialize filtered DataFrame objects
    filtered_low_points = pd.DataFrame(columns=selected_low_points.columns)
    filtered_high_points = pd.DataFrame(columns=selected_high_points.columns)

    # Initialize indexes
    low_points_index = 0
    high_points_index = 0

    while low_points_index < len(selected_low_points)-1 and \
          high_points_index < len(selected_high_points)-1:
        #======================================================================
        # Find next low point starting from the current high point index
        while (low_points_index < len(selected_low_points) - 1 and
               #selected_low_points.index[low_points_index] < selected_high_points.index[high_points_index] and
               selected_low_points.iloc[low_points_index]['Price'] >= \
                   selected_low_points.iloc[low_points_index + 1]['Price']):
            low_points_index += 1
        
        # Append data to filtered_low_points
        filtered_low_points = pd.concat([filtered_low_points, selected_low_points.iloc[[low_points_index]]])
        #print("filtered_low_points:", filtered_low_points)

        # Get the datetime from the low_points_index
        low_point_datetime = selected_low_points.index[low_points_index]
        #print("low_point_datetime:\n",low_point_datetime)

        #======================================================================
        # Find the starting index in selected_high_points
        while high_points_index < len(selected_high_points) - 1 and \
            selected_high_points.index[high_points_index] < low_point_datetime:
            high_points_index += 1
        #print("high_points_index starts at:\t",high_points_index)
        
        #======================================================================
        # Find next high point starting from the current low point index
        while (high_points_index < len(selected_high_points) - 1 and
               selected_high_points.iloc[high_points_index]['Price'] <= \
                   selected_high_points.iloc[high_points_index + 1]['Price']):
            high_points_index += 1
        
        # Append data to filtered_high_points
        filtered_high_points = pd.concat([filtered_high_points, selected_high_points.iloc[[high_points_index]]])
        #print("filtered_high_points:",filtered_high_points)

        # Get the datetime from the high_points_index
        high_point_datetime = selected_high_points.index[high_points_index]
        #print("high_point_datetime:\n",high_point_datetime)

        #======================================================================
        # Find the starting index in selected_low_points
        while low_points_index < len(selected_low_points) - 1 and \
            selected_low_points.index[low_points_index] < high_point_datetime:
            low_points_index += 1
        #print("low_points_index starts at:\t", low_points_index)    

    return filtered_low_points, filtered_high_points


def find_point_index_int(filtered_point, selected_points):
    """
    Find the index and location of a point in the original selected_points DataFrame.
    
    Parameters:
        filtered_point (Series): A single row Series representing the filtered point.
        selected_points (DataFrame): The original DataFrame containing all selected points.
        
    Returns:
        tuple: A tuple containing the index and location of the filtered point in the original selected_points DataFrame.
    """
    # Get the index of the filtered_point
    filtered_index = filtered_point.name
    
    # Compare the index of each row in selected_points with the index of the filtered_point
    for idx in selected_points.index:
        if idx == filtered_index:
            # Get the integer location of the row in the DataFrame
            location = selected_points.index.get_loc(idx)
            return idx, location

    # If the point is not found, return None
    return None, None

if __name__ == "__main__":
   # Define the expanded test data for selected_low_points
    selected_low_points_data = {
        'Datetime': ['2024-04-09 10:11:00-04:00', '2024-04-09 10:54:00-04:00', '2024-04-09 12:55:00-04:00',
                     '2024-04-09 14:02:00-04:00', '2024-04-09 15:10:00-04:00', '2024-04-10 09:48:00-04:00',
                     '2024-04-10 11:18:00-04:00', '2024-04-10 11:44:00-04:00', '2024-04-10 12:14:00-04:00',
                     '2024-04-10 13:10:00-04:00', '2024-04-10 14:25:00-04:00', '2024-04-10 14:33:00-04:00',
                     '2024-04-10 15:05:00-04:00', '2024-04-11 09:47:00-04:00', '2024-04-11 10:16:00-04:00',
                     '2024-04-11 13:40:00-04:00', '2024-04-11 15:15:00-04:00', '2024-04-12 10:16:00-04:00',
                     '2024-04-12 10:52:00-04:00', '2024-04-12 12:48:00-04:00'],
        'Price': [518.78, 514.52, 516.72, 516.42, 515.86, 512.91, 513.62, 513.19, 513.05, 512.21,
                  513.25, 513.22, 512.56, 514.38, 512.34, 517.18, 518.66, 513.83, 512.30, 510.64]
    }

    selected_high_points_data = {
        'Datetime': ['2024-04-09 11:18:00-04:00', '2024-04-09 12:13:00-04:00', '2024-04-09 12:35:00-04:00',
                     '2024-04-09 13:22:00-04:00', '2024-04-09 14:21:00-04:00', '2024-04-09 14:53:00-04:00',
                     '2024-04-09 15:32:00-04:00', '2024-04-09 15:58:00-04:00', '2024-04-10 10:20:00-04:00',
                     '2024-04-10 10:39:00-04:00', '2024-04-10 11:29:00-04:00', '2024-04-10 12:04:00-04:00',
                     '2024-04-10 12:37:00-04:00', '2024-04-10 13:25:00-04:00', '2024-04-10 14:14:00-04:00',
                     '2024-04-10 15:25:00-04:00', '2024-04-11 09:34:00-04:00', '2024-04-11 11:03:00-04:00',
                     '2024-04-11 11:22:00-04:00', '2024-04-11 14:08:00-04:00'],
        'Price': [516.41, 517.50, 517.69, 517.52, 517.15, 516.67, 518.70, 518.91, 514.84, 515.93,
                  514.36, 513.63, 513.67, 514.90, 514.34, 514.82, 515.24, 513.73, 514.03, 519.04]
    }

     # Create DataFrame objects for selected_low_points and selected_high_points
    selected_low_points = pd.DataFrame(selected_low_points_data)
    selected_high_points = pd.DataFrame(selected_high_points_data)

    # Convert 'Datetime' column to datetime format and set as index
    selected_low_points['Datetime'] = pd.to_datetime(selected_low_points['Datetime'])
    selected_low_points.set_index('Datetime', inplace=True)
    selected_high_points['Datetime'] = pd.to_datetime(selected_high_points['Datetime'])
    selected_high_points.set_index('Datetime', inplace=True)

    # Output filtered_low_points and filtered_high_points
    print("\nSelected Low Points:")
    print(selected_low_points)
    print("\nFiltered High Points:")
    print(selected_high_points)
    
    # Call the filtering function
    filtered_low_points, filtered_high_points = filter_points(selected_low_points, selected_high_points)

    # Output filtered_low_points and filtered_high_points
    print("\nFiltered Low Points:")
    print(filtered_low_points)
    print("\nFiltered High Points:")
    print(filtered_high_points)

    # Example usage of find_point_index_int
    print("\nIndex and location of filtered points in selected_low_points:")
    for i, filtered_point in filtered_low_points.iterrows():
        original_index, location = find_point_index_int(filtered_point, selected_low_points)
        if original_index is not None:
            print(f"Filtered point at index {i} corresponds to original index {original_index} \
                    and location No. {location} in selected_low_points.")
        else:
            print(f"Filtered point at index {i} not found in selected_low_points.")
        
        
    combined_filtered_points = pd.concat([filtered_low_points, filtered_high_points])
    combined_filtered_points_sorted = combined_filtered_points.sort_index()
    
    # Plot selected_low_points, selected_high_points, filtered_low_points, and filtered_high_points
    plt.figure(figsize=(10, 6))
    plt.plot(selected_low_points.index, selected_low_points['Price'], 'ro', label='Selected Low Points')
    plt.plot(selected_high_points.index, selected_high_points['Price'], 'go', label='Selected High Points')
    plt.xlabel('Datetime')
    plt.ylabel('Price')
    plt.title('Selected Low Points and High Points')
    plt.legend()
    
    # Plot the sorted dataframe
    combined_filtered_points_sorted['Price'].plot()

    plt.show()