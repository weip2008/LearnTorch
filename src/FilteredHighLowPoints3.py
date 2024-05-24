import pandas as pd

def filter_points(selected_low_points, selected_high_points):
    # Initialize filtered DataFrame objects
    filtered_low_points = pd.DataFrame(columns=selected_low_points.columns)
    filtered_high_points = pd.DataFrame(columns=selected_high_points.columns)

    # Initialize indexes
    low_points_index = 0
    high_points_index = 0

    while low_points_index < len(selected_low_points) and high_points_index < len(selected_high_points):
        # Find next low point starting from the current high point index
        while (low_points_index < len(selected_low_points) - 1 and
               selected_low_points.index[low_points_index] < selected_high_points.index[high_points_index] and
               selected_low_points.iloc[low_points_index]['Price'] >= selected_low_points.iloc[low_points_index + 1]['Price']):
            low_points_index += 1
        filtered_low_points = pd.concat([filtered_low_points, selected_low_points.iloc[[low_points_index]]])

        # Update high_points_index based on the last filtered low point's index
        low_point_index_utc = filtered_low_points.index[-1].tz_convert(None)  # Convert to naive datetime
        high_points_index = selected_high_points.index.get_loc(low_point_index_utc) + 1

        # Find next high point starting from the current low point index
        while (high_points_index < len(selected_high_points) - 1 and
               selected_high_points.index[high_points_index] < filtered_low_points.index[-1] and
               selected_high_points.iloc[high_points_index]['Price'] <= selected_high_points.iloc[high_points_index + 1]['Price']):
            high_points_index += 1
        filtered_high_points = pd.concat([filtered_high_points, selected_high_points.iloc[[high_points_index]]])

        # Update low_points_index based on the last filtered high point's index
        high_point_index_utc = filtered_high_points.index[-1].tz_convert(None)  # Convert to naive datetime
        low_points_index = selected_low_points.index.get_loc(high_point_index_utc) + 1

    return filtered_low_points, filtered_high_points

if __name__ == "__main__":
    # Debug: Define the data for selected_low_points and selected_high_points
    selected_low_points_data = {
        'Datetime': ['2024-04-09 10:11:00-04:00', '2024-04-09 10:54:00-04:00', '2024-04-09 12:55:00-04:00',
                     '2024-04-09 14:02:00-04:00', '2024-04-09 15:10:00-04:00', '2024-04-10 09:48:00-04:00',
                     '2024-04-10 11:18:00-04:00', '2024-04-10 11:44:00-04:00', '2024-04-10 12:14:00-04:00',
                     '2024-04-10 13:10:00-04:00'],
        'Price': [518.78, 514.52, 516.72, 516.42, 515.86, 512.91, 513.62, 513.19, 513.05, 512.21]
    }

    selected_high_points_data = {
        'Datetime': ['2024-04-09 11:18:00-04:00', '2024-04-09 12:13:00-04:00', '2024-04-09 12:35:00-04:00',
                     '2024-04-09 13:22:00-04:00', '2024-04-09 14:21:00-04:00', '2024-04-09 14:53:00-04:00',
                     '2024-04-09 15:32:00-04:00', '2024-04-09 15:58:00-04:00', '2024-04-10 10:20:00-04:00',
                     '2024-04-10 10:39:00-04:00'],
        'Price': [516.41, 517.50, 517.69, 517.52, 517.15, 516.67, 518.70, 518.91, 514.84, 515.93]
    }

    # Convert data to DataFrame
    selected_low_points = pd.DataFrame(selected_low_points_data)
    selected_low_points['Datetime'] = pd.to_datetime(selected_low_points['Datetime'])
    selected_low_points.set_index('Datetime', inplace=True)

    selected_high_points = pd.DataFrame(selected_high_points_data)
    selected_high_points['Datetime'] = pd.to_datetime(selected_high_points['Datetime'])
    selected_high_points.set_index('Datetime', inplace=True)

    # Call the filtering function
    filtered_low_points, filtered_high_points = filter_points(selected_low_points, selected_high_points)

    # Output filtered_low_points and filtered_high_points
    print("Filtered Low Points:")
    print(filtered_low_points)
    print("\nFiltered High Points:")
    print(filtered_high_points)
