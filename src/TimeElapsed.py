import pandas as pd
import time

def measure_operation_time(operation_func, *args, **kwargs):
    """
    Measure the time taken by a given operation function.
    
    Parameters:
        operation_func (function): The function to measure.
        *args: Positional arguments to pass to the operation function.
        **kwargs: Keyword arguments to pass to the operation function.
        
    Returns:
        tuple: A tuple containing the time elapsed in seconds and the result(s) of the operation function.
    """
    start_time = time.time()
    results = operation_func(*args, **kwargs)
    end_time = time.time()
    time_elapsed = end_time - start_time
    return time_elapsed, results

# Example usage:
def read_CSV_file(file_path):
    
    # Read the data from CSV into a DataFrame
    df = pd.read_csv(file_path, index_col=0, parse_dates=True)
    
    return df


if __name__ == "__main__":
    # Define the file path
    file_path = "data/SPY_2024-04-15_2024-04-21_1m.csv"
    
    # Corrected usage: pass function object and its arguments separately
    time_elapsed, results = measure_operation_time(read_CSV_file, file_path)
    
    print("Time elapsed:", time_elapsed, "seconds")
    print("Results dataframe length:", len(results))  # This will print the tuple returned by read_CSV_file
