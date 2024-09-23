"""
All variable name in config.ini must be lowercase!
"""
import configparser
import time
import functools

class Config:
    def __init__(self, config_file):
        try:
            self.config = configparser.ConfigParser()
            self.config.read(config_file)

            if not self.config.sections():
                raise FileNotFoundError(f"Configuration file '{config_file}' not found or is empty.")
            for section in self.config.sections():
                for key, value in self.config.items(section):
                    setattr(self, key, value)
        except configparser.Error as e:
            print(f"Error parsing config file: {e}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

    def get(self, section, key, fallback=None):
        try:
            return self.config.get(section, key, fallback=fallback)
        except configparser.Error as e:
            print(f"Error retrieving '{key}' from section '{section}': {e}")
            return fallback

def execution_time(func):
    """
    A decorator to calculate the execution time of any function.
    
    :param func: The function to be wrapped.
    :return: A wrapper function that calculates the execution time.
    """
    @functools.wraps(func)  # This preserves the original function's metadata
    def wrapper(*args, **kwargs):
        start_time = time.time()  # Get the start time
        result = func(*args, **kwargs)  # Execute the wrapped function
        end_time = time.time()  # Get the end time
        execution_duration = end_time - start_time
        print(f"Execution time of {func.__name__}: {execution_duration:.4f} seconds")
        return result  # Return the result of the function
    return wrapper

# Example usage
@execution_time
def example_function(n):
    total = 0
    for i in range(n):
        total += i
    return total

if __name__ == '__main__':
    config = Config('gru\src\config.ini')
    
    if hasattr(config, 'host'):
        print(config.host)  # Output: localhost if the file and key are correct
    else:
        print("Host attribute is not found in the config")
    print(config.sqlite_db)

    # Call the function to test
    result = example_function(1000000)
    print("Result:", result)