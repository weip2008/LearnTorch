import os
from pathlib import Path

project = "goldpin"

# Dynamically determine the base directory
try:
    # Use __file__ to get the directory of this file
    script_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    # Fall back to the current working directory if __file__ is not defined
    # script_dir = os.getcwd()
    script_dir = str(Path().resolve())

# Define base_dir relative to the script's location
base_dir = os.path.abspath(os.path.join(script_dir, '..', '..', project))

# Other common paths
src_path = os.path.join(base_dir, 'src')
log_path = os.path.join(base_dir, 'log')


# Exported variables for easy access
__all__ = ['base_dir', 'src_path', 'log_path']
