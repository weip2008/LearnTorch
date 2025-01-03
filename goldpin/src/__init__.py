from a1_preparation import *
from logger import *
from config import *
from utilities import *
import zigzagplus1

import sys
import os

project = "goldpin"

# Dynamically determine the base directory using the current working directory
script_dir = os.getcwd() + '/../../'
base_dir = os.path.abspath(os.path.join(script_dir, project))
src_path = os.path.join(base_dir, 'src')

print(src_path)  # Check the resolved path
sys.path.append(src_path)
