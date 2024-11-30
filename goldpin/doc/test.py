import sys
import os

base_dir = "C:/Users/wangq/workspace/LearnTorch/goldpin"
src_path = os.path.abspath(os.path.join(base_dir, 'src'))
print(src_path)  # Check the resolved path
sys.path.append(src_path)

from a1_preparation import DataProcessor