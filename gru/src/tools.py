"""
Gather small tools to verify our data and ensure it meets our expectations.
"""

import csv
from config import Config
import matplotlib.pyplot as plt

def plot(filepath, row, yLabel = "Price (close)"):
    """
    plot column of the row in data format of '1, 0, -1,[(1,2,3,4,5),(6,7,8,9,10), ...]'
    """
    priceList = getPrice(filepath, row)
    # Plot the price list
    plt.plot(priceList)
    plt.xlabel('Index')
    plt.ylabel(yLabel)
    plt.title(f'Plot for Row {row}')
    plt.show()

def getPrice(filepath, line=0):
    with open(filepath, mode='r') as file:
        count = 0
        for row in file:
            if count == line:
                priceList = parse(row)
                break
            count += 1
    return priceList

def parse(data_line, col=2):
    data_part = '[' + data_line.split('[', 1)[1]
    data_tuples = eval(data_part)
    print(f"Number of points in the slice: {len(data_tuples)}")
    element_list = [tup[col] for tup in data_tuples]

    return element_list

if __name__ == "__main__":
    config = Config('gru/src/config.ini')
    filepath = config.training_file_path
    filepath = "data\SPX_1m_TrainingData_HL_76_500.txt"
    plot(filepath, 3735)