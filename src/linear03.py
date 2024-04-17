"""
our purpose is to find the slope(weight) and intercept(bias) for the data which come from internet
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def line (p1, p2): # p1, and p2 are tuple
    slope = (p2[1] - p1[1])/(p2[0] - p1[0])
    intercept = p1[1] - slope * p1[0]
    print(f"weight: {slope}, bias: {intercept}")
    def fn(x):
        return slope*x + intercept

    return fn

def mse(y, yPred): #Mean Square Error
    n = len(y)
    return np.sum((y-yPred)**2)/n

if __name__ == '__main__':
    df = pd.read_csv("data/death.csv") # df: DataFrame
    # print(df)
    column = "Recent 5-Year Trend (2) in Death Rates"
    df[column] = pd.to_numeric(df[column], errors='coerce').astype(np.float32) # change the column data to be float

    p1 = (0, 0)
    p2 = (2500, -5)
    fn = line(p1, p2)

    x = df.index
    y = df[column]
    yPred = fn(x)
    yPred1 = -0.001*x # partialy make change on slope only
    yPred2 = -0.0013*x + 0.6 # partialy make change on bias only
    yPred3 = -0.001158034661784768*x + 0.7159203290939331 # partialy make change on bias only

    error = mse(y, yPred2)
    print(error)
    plt.plot(x, y, 'ro')
    plt.plot([p1[0], p2[0]], [p1[1], p2[1]], 'b^')
    plt.plot(x, yPred)
    plt.plot(x, yPred1, 'g-' ) 
    plt.plot(x, yPred2, 'p-' ) 
    plt.plot(x, yPred3, color='#5589f4')
    plt.show()

""" Errors
8.934627507617003
5.16589120088249
5.06538725554982
5.112743821887031 ; computer model
5.090074851747643
"""