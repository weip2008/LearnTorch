# AI Stock Modeling Idea

## Overview
This document outlines the idea of using AI for stock modeling by incorporating various technical indicators and creating a dataset to train and test a model for predicting long (buy) and short (sell) positions. 

## Steps

1. **Use Bollinger Bands to Determine Long/Short Points**
   - Identify the maximum point as a long position and the minimum point as a short position.

2. **Smooth Data**
   - Apply smoothing with periods of 9 and 15 to the data and use the smoothed data as the close prices in the dataset.

3. **Calculate Velocity and Acceleration**
   - Calculate velocity and acceleration for all data points.

4. **Windowed Data for Stock Inputs**
   - Use a moving window to segment the data for model input.

5. **Create Datasets**
   - Input: close prices, velocity, acceleration, weekdays, time, and volume.
   - Output: long and short positions.

6. **Create Model**
   - Build a neural network model to predict long and short positions based on the input dataset.

7. **Test Model**
   - Use the trained model to test on the training data to evaluate performance.

## Idea of Selecting Long, Short, and Hold Points
🛠🎯 **Leave this section for 周浩**

### Concerns and Issues
> *To be addressed by 周浩*

## Create Datasets
- [Create datasets from stock raw data](../src/datasets.py)

## Save and Load Datasets from File

### CSV File Format
```
long,short,[(weekdays,time,close,velocity,acceleration,volume),(...)]
0.1,0.2,0.3,0.4,0.5,0.6,0.7
0.2,0.3,0.4,0.5,0.6,0.8,0.9
0.3,0.4,0.5,0.6,0.7,0.5,0.4
...
```

## Velocity and Acceleration
Velocity:
$$v_i=\frac {c_{i+1}-c_{i-1}} {t_{i+1}-t_{i-1}}$$
Acceleration:
$$a_i=\frac {v_{i+1}-v_{i-1}} {t_{i+1}-t_{i-1}}$$

## Training and Test Data Design

### CSV File Format
```
long,short,weekdays,time,price,volume,velocity,acceleration,...
1,0,2.0,10.0,503.039,2.3,0.12,1232,2.0,10.123,503.3,2.1,0.3,1354,...
1,0,2.0,10.0,503.039,2.3,0.12,1232,2.0,10.123,503.3,2.1,0.3,1354,...
... ...
```
- [Sample data file](../data/SPY_TraningData06.csv)

### Tensor Format
```python
outputs_tensor = torch.tensor(outputs).reshape(18,2)
inputs_tensor = torch.tensor(inputs).reshape(18,1,6,10)
```
- `18`: Number of training data points.
- `2`: Output dimensions for `long` and `short`.
- `6`: Columns for input features: weekdays, time, close, velocity, acceleration, volume.
- `10`: Window size.

#### Sample Input Tensor
```python
tensor([[[ 4.0000e+00,  1.0117e+01,  5.1337e+02,  ...,  1.0133e+01,
           5.1327e+02,  3.8961e+05],
         ...
         [-1.8000e-01,  ..., -2.4000e-01, -6.0000e-02]],

        [[ 5.0000e+00,  ...,  1.5167e+01],
         ...
         [-6.0000e-02,  ...,  2.0000e-02]],

        ...
        [[ 3.0000e+00,  ...,  1.4733e+01],
         ...
         [ 1.8000e-01,  ...,  3.0000e-02]]])
```

#### Sample Training Output Tensor
```python
tensor([[1., 0.],
        ...
        [0., 1.]])
```
- `index=0`: Long position.
- `index=1`: Short position.

### Test Dataset Format
The input structure for test datasets is the same as for training datasets. However, the output structure is different; it is a one-dimensional array indicating the correct class (index) for each window.

#### Example:

```py
test_output_tensor = torch.tensor([int(y == 1.0) for x, y in outputs])
```

### Conclusion
Running
- [Read stock data, build model, save model to a file, stock.py](../src/stock.py)

Results:
- Most of the time, accuracy is around 50%.
- Occasionally, accuracy reaches 72%.

```py
file_path = 'stockdata/SPY_TraningData_30_07.csv'
```

❌😢 **Conclusion**:
A 50% accuracy indicates that the current data structure and NN model cannot reliably predict stock movements.

- [Use model file to predict stock data (same as training data)](../src/stock1.py)

Results:
- Occasionally, accuracy reaches 83%.

```python
file_path = 'stockdata/SPY_TraningData_30_07.csv'
```

![Occasionally 83% accuracy](images/StockTrainModel.png)