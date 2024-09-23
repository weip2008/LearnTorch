<h1> GRU Action Forecast</h1>

```mermaid
graph LR

Data[generateDataset.py<br><br>SPX_1m_TrainingData_HL_80_500.txt<br>SPX_1m_TestingData_HL_80_500.txt]
Model[gruModel.py<br><br>GRU_model_with_LH_fixlen_data_501.pth]
Test[test.py]
Data-->Model-->Test
```

## Table of Contents
- [Table of Contents](#table-of-contents)
- [Todo](#todo)
- [Generate Dataset](#generate-dataset)
  - [Output files](#output-files)
  - [Input](#input)
  - [Output](#output)
- [Create GRU Model](#create-gru-model)
  - [Input](#input-1)
  - [Output](#output-1)
- [Test the model](#test-the-model)
  - [Input](#input-2)
  - [Output](#output-2)

## Todo
1. change trainning data format
2. send Test output to a file for future reference
3. all global variables should read from a configuration file
4. clean code make all definitions at begining
5. use class
6. get rid of zigzagplus1.py
7. optimize Debug
8. optimize logging


## Generate Dataset
* [Generate dataset Source Code](../src/generateDataset.py)
### Output files
1. [traning dataset](../data/SPX_1m_TrainingData_HL_80_500.txt)
2. [testing dataset](../data/SPX_1m_TestingData_HL_80_500.txt)

5 column data group
1. day of weeek
2. time of day
3. close price
4. velocity
5. accelerat

first column
1=long
0=short

total 80 points end by long/short point for each row

### Input
SQLite database file: [data/stock_bigdata_2019-2023.db]

### Output
![](images/trainning_testing_data.png)
* [Trainning Dataset](/data/SPX_1m_TrainingData_HL_80_500.txt)
* [Testing Dataset](/data/SPX_1m_TestingData_HL_80_500.txt)

## Create GRU Model
* [Generate GRU Action Forecast model](../src/gruModel.py)

### Input
* [Trainning Dataset](/data/SPX_1m_TrainingData_HL_80_500.txt)
* [Testing Dataset](/data/SPX_1m_TestingData_HL_80_500.txt)

### Output
* [/GRU_model_with_LH_fixlen_data_501.pth](/GRU_model_with_LH_fixlen_data_501.pth)

## Test the model
* [Test model get R-Square and MSE](../src/test.py)
  
### Input
* [/GRU_model_with_LH_fixlen_data_501.pth](/GRU_model_with_LH_fixlen_data_501.pth)

### Output

```txt
Current date and time: 2024-09-23 09:36:55
1. Load testing data from data/SPX_1m_TestingData_HL_80_500.txt
Data shape: (1684, 80, 5)
Targets shape: (1684, 1)
2. Define dataset and dataloader
Current date and time: 2024-09-23 09:36:56
3. Instantiate the model, define the loss function and the optimize
Current date and time: 2024-09-23 09:36:56
Number of layers: 5
3. Load trained model from GRU_model_with_LH_fixlen_data_501.pth
4. Start testing loop
Current date and time: 2024-09-23 09:36:56
Test Loss (MSE): 0.00353319
Mean Absolute Error (MAE): 0.02026430
R-squared (R2): 0.99644500
Current date and time: 2024-09-23 09:36:58
Saved categorized signals to file : data/SPX_1m_HL_80_500_GRU_fixlen_500.txt
Current date and time: 2024-09-23 09:36:59
```