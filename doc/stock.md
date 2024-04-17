<h1>Idea of AI Stock Modeling</h1>

1. use boolinger line to determine sell/buy points
2. use a window find stock input raw data
3. create datasets: 
   a. input(close, slope, accelerate, weekdays, time)
   b. output(sell, buy)
4. create model
5. use the model to test training data

## Create datasets


* [create datasets from stock raw data](../src/datasets.py)

## save and load datasets from file

* better file format

```csv
close,slope,accelerate,weekdays,time,sell,buy
0.1,0.2,0.3,0.4,0.5,0.6,0.7
0.2,0.3,0.4,0.5,0.6,0.8,0.9
0.3,0.4,0.5,0.6,0.7,0.5,0.4
...
```

