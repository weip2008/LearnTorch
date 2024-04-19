<h1>Idea of AI Stock Modeling</h1>

1. use boolinger line to determine sell/buy points(max:sell, min:buy value)
2. smooth (9, 15) all data, use the smoothed data as close[array]
3. calculate vilocity for all points(array)
4. calculate accelerate for all points(array)
5. use a window find stock input smooth data
6. create datasets: 
   a. input(close, vilocity, accelerate, weekdays, time, volume)
   b. output(sell, buy)
7. create model
8. use the model to test training data

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

