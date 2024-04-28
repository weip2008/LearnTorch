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
sell,buy,[(weekdays,time,close,slope,accelerate,volume),(...)]
0.1,0.2,0.3,0.4,0.5,0.6,0.7
0.2,0.3,0.4,0.5,0.6,0.8,0.9
0.3,0.4,0.5,0.6,0.7,0.5,0.4
...
```

## velocity and acceleration

$$v_i=\frac {c_{i+1}-c_{i-1}} {t_{i+1}-t_{i-1}}$$
i.e. the velocity at $t_i$ equals the difference of the "close" at $t_{i+1}$ and $t_{i-1}$. same as accelerate as below:
$$a_i=\frac {v_{i+1}-v_{i-1}} {t_{i+1}-t_{i-1}}$$
