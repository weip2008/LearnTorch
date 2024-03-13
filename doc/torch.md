<h1>PyTorch Learning Notes</h1>

## Getting Started

1. set local virtual environment (env)
python -m venv env
2. pip install torch

## Errors
❓ not available module
![](images/unavailable%20modue.png)
📝 close VS Code, reopen, and run all code from the top.
## AI on Fashion
![](images/neuralNetwork4handwritingDigits.png)
28X28=784 input, 2 modle layer,  0-9 output
$$f_{l+1} = \sigma (w_l a_l + b_l) $$
$w_l$: weight for layer l
$b_l$: bias for layer l
$sigma$: activation function
$f_{l+1}$: l+1 function of layer l
the purpose of modeling is find each $w_l$ and $b_l$

![sample fashion images](images/fashionSample.png)

* [tensor basics](torchBasics.ipynb)

* [Understand image data, and squeeze(), transpose()](src/fashion01.py)