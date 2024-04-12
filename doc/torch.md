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

* [Load data from network, Understand image data, and squeeze(), transpose()](../src/fashion01.py)
* [create model based on all images, and save model into a file](../src/fashion02.py)
* [load model from file, and predict a given image](../src/fashion03.py)

```mermaid
graph LR

TRAIN(train data)
MODEL(create a model)
TRAINING[Training Process]
TRAIN_OUTPUT(expected Train<br>Output)
TRAINED_MODEL(Trained Model)

TEST(test data)
TEST_OUTPUT(expected Test<br>Output)
TESTING[Testing Process]
ACCURACY[Modeling Accuracy]

REAL(Real Data)
PRED[Predictation process]
RESULT[Prediction <br>Result]

TRAIN --> TRAINING
TRAIN_OUTPUT --> TRAINING
MODEL --> TRAINING
TRAINING ==> TRAINED_MODEL

TEST --> TESTING
TRAINED_MODEL --> TESTING
TEST_OUTPUT --> TESTING
TESTING ==> ACCURACY

TRAINED_MODEL --> PRED
REAL --> PRED
PRED ==> RESULT
ACCURACY --How good it is--> RESULT

classDef start fill:#3cdf77,stroke:#1a6d38,stroke-width:2px,color:white;
classDef html fill:#F46624,stroke:#F46624,stroke-width:4px,color:white;
classDef js fill:#73dbf7,stroke:#194652,stroke-width:2px;
classDef if fill:#f2e589,stroke:black,stroke-width:2px;
classDef db fill:#aaafb0,stroke:#1a404a,stroke-width:2px;
classDef end1 fill:#f17168,stroke:#902a23,stroke-width:2px,color:white;

class TRAIN,TRAIN_OUTPUT,MODEL,TEST,TRAINED_MODEL,TEST_OUTPUT,REAL start
class TRAINING,TESTING,PRED js
class ACCURACY,RESULT html
```

## Homework
* create model for handwriting digits.
```py
train_dataset = datasets.MNIST('data/MNIST_data/', download=True, train=True, transform=ToTensor())
test_data = datasets.MNIST('data/MNIST_data/', download=True, train=False, transform=ToTensor())

```

* [Understand weight in linear function](../src/weight.py)
* [Understand ReLU activate function](../src/relu.py)

## backpropagation
* [wikipedia Backpropagation Explain](https://en.wikipedia.org/wiki/Backpropagation)