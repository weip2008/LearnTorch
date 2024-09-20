<h1>Understand LLM</h1>

* [Paper PDF](../LLM.pdf)
* [LLM 源代码](https://github.com/ictnlp/llama-omni)


## fix pip install zigzag issue
1. download

```
(env) C:\Users\wangq\workspace\LearnTorch\zigzag-0.3.2>curl -O https://files.pythonhosted.org/packages/d4/88/028c3d29b17cd2d4cfcc2305bc42e0363d64451e3eda880aab1d0e579069/zigzag-0.3.2.tar.gz
```
2. [modify pyprojct.toml](../../zigzag-0.3.2/pyproject.toml)
3. [modify build.py](../../zigzag-0.3.2/build.py)
4. install wheel, setuptools, cython

```
(env) C:\Users\wangq\workspace\LearnTorch\zigzag-0.3.2>pip install --upgrade pip setuptools wheel cython numpy   
```

5. install zigzag
   
```
(env) C:\Users\wangq\workspace\LearnTorch\zigzag-0.3.2>pip install .
Processing c:\users\wangq\workspace\learntorch\zigzag-0.3.2
  Installing build dependencies ... done
  Getting requirements to build wheel ... done
  Preparing metadata (pyproject.toml) ... done
Building wheels for collected packages: zigzag
  Building wheel for zigzag (pyproject.toml) ... done
  Created wheel for zigzag: filename=zigzag-0.0.0-py3-none-any.whl size=2582 sha256=dccead129bc7bb74ff259483980239c580074dd2c9843193f69c543cd60a51ca
  Stored in directory: c:\users\wangq\appdata\local\pip\cache\wheels\f8\f0\55\59c4bc0f21689350e2249e57dc0396923cdf16d9f733a957e3
Successfully built zigzag
Installing collected packages: zigzag
Successfully installed zigzag-0.0.0

```

```
(env) C:\Users\wangq\workspace\LearnTorch>pip list
Package            Version
------------------ -----------
asttokens          2.4.1
beautifulsoup4     4.12.3
certifi            2024.7.4
charset-normalizer 3.3.2
colorama           0.4.6
comm               0.2.2
contourpy          1.2.1
cycler             0.12.1
Cython             3.0.11
debugpy            1.8.3
decorator          5.1.1
exceptiongroup     1.2.2
executing          2.0.1
filelock           3.15.4
fonttools          4.53.1
frozendict         2.4.4
fsspec             2024.6.1
html5lib           1.1
idna               3.7
intel-openmp       2021.4.0
ipykernel          6.29.5
ipython            8.26.0
jedi               0.19.1
Jinja2             3.1.4
jupyter_client     8.6.2
jupyter_core       5.7.2
kiwisolver         1.4.5
lxml               5.2.2
MarkupSafe         2.1.5
matplotlib         3.9.1
matplotlib-inline  0.1.7
mkl                2021.4.0
mplfinance         0.12.10b0
mpmath             1.3.0
multitasking       0.0.11
nest-asyncio       1.6.0
networkx           3.3
numpy              2.1.1
packaging          24.1
pandas             2.2.2
pandas-ta          0.3.14b0
parso              0.8.4
peewee             3.17.6
pillow             10.4.0
pip                24.2
platformdirs       4.2.2
plotly             5.23.0
prompt_toolkit     3.0.47
psutil             6.0.0
pure_eval          0.2.3
Pygments           2.18.0
pyparsing          3.1.2
python-dateutil    2.9.0.post0
pytz               2024.1
pywin32            306
pyzmq              26.0.3
requests           2.32.3
scipy              1.14.0
setuptools         75.1.0
six                1.16.0
soupsieve          2.5
stack-data         0.6.3
sympy              1.13.1
ta                 0.11.0
tbb                2021.13.0
tenacity           9.0.0
torch              2.3.1
torchvision        0.18.1
tornado            6.4.1
traitlets          5.14.3
typing_extensions  4.12.2
tzdata             2024.1
urllib3            2.2.2
wcwidth            0.2.13
webencodings       0.5.1
wheel              0.44.0
yfinance           0.2.41
zigzag             0.0.0
```