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