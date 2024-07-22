
```
(env) C:\Users\wangq\workspace\LearnTorch>pip install zigzag
Collecting zigzag
  Using cached zigzag-0.3.2.tar.gz (112 kB)
error: invalid-pyproject-build-system-requires

× Can not process zigzag from https://files.pythonhosted.org/packages/d4/88/028c3d29b17cd2d4cfcc2305bc42e0363d64451e3eda880aab1d0e579069/zigzag-0.3.2.tar.gz
╰─> This package has an invalid `build-system.requires` key in pyproject.toml.
    It contains an invalid requirement: 'Cython>=^0.29'

note: This is an issue with the package mentioned above, not pip.
hint: See PEP 518 for the detailed specification.
```

```
pip cache purge
```

```
(env) C:\Users\wangq\workspace\LearnTorch>pip cache purge
Files removed: 462

(env) C:\Users\wangq\workspace\LearnTorch>pip install zigzag
Collecting zigzag
  Downloading zigzag-0.3.2.tar.gz (112 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 112.3/112.3 kB 3.2 MB/s eta 0:00:00
error: invalid-pyproject-build-system-requires

× Can not process zigzag from https://files.pythonhosted.org/packages/d4/88/028c3d29b17cd2d4cfcc2305bc42e0363d64451e3eda880aab1d0e579069/zigzag-0.3.2.tar.gz
╰─> This package has an invalid `build-system.requires` key in pyproject.toml.
    It contains an invalid requirement: 'Cython>=^0.29'

note: This is an issue with the package mentioned above, not pip.
hint: See PEP 518 for the detailed specification.

```