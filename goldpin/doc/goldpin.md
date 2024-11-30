<h1>Gold Pin development notes</h1>

## File Structure
goldpin/
├── doc/
│   └── test.py
├── src/
│   ├── __init__.py
│   └── a1_preparation.py

```py test.py
import sys
import os

project = "goldpin"
base_dir = "C:/Users/wangq/workspace/LearnTorch/" + project
src_path = os.path.abspath(os.path.join(base_dir, 'src'))
print(src_path)  # Check the resolved path
sys.path.append(src_path)

from a1_preparation import DataProcessor
```

```py utilities.py
class DataSource:
    config = Config("goldpin/src/config.ini")
    log = Logger("goldpin//log/gru.log", logger_name='data')
    conn = None

```
## Rule

MNQ 5 minutes time frame
1. startTime: Record the current price as startPrice (variable).
2. MACD Condition: From startTime - 3 bars, ensure MACD > 0 and each histogram value decreases sequentially.
3. 5x MACD Condition: From startTime, confirm the MACD crosses above the signal, with histogram values decreasing sequentially.
4. Stochastic RSI: Ensure the fast line crosses above the slow line.
5. Fishing: Within the interval startTime + timeInterval (variable), set the buy price to startPrice - 30 (variable) points if all conditions are satisfied.
   
