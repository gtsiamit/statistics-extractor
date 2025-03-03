# Statistics Extractor


## Overview
Statistics Extractor is a Python package designed to compute various statistical measures from tabular data. It supports input in the form of NumPy arrays, Pandas Series, or DataFrames and efficiently extracts the following statistics:
- Mean
- Median
- Mode
- Variance
- Standard Deviation
- 25th Percentile
- 75th Percentile
- Skewness
- Kurtosis

## Repository files

### File structure
```
statistics-extractor
├─ README.md
├─ requirements.txt
├─ setup.py
├─ statistics_extractor
│  ├─ __init__.py
│  ├─ extractor.py
│  ├─ handler.py
│  ├─ statistics_functions.py
│  └─ utils.py
└─ tests
   ├─ run_local.py
   ├─ test_extractor.py
   ├─ test_handler.py
   ├─ test_statistic_functions.py
   └─ test_utils.py
```

### File descriptions
- `statistics_extractor/extractor.py` : Contains the `extract_statistics` function, which serves as the main entry point for computing statistics on tabular data.
- `statistics_extractor/handler.py` : Defines the `handle_extraction` function, which orchestrates the extraction process by converting input data and computing required statistics.
- `statistics_extractor/statistics_functions.py` : Includes implementations of various statistical functions.
- `statistics_extractor/utils.py` : Contains utility functions such as handling arrays and type conversions.
- `tests/` : Contains python tests for the package modules.


## Installation

### Dependencies
- Python 3.12 is required.
- `setup.py` : Used for building and installing the package.
- `requirements.txt` : Contains the python library requirements.
- In order for the package to be installed the GitHub repository needs to be cloned locally.

### Package installation
- To install the package and its dependencies, in the root directory of the project, `statistics-extractor`, run:
```bash
pip install .
```

### Development and Testing installation
- To install the package and its dependencies, for development and testing purposes, in the root directory of the project, `statistics-extractor`, run:
```bash
pip install -e .[dev]
```


## Usage
The statistics extractor can be used with 1D and 2D NumPy arrays, Pandas Series and DataFrames. If a 2D array or dataframe is passed then the statistics are calculated for each column.

Example of usage with a 2D NumPy array.
```python
import numpy as np
from statistics_extractor.extractor import extract_statistics

# define a numpy array
array = np.array([[1, 2, 3], [4, 5, 6], [10, 12, 15], [4, 5, 6]])

# call the extract_statistics function
# column names that will be displayed in the output are defined
result = extract_statistics(x=array, feature_names=["feature1", "feature2", "feature3"])
```

Example of usage with an 1D NumPy array.
```python
import numpy as np
from statistics_extractor.extractor import extract_statistics

# define a numpy array
array = np.array([1, 2, 3, 4, 5, 6, 10, 12, 4, 24])

# call the extract_statistics function
# the column name that will be displayed in the output is defined
result = extract_statistics(x=array, feature_names=["feature1"])
```

Example of usage with a pandas DataFrame.
```python
import pandas as pd
from statistics_extractor.extractor import extract_statistics

# define a pandas DataFrame
df = pd.DataFrame([[1, 2, 3], [4, 5, 6], [10, 12, 15], [4, 5, 6]])

# call the extract_statistics function
# column names that will be displayed in the output are defined
result = extract_statistics(x=df, feature_names=["feature1", "feature2", "feature3"])
```

The feature_names is not required to be set. If not set then the column names will be generated automatically by the function.
```python
import numpy as np
from statistics_extractor.extractor import extract_statistics

# define a numpy array
array = np.array([[1, 2, 3], [4, 5, 6], [10, 12, 15], [4, 5, 6]])

# call the extract_statistics function
result = extract_statistics(x=array)
```


## Development and Testing
If the `dev` requirements are installed as mentioned in [Development and Testing installation](#development-and-testing-installation), the python tests can be executed locally with pytest:
```bash
pytest tests/
```
