# TSOP

**TSOP** — A Cython/C++ operator library for 2D financial time series.



## Features

- **Cross-sectional**: `cs_zscore`, `cs_winsor`, `cs_scale`, `cs_rank`  
- **Time-series**: `ts_mean`, `ts_std`, `ts_diff`, `ts_corr_binary`, `ts_delay` …  
- Full NumPy API compatibility & zero-copy  
- High performance: C++17 + OpenMP (optional) + release-GIL  

## Installation

```bash
# From PyPI
pip install tsop

# From source
git clone https://github.com/yourname/tsop.git
cd tsop
pip install .
```


## Quickstart

```python
import numpy as np
from tsop import cs_zscore, ts_mean

A = np.random.randn(1000, 250)
Z = cs_zscore(A)         # cross-sectional z-score
M = ts_mean(A, days=5)   # 5-day rolling mean
```

## Development

```bash
# install dev dependencies
pip install -r requirements-dev.txt

# lint & format
make fmt
make lint

# build & test
make build
pytest
```


## CI / Cross-Platform Testing

We use GitHub Actions to test on **Linux**, **macOS** and **Windows** across multiple Python versions.