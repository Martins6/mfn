# mfn

Implementation in Python of the MFN method to measure complexity & entropy of time series.
This is useful if you need features for a model such as Porfolio Optimization, clustering of time series, etc.
This method is the implementation of the paper _Scientific progress in information theory quantifiers._ (Chaos, Solitons & Fractals, 170, 113260.,
Martins, A. M. F., Fernandes, L. H. S., & Nascimento, A. D. C. (2023).)

## Installation

```bash
pip install mfn
```

or using Poetry

```bash
poetry add mfn
```

## Usage

```python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from mfn.entropy import MFN

## Generating a time series with trend and noise.
time_series = np.arange(0, 100, 1)
time_series = time_series + np.random.normal(0, 10, size=len(time_series))

value_dict = MFN(
    time_series,
    b=10,
    B=.1,
    size=100,
    dx=3
)

f, ax = plt.subplots(figsize=(6, 6))
value_df = pd.DataFrame(value_dict).reset_index()
value_df = value_df.melt(id_vars='index', value_vars=value_df.columns[1:])
sns.barplot(value_df, x='variable', y='value', errorbar="sd")
plt.title("MFN method results")
f.tight_layout()
plt.show()
```
