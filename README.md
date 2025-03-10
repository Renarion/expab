# AB Testing Library

A comprehensive Python library designed for A/B testing analysis, providing essential statistical tools for hypothesis testing, confidence interval calculations, p-value visualizations, and multiple hypothesis correction methods. Whether you're running simple two-group comparisons or more complex multi-group analyses, this library equips you with the necessary functions to derive meaningful insights from your experiments.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
  - [Calculate Minimum Detectable Effect (MDE)](#calculate-minimum-detectable-effect-mde)
  - [Calculate MDE for Ratios](#calculate-mde-for-ratios)
  - [Plot P-value Over Time](#plot-p-value-over-time)
  - [Perform T-Tests Between Groups](#perform-t-tests-between-groups)
  - [Perform Proportion Tests Between Groups](#perform-proportion-tests-between-groups)
  - [Perform T-Tests on Delta Between Ratios](#perform-t-tests-on-delta-between-ratios)
  - [Plot P-value Distribution from A/A Tests](#plot-p-value-distribution-from-aa-tests)
  - [Plot P-value ECDF](#plot-p-value-ecdf)
  - [Apply Benjamini-Hochberg Procedure](#apply-benjamini-hochberg-procedure)
- [Example](#example)
- [License](#license)

## Features

- Minimum Detectable Effect (MDE) Calculations: Determine the smallest effect size you can detect with your experimental setup.
- Ratio-Specific MDE: Calculate MDE for metrics expressed as ratios.
- Statistical Testing: Perform t-tests and proportion tests between groups to evaluate significance.
- P-value Visualization: Visualize p-value dynamics over time and their distributions.
- Multiple Hypothesis Correction: Apply the Benjamini-Hochberg procedure to control the false discovery rate.

## Essential Mathematical Formulas

#### <br> Sample Size </br>

$$
n = \frac{(Z_{1-\alpha/2} + Z_{1-\beta})^2 \cdot (Var_{\text{test}}+Var_{\text{control}})}{\delta^2}
$$

#### <br> MDE </br>        

$$
\text{MDE} = \frac{(Z_{1-\alpha/2} + Z_{1-\beta}) \cdot \sqrt{(Var_{\text{test}}+Var_{\text{control}})}}{\sqrt{n}}
$$

#### <br> T-test (Welch's) </br>

$$
t = \frac{\bar{X}_1 - \bar{X}_2}{\sqrt{\frac{S_1^2}{n_1} + \frac{S_2^2}{n_2}}}
$$

#### <br> Z-test (Wald's) </br>

$$
z = \frac{\hat{p}_1 - \hat{p}_2}{\sqrt{\hat{p}(1 - \hat{p}) \left(\frac{1}{n_1} + \frac{1}{n_2}\right)}}
$$

#### <br> Delta method </br>

$$
\text{Var}\left(\frac{X}{Y}\right) \approx \frac{\text{Var}(X)}{\bar{Y}^2} + \frac{\bar{X}^2 \cdot \text{Var}(Y)}{\bar{Y}^4} - 2 \cdot \frac{\bar{X} \cdot \text{Cov}(X, Y)}{\bar{Y}^3}
$$


#### <br> Benjamini-Hochberg </br>

$$
p_k \leq \frac{k}{m} \cdot Q
$$


## Installation

You can install the library using pip:

```bash
pip install expab
```

Or, if you have the source code cloned locally, install it in editable mode:

```bash
pip install -e .
```

Deleting the package with terminal to re-install

```bash
pip uninstall expab
```

When new version is released, it needs to be updated with the following command

```bash
pip install expab --upgrade
```

## Usage

### Generated data for testing library
```python
import random
import numpy as np
import pandas as pd

# Data for using mde

sample_size = 10000
dataset = [random.randrange(1000, sample_size) for i in range(sample_size)]
dataset_binary = [random.randrange(0, 2) for i in range(sample_size)]
mean = np.mean(dataset)
std = np.std(dataset)

# Dataframe for applying tests

sample_size = 10000

df = pd.DataFrame()
df['user_id'] = pd.Series([i for i in range(sample_size)])
df['has_treatment'] = pd.Series(np.random.randint(0, 2, sample_size))
df['value'] = pd.Series([random.randrange(1000, sample_size) for i in range(sample_size)])
df['order_number'] = pd.Series([random.randrange(0, 10) for i in range(sample_size)])
df['binary'] = pd.Series([random.randrange(0, 2) for i in range(sample_size)])
```

### Calculate Minimum Detectable Effect (MDE)

Calculate the Minimum Detectable Effect (MDE) given the mean, standard deviation, and sample size.
```python
from expab import get_mde

mean = 100
std = 15
sample_size = 1000

mde_percentage, mde_absolute = get_mde(mean, std, [sample_size])
print(f"MDE: {mde_percentage}% ({mde_absolute})")
```
### Calculate MDE for Ratios

Calculate MDE when your metric is a ratio (e.g., conversion rate).
```python
from expab import get_mde_ratio
import numpy as np

numerator = np.array([50050, 55051, 60502, 65506, 70304])
denominator = np.array([500, 550, 600, 650, 700])
sample_size = 1000

mde_ratio_percentage, mde_ratio_absolute = get_mde_ratio(numerator, denominator, [sample_size])
print(f"MDE Ratio: {mde_ratio_percentage}% ({mde_ratio_absolute})")
```
### Plot P-value Over Time

Visualize how p-values change over different time periods during your experiment.

```python
from expab import plot_p_value_over_time

dates = ['2024-01', '2024-02', '2024-03', '2024-04']
test_group = [[1.2, 1.3, 1.1], [1.4, 1.5, 1.3], [1.5, 1.6, 1.4], [1.7, 1.8, 1.6]]
control_group = [[1.1, 1.0, 1.2], [1.2, 1.1, 1.3], [1.3, 1.2, 1.4], [1.4, 1.3, 1.5]]

plot_p_value_over_time(dates, test_group, control_group)
```

### Perform T-Tests Between Groups

Conduct t-tests between two groups and obtain statistical metrics.
```python
from expab import ttest
import pandas as pd
import numpy as np

# Sample DataFrame
data = {
    'group': [0]*100 + [1]*100,
    'metric': np.random.normal(100, 15, 200)
}
df = pd.DataFrame(data)

results = ttest(df, metric_col='metric', ab_group_col='group')
results
```
### Perform Proportion Tests Between Groups

Perform proportion tests to compare binary outcomes between groups.
```python
from expab import ztest_proportion
import pandas as pd
import numpy as np

# Sample DataFrame
data = {
    'group': [0]*1000 + [1]*1000,
    'success': np.random.binomial(1, 0.1, 2000)
}
df = pd.DataFrame(data)

results = ztest_proportion(df, metric_col='success', ab_group_col='group')
results
```
### Perform T-Tests on Delta Between Ratios

Compare the delta between two ratio metrics across groups.
```python
from expab import ttest_delta
import pandas as pd
import numpy as np

# Sample DataFrame
data = {
    'group': [0]*1000 + [1]*1000,
    'numerator': np.random.binomial(1, 0.1, 2000),
    'denominator': np.random.binomial(10, 0.5, 2000)
}
df = pd.DataFrame(data)

results = ttest_delta(
    df, 
    metric_num_col='numerator', 
    metric_denom_col='denominator', 
    ab_group_col='group'
)
results
```
### Plot P-value Distribution from A/A Tests

Visualize the distribution of p-values from A/A testing to assess test calibration.
```python
from expab import plot_p_value_distribution
import numpy as np
import pandas as pd

control_group = np.random.normal(100, 15, 1000)
test_group = np.random.normal(100, 15, 1000)

plot_p_value_distribution(control_group, test_group)
```
### Plot P-value ECDF

Create Empirical Cumulative Distribution Function (ECDF) plots for p-values.
```python
from expab import plot_pvalue_ecdf
import pandas as pd
import numpy as np

# Sample DataFrame
data = {
    'has_treatment': [1]*1000 + [0]*1000,
    'gmv': np.random.normal(100, 15, 2000)
}
control_group = pd.DataFrame(data)
test_group = pd.DataFrame(data)

plot_pvalue_ecdf(control_group, test_group, title='P-value ECDF')
```
### Apply Benjamini-Hochberg Procedure

Control the false discovery rate when performing multiple hypothesis tests.
```python
from expab import method_benjamini_hochberg
import numpy as np
import pandas as pd

pvalues = np.random.uniform(0, 1, 100)
adjusted = method_benjamini_hochberg(pvalues, alpha=0.05)
print(adjusted)
```

# License

This project is licensed under the [MIT License.](https://opensource.org/license/mit)
