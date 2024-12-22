import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm, ttest_ind
from scipy import stats
from statsmodels.stats.proportion import test_proportions_2indep
import statsmodels.stats.weightstats as ws
from typing import List, Tuple, Union
import seaborn as sns
import tqdm

def get_mde(
    mean: float,
    std: float,
    sample_size: int,
    n_groups: int = 2,
    n_metrics: int = 1,
    compare: str = 'only_control',
    alpha_correction: bool = False,
    alpha: float = 0.05,
    beta: float = 0.2
) -> Tuple[float, float]:
    """Calculate the Minimum Detectable Effect (MDE)."""
    if alpha_correction and compare == 'together':
        alpha_correction = math.factorial(n_groups) / (math.factorial(n_groups - 2) * 2)
        t_alpha = norm.ppf(1 - (alpha / 2) / (alpha_correction * n_metrics))
    elif alpha_correction and compare == 'only_control':
        alpha_correction = n_groups - 1
        t_alpha = norm.ppf(1 - (alpha / 2) / (alpha_correction * n_metrics))
    else:
        t_alpha = norm.ppf(1 - (alpha / 2))
    
    t_beta = norm.ppf(1 - beta)
    variance = std**2
    mde = (t_alpha + t_beta) * np.sqrt((variance*4) / (sample_size))
    return mde * 100 / mean, mde

def get_mde_ratio(
    num: np.ndarray,
    denom: np.ndarray,
    sample_size: int,
    n_groups: int = 2,
    n_metrics: int = 1,
    compare: str = 'only_control',
    alpha_correction: bool = False,
    alpha: float = 0.05,
    beta: float = 0.2
) -> Tuple[float, float]:
    """Calculate MDE for ratios."""
    if alpha_correction and compare == 'together':
        alpha_correction = math.factorial(n_groups) / (math.factorial(n_groups - 2) * 2)
        t_alpha = norm.ppf(1 - (alpha / 2) / (alpha_correction * n_metrics))
    elif alpha_correction and compare == 'only_control':
        alpha_correction = n_groups - 1
        t_alpha = norm.ppf(1 - (alpha / 2) / (alpha_correction * n_metrics))
    else:
        t_alpha = norm.ppf(1 - (alpha / 2))
    
    mean_nom = np.mean(num)
    mean_denom = np.mean(denom)
    std_nom = np.std(num)
    std_denom = np.std(denom)
    cov_nom_denom = np.cov(num, denom)[0, 1]
    mean = np.sum(num) / np.sum(denom)
    var_metric = (
        (std_nom**2) / (mean_denom**2) +
        (mean_nom**2) / (mean_denom**4) * (std_denom**2) -
        2 * mean_nom / (mean_denom**3) * cov_nom_denom
    )
    variance = var_metric
    t_beta = norm.ppf(1 - beta)
    mde = (t_alpha + t_beta) * np.sqrt((variance*4) / (sample_size))
    return mde * 100 / mean, mde

def plot_p_value_over_time(
    dates: List[Union[str, float]],
    test_group: List[List[float]],
    control_group: List[List[float]],
    significance_level: float = 0.05
) -> None:
    """Plot P-value dynamics over time."""
    if len(dates) != len(test_group) or len(dates) != len(control_group):
        raise ValueError("Lengths of 'dates', 'test_group', and 'control_group' must match.")
    
    p_values = [
        ttest_ind(test_data, control_data, equal_var=False)[1]
        for test_data, control_data in zip(test_group, control_group)
    ]
    
    plt.figure(figsize=(15, 6))
    plt.plot(dates, p_values, marker='o', linestyle='-', label='P-value', color='blue')
    plt.axhline(y=significance_level, color='red', linestyle='--', label=f'Significance level ({significance_level})')
    plt.fill_between(
        dates, 0, p_values, where=np.array(p_values) < significance_level,
        color='green', alpha=0.2, label='Below significance'
    )
    plt.title('P-value Over Time During Experiment', fontsize=14)
    plt.xlabel('Date/Period', fontsize=12)
    plt.ylabel('P-value', fontsize=12)
    plt.xticks(rotation=45, fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(alpha=0.5)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.show()

def ttest(
    df: pd.DataFrame,
    metric_col: str,
    ab_group_col: str,
    pairs_list: List[Tuple[str, str]] = [(0, 1)],
    corrected_ci: float = 0.95,
    flag_notation: bool = False
) -> pd.DataFrame:
    """Perform t-tests between two groups."""
    res_table = pd.DataFrame()
    tail = (1 + corrected_ci) / 2
    for pair in pairs_list:
        sample0 = df.loc[df[ab_group_col] == pair[0], metric_col]
        sample1 = df.loc[df[ab_group_col] == pair[1], metric_col]
        m0 = sample0.mean()
        m1 = sample1.mean()
        v0 = sample0.std()**2
        v1 = sample1.std()**2
        n0 = len(sample0)
        n1 = len(sample1)
        value = ws.ttest_ind(
            sample0,
            sample1,
            alternative='two-sided',
            usevar='unequal'
        )
        se = np.sqrt(v0 / n0 + v1 / n1)
        delta = m1 - m0
        delta_per = (m1 / m0 - 1) * 100
        lb = delta - stats.t.ppf(tail, value[2]) * se
        ub = delta + stats.t.ppf(tail, value[2]) * se
        lb_per = lb * 100 / m0
        ub_per = ub * 100 / m0
        
        if flag_notation == True:
            print(f'\nComparison between groups: {pair[0]} and {pair[1]}')
            print(f't-statistic: {value[0]}, pvalue: {value[1]}, df: {value[2]}')
            print(f'delta = {delta}')
            print(f'delta,% = {delta_per}%')
            print(f'Confidence interval for delta: ({lb}, {ub})')
            print(f'Confidence interval for delta, %: ({lb_per}, {ub_per})')

        result = pd.DataFrame(
            np.array([metric_col, n0, n1, pair[0], pair[1], value[0], value[2], value[1], m0, m1, delta, delta_per, lb, ub, lb_per, ub_per]).reshape(1, -1),
            columns=['metric_name', 
                     'group0_sample_size', 
                     'group1_sample_size',
                     'group0', 
                     'group1', 
                     't_statistic', 
                     'df', 
                     'pvalue', 
                     'mean0', 
                     'mean1', 
                     'diff_mean', 
                     'diff_mean_%', 
                     'lower_boundary', 
                     'upper_boundary', 
                     'lower_boundary_%', 
                     'upper_boundary_%']
        )
        res_table = pd.concat([res_table, result])
    
    for column in res_table.columns[5:]:
        res_table[column] = res_table[column].astype(float)

    return res_table

def ztest_proportion(
    df: pd.DataFrame,
    metric_col: str,
    ab_group_col: str,
    pairs_list: List[Tuple[str, str]] = [(0, 1)],
    corrected_ci: float = 0.95,
    flag_notation: bool = False
) -> pd.DataFrame:
    """Perform proportion tests between two groups."""
    res_table = pd.DataFrame()
    tail = (1 + corrected_ci) / 2
    for pair in pairs_list:
        num0 = df[df[ab_group_col] == pair[0]][metric_col].sum()
        denom0 = df[df[ab_group_col] == pair[0]][metric_col].count()
        num1 = df[df[ab_group_col] == pair[1]][metric_col].sum()
        denom1 = df[df[ab_group_col] == pair[1]][metric_col].count()
        p0 = num0 / denom0
        p1 = num1 / denom1
        std0 = df[df[ab_group_col] == pair[0]][metric_col].std()
        std1 = df[df[ab_group_col] == pair[1]][metric_col].std()
        r = test_proportions_2indep(
            num0, denom0,
            num1, denom1,
            value=0,
            method='wald',
            compare='diff',
            alternative='two-sided',
            return_results = True
        )
        se = np.sqrt(r.variance)
        delta = p1 - p0
        delta_per = (p1 / p0 - 1) * 100
        lb = delta - stats.norm.ppf(tail) * se
        ub = delta + stats.norm.ppf(tail) * se
        lb_per = lb * 100 / p0
        ub_per = ub * 100 / p0
        
        if flag_notation == True:
            print(f'\nComparison between groups: {pair[0]} and {pair[1]}')
            print(f'statistic: {r.statistic}, pvalue: {r.pvalue}')
            print(f'delta = {delta}')
            print(f'delta,% = {delta_per}%')
            print(f'Confidence interval for delta: ({lb}, {ub})')
            print(f'Confidence interval for delta, %: ({lb_per}, {ub_per})')

        result = pd.DataFrame(
            np.array([metric_col, denom0, denom1, pair[0], pair[1], r.statistic, r.pvalue, p0, p1, delta, delta_per, lb, ub, lb_per, ub_per]).reshape(1, -1),
            columns=['metric_name', 
                     'group0_sample_size', 
                     'group1_sample_size', 
                     'group0', 
                     'group1', 
                     'statistic', 
                     'pvalue', 
                     'mean0', 
                     'mean1', 
                     'diff_mean', 
                     'diff_mean, %', 
                     'lower_boundary', 
                     'upper_boundary', 
                     'lower_boundary_%', 
                     'upper_boundary_%',]
        )
        res_table = pd.concat([res_table, result])

        for column in res_table.columns[5:]:
            res_table[column] = res_table[column].astype(float)
        
    return res_table

def ttest_delta(
    df: pd.DataFrame,
    metric_num_col: str,
    metric_denom_col: str,
    ab_group_col: str,
    pairs_list: List[Tuple[str, str]] = [(0, 1)],
    corrected_ci: float = 0.95,
    flag_notation: bool = False
    ) -> pd.DataFrame:
    """Perform t-tests on delta between two ratios."""
    def get_ratio_var(
    num: np.ndarray,
    denom: np.ndarray
    ) -> float:
        cov = np.cov(num, denom, ddof=1)[0, 1]
        var = (
            (np.std(num) ** 2) / (np.mean(denom) ** 2) +
            (np.mean(num) ** 2) / (np.mean(denom) ** 4) * (np.std(denom) ** 2) -
            2 * np.mean(num) / (np.mean(denom) ** 3) * cov
        )
        return var

    res_table = pd.DataFrame()
    for pair in pairs_list:
        num0 = df.loc[df[ab_group_col] == pair[0], metric_num_col]
        denom0 = df.loc[df[ab_group_col] == pair[0], metric_denom_col]
        num1 = df.loc[df[ab_group_col] == pair[1], metric_num_col]
        denom1 = df.loc[df[ab_group_col] == pair[1], metric_denom_col]
        group0_sample_size = df.loc[df[ab_group_col] == pair[0], metric_num_col].count()
        group1_sample_size = df.loc[df[ab_group_col] == pair[1], metric_num_col].count()
        metric_name = f'({metric_num_col}, {metric_denom_col})'
        ratio0 = np.sum(num0) / np.sum(denom0)
        ratio1 = np.sum(num1) / np.sum(denom1)
        se = np.sqrt(get_ratio_var(num0, denom0)/len(num0) + get_ratio_var(num1, denom1)/len(num1))
        delta = ratio1 - ratio0
        delta_per = (ratio1 / ratio0 - 1) * 100
        statistic = delta / se
        df_ = len(num0) + len(num1) - 2
        pvalue = (1 - stats.t.cdf(np.abs(statistic), df_)) * 2
        tail = (1 + corrected_ci) / 2
        lb = delta - stats.t.ppf(tail, df_) * se
        ub = delta + stats.t.ppf(tail, df_) * se
        lb_per = lb * 100 / ratio0
        ub_per = ub * 100 / ratio0
        
        if flag_notation == True:
            print(f'\nComparison between groups: {pair[0]} and {pair[1]}')
            print(f'statistic: {statistic}, pvalue: {pvalue}')
            print(f'delta = {delta}')
            print(f'delta,% = {delta_per}%')
            print(f'Confidence interval for delta: ({lb}, {ub})')
            print(f'Confidence interval for delta, %: ({lb_per}, {ub_per})')

        result = pd.DataFrame(
            np.array([metric_name, group0_sample_size, group1_sample_size, pair[0], pair[1], statistic, pvalue, ratio0, ratio1, delta, delta_per, lb, ub, lb_per, ub_per]).reshape(1, -1),
            columns=['metric_name', 'group0_sample_size', 'group1_sample_size', 'group0', 'group1', 't_statistic', 'pvalue', 'mean0', 'mean1', 'diff_mean', 'diff_mean, %', 'lb', 'ub', 'lb%', 'ub%']
        )
        res_table = pd.concat([res_table, result])

        for column in res_table.columns[5:]:
            res_table[column] = res_table[column].astype(float)

    return res_table

def plot_p_value_distribution(
    control_group: np.ndarray,
    test_group: np.ndarray,
    num_tests: int = 1000
) -> None:
    """Plot the distribution of p-values from A/A tests."""
    np.random.seed(42)

    p_values = [
        ttest_ind(np.random.choice(control_group, size=len(control_group), replace=True),
                  np.random.choice(test_group, size=len(test_group), replace=True), equal_var=False)[1]
        for _ in tqdm(range(num_tests))
    ]
    
    plt.figure(figsize=(15, 6))
    plt.hist(p_values, bins=50, color='blue', alpha=0.7, edgecolor='black')
    plt.axvline(x=0.05, color='red', linestyle='--', label='Significance level (0.05)')
    plt.title('P-value Distribution from A/A Tests', fontsize=14)
    plt.xlabel('P-value', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.grid(alpha=0.5)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.show()

def plot_pvalue_ecdf(control_group, test_group, title=None):

    pvalues = [
        ttest_ind(np.random.choice(control_group[control_group['has_treatment'] == 1]['gmv'], size=64, replace=True),
                np.random.choice(test_group[test_group['has_treatment'] == 0]['gmv'], size=64, replace=True), equal_var=False)[1]
        for _ in tqdm(range(1000))
]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    if title:
        plt.suptitle(title)

    sns.histplot(pvalues, ax=ax1, bins=20, stat='density') # type: ignore
    ax1.plot([0,1],[1,1], 'k--')
    ax1.set(xlabel='p-value', ylabel='Density')

    sns.ecdfplot(pvalues, ax=ax2) # type: ignore
    ax2.plot([0,1],[0,1], 'k--')
    ax2.set(xlabel='p-value', ylabel='Probability')
    ax2.grid()

def method_benjamini_hochberg(
    pvalues: np.ndarray,
    alpha: float = 0.05
) -> np.ndarray:
    """Apply the Benjamini-Hochberg procedure for multiple hypothesis testing."""
    m = len(pvalues)
    array_alpha = np.arange(1, m + 1) * alpha / m
    sorted_pvalue_indexes = np.argsort(pvalues)
    res = np.zeros(m)
    for idx, pvalue_index in enumerate(sorted_pvalue_indexes):
        pvalue = pvalues[pvalue_index]
        alpha_ = array_alpha[idx]
        if pvalue <= alpha_:
            res[pvalue_index] = 1
        else:
            break
    return res.astype(int)