import math
import pandas as pd
from scipy import stats


def spearman_ds(ds, actual_col, pred_col, p_value=False, nan_policy='omit'):
    '''
    Calculate Spearman correlation coefficient between actual and predicted efficiencies.
    By default, omits NaN values and outputs only the coefficient without the corresponding p-value.

    :parameter ds: dataset to analyse.
    :parameter actual_col: # position of column with actual efficiencies e.g. 2nd column -> 2.
    :parameter pred_col: # position of column with each model's predictions.
    :parameter p_value: If True, calculate the two-sided p-value for a hypothesis test whose null hypothesis is that two sets of data are uncorrelated.
    :parameter nan_policy: Defines how to handle when input contains nan. The following options are available (default is 'omit'):
                            'propagate': returns nan, 'raise': throws an error, 'omit': performs the calculations ignoring nan values.
    :return: Spearman correlation coefficient (or a tuple of Spearman correlation coefficient and corresponding p-value).
    '''

    Spearman, pval = stats.spearmanr(
        ds.iloc[:, actual_col - 1], ds.iloc[:, pred_col - 1], nan_policy=nan_policy)
    if p_value == False:
        return Spearman
    else:
        return Spearman, pval


def ndcg_at_k(ds, k, actual_col, pred_col, bins=False, reverse=False, multiple=False):
    """
    Calculate nDCG@k score using logarithmic discount given a dataset with actual and predicted efficiencies.
    The relevance value of each gRNA is its efficacy score.

    :parameter ds: dataset to analyse.
    :parameter k: highest value to calculate nDCG i.e. only consider the highest k scores in the ranking.
    :parameter actual_col: # position of column with actual efficiencies e.g. 2nd column -> 2.
    :parameter pred_col: # position of column with each model's predictions.
    :parameter bins (default=False): If True, group actual efficiencies into 5 equal width bins to have a discrete relevance value.
                                     By default, use actual efficiency as the relevance value.
    :parameter reverse (default=False): If True, calculate nDCG for reverse ordering to be used as a baseline.
    :parameter multiple (default=False): If True, calculate and store nDCG together with samples' indices for plotting.
    :return: nDCG@k score (or a list of scores and indices if parameter multiple is True).
    """

    # Discrete relevance value

    if bins == True:
        quantile_list = [0.0, 0.20, 0.40, 0.60, 0.80, 1.0]
        bins = ds.iloc[:, actual_col-1].quantile(quantile_list)
        labels = [0, 1, 2, 3, 4]
        ds['binned'] = pd.cut(ds.iloc[:, actual_col-1],
                              bins, labels=labels, include_lowest=True)

    # Reverse ordering (worst case)
        if reverse == True:
            ds_sort = ds.sort_values(by=ds.columns[actual_col-1])
            y_pred = ds_sort['binned'].values

    # Ordering based on each model's predictions
        else:
            ds_sort = ds.sort_values(
                by=ds.columns[pred_col-1], ascending=False)
            y_pred = ds_sort['binned'].values

    # Actual relevance value

    else:

        # Reverse ordering (worst case)
        if reverse == True:
            ds_sort = ds.sort_values(by=ds.columns[actual_col-1])
            y_pred = ds_sort.iloc[:, actual_col-1].values

    # Ordering based on each model's predictions
        else:
            ds_sort = ds.sort_values(
                by=ds.columns[pred_col-1], ascending=False)
            y_pred = ds_sort.iloc[:, actual_col-1].values

    # Ideal ordering
    Y_test = sorted(y_pred, reverse=True)

    # Calculate nDCG@k

    thr = []
    score = []

    a = b = 0
    for i in range(0, k):
        a += y_pred[i]/math.log2(i+2)
        b += Y_test[i] / math.log2(i + 2)
        thr.append(i)
        if not b:
            score.append(0)
        else:
            score.append(a/b)

    if multiple == True:
        return score, thr
    else:
        return score[-1]
