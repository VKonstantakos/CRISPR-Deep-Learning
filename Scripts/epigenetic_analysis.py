import numpy as np
import pandas as pd
from scipy import stats


def extract_identical_sequences(ds, dt):
    '''
    Extract identical sequences between pairs of datasets, including
    the corresponding epigenetic features and actual efficiency.
    The resulting dataframe can then be used to calculate the correlation
    between the efficiencies of these sequences.

    :parameters ds, dt: pairs of datasets to analyse.
    :return: dataframe with identical sequences, epigenetic features, and efficiencies.
    '''

    a = ds['seq']
    b = dt['seq']
    ds['seq2'] = a.isin(b)
    dt['seq2'] = b.isin(a)

    ds = ds[ds['seq2'] == True]
    dt = dt[dt['seq2'] == True]

    cell1 = ds.sort_values("seq").reset_index(drop=True)
    cell2 = dt.sort_values("seq").reset_index(drop=True)

    Y = cell1['Normalized efficacy']
    y = cell2['Normalized efficacy']

    Y_class = cell1['Efficacy']
    y_class = cell2['Efficacy']

    df = pd.DataFrame(
        {'Sequence': cell1['seq'], 'CTCF Cell_1': cell1['ctcf'], 'DNase Cell_1': cell1['dnase'], 'H3K4me3 Cell_1': cell1['h3k4me3'], 'RRBS Cell_1': cell1['rrbs'],
         'CTCF Cell_2': cell2['ctcf'], 'DNase Cell_2': cell2['dnase'], 'H3K4me3 Cell_2': cell2['h3k4me3'], 'RRBS Cell_2': cell2['rrbs'],
         'Normalized Efficacy Cell_1': Y, 'Normalized Efficacy Cell_2': y, 'Efficacy Cell_1': Y_class, 'Efficacy Cell_2': y_class,
         'Absolute Error': np.abs(Y-y), 'Squared Error': np.square(Y-y), 'Misclassification': np.abs(Y_class-y_class)}
    )

    return df


def pair_epigenetic_correlation(ds):
    '''
    Calculate the correlation and mean absolute error of gRNA efficiency
    between pairs of datasets, which include identical sequences
    with at least one different epigenetic feature.

    :parameter ds: dataset with the identical sequences to analyse.
    :return: dataframe with the relevant sequences.
    '''

    epi = ds[(ds.iloc[:, 1] != ds.iloc[:, 5]) |
             (ds.iloc[:, 2] != ds.iloc[:, 6]) |
             (ds.iloc[:, 3] != ds.iloc[:, 7]) |
             (ds.iloc[:, 4] != ds.iloc[:, 8])]

    spearman, _ = stats.spearmanr(epi.iloc[:, 9], epi.iloc[:, 10])
    error = epi.iloc[:, 13]
    mean = error.mean()

    print("Spearman correlation = %.3f" % (spearman))
    print("Mean absolute error = %.3f" % (mean))

    return epi


def total_epigenetic_correlation(epi1, epi2, epi3):
    '''
    Calculate the overall correlation and mean absolute error of gRNA efficiency
    between all the pairs of datasets, which include identical sequences
    with at least one different epigenetic feature.

    :parameters epi1, epi2, epi3: datasets with the identical sequences to analyse.
    :return: Spearman correlation coefficient.
    '''

    Y = np.concatenate(
        (epi1.iloc[:, 9], epi2.iloc[:, 9], epi3.iloc[:, 9]), axis=0)
    y = np.concatenate(
        (epi1.iloc[:, 10], epi2.iloc[:, 10], epi3.iloc[:, 10]), axis=0)

    error = np.concatenate(
        (epi1.iloc[:, 13], epi2.iloc[:, 13], epi3.iloc[:, 13]), axis=0)
    mean = error.mean()
    spearman, _ = stats.spearmanr(Y, y)

    print("Spearman correlation = %.3f" % (spearman))
    print("Mean absolute error = %.3f" % (mean))

    return spearman
