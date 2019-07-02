from typing import *
from pandas import DataFrame
import numpy as np


def columns_by_correlation(df: DataFrame, relate_to: Union[int, str], lag=0) -> List[Tuple[Union[int, str], float]]:
    correlations = cross_correlate(**locals())
    sort_corr: List[int] = sorted(df.columns, key=lambda x: np.abs(correlations)[x], reverse=True)
    return [(col, corr) for col, corr in zip(sort_corr, correlations[sort_corr])]


def cross_correlate(df: DataFrame, relate_to: Union[int, str], lag=0):
    correlations = df.corrwith(df[relate_to].shift(-lag))
    return correlations


def columns_by_max_cross_correlation(df: DataFrame, relate_to: Union[int, str], lag=np.ndarray) -> List[Tuple[Union[str, int], float, int]]:
    """Find lag of highest correlation and return relevant information for all tags.

    Args:
        df (DataFrame): Time series data
        relate_to (Union[int, str]): Column to compare others with
        lag (np.ndarray): Iterable list of lag values to try

    Returns:
        List[Tuple[Union[str, int], float, int]]: Sorted list of (column, max_correlation, lag) by descending correlation
    """
    cross_correlations = np.zeros(shape=(lag.shape[0], df.shape[1]))
    for i, l in enumerate(lag):
        cross_correlations[i] = cross_correlate(df, relate_to, lag=l)
    # Find optimal lag
    max_corr_idxs = np.abs(cross_correlations).argmax(axis=0)
    max_lags = lag[max_corr_idxs]


    return [(col, corr, lag) for col, corr, lag in zip()]


