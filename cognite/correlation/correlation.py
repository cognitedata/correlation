from typing import *
from pandas import DataFrame
import numpy as np


def columns_by_correlation(df: DataFrame, relate_to: Union[int, str], lag=0) -> List[Tuple[Union[int, str], float]]:
    correlations = cross_correlate(df, df[relate_to], lag)
    sort_corr: List[int] = sorted(df.columns, key=lambda x: np.abs(correlations)[x], reverse=True)
    return [(col, corr) for col, corr in zip(sort_corr, correlations[sort_corr])]


def cross_correlate(df: DataFrame, relate_to_df: DataFrame, lag_idx=0):
    # Uneven spacing will lead to NaNs, which means less data will be used
    correlations = df.corrwith(relate_to_df.shift(-lag_idx))
    return correlations


def make_even(df, interval_ms):
    # Every point must fit the granularity
    # print(interval_ms, df.timestamp.diff(1))
    # print(df.timestamp.diff(1)[1:][50], interval_ms, np.unique(df.timestamp.diff(1)[1:]))
    print(np.unique(np.round(df.timestamp.diff()[1:] % interval_ms, decimals=5) == 0, return_counts=True))
    assert np.all(np.round(df.timestamp.diff(1)[1:] % interval_ms) == 0), \
        'Every data point in data frame must fit with the mode of the time deltas'

    # Check if already even
    if np.all(df.timestamp.diff()[1:] == interval_ms):
        return df

    # Get start and end points
    start = df.timestamp.values[0]
    end = df.timestamp.values[-1]
    df = df.set_index("timestamp")

    # Create a new index with all values in-between
    new_index = np.arange(start, end + 1, interval_ms)
    print(df)
    df = df.reindex(new_index)
    print(df)

    # Fill nans with interpolated values
    df = df.interpolate()

    return df.reset_index()


def columns_by_max_cross_correlation(df: DataFrame, relate_to: Union[int, str], lag: np.ndarray,
                                     interpolator: str = 'akima') -> DataFrame:
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

    # Sorted indices by correlation
    max_corrs = np.abs(cross_correlations).max(axis=0)
    sorted_idxs = cross_correlations[::-1].argsort()
    return [(col, corr, lag) for col, corr, lag in
            zip(df.columns[sorted_idxs], max_corrs[sorted_idxs], max_lags[sorted_idxs])]


