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


def make_even(df, interval_ms, interpolator):
    # Every point must fit the granularity
    # print(interval_ms, df.timestamp.diff(1))
    # print(df.timestamp.diff(1)[1:][50], interval_ms, np.unique(df.timestamp.diff(1)[1:]))
    # print(np.unique(np.round(df.timestamp.diff()[1:] % interval_ms, decimals=5) == 0, return_counts=True))
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
    # print(df)
    df = df.reindex(new_index)
    # print(df)

    # Fill nans with interpolated values
    df = df.interpolate(method=interpolator)

    return df.reset_index()


def columns_by_max_cross_correlation(df: DataFrame, relate_to: Union[int, str], lag: np.ndarray,
                                     interpolator: str = 'linear') -> DataFrame:
    """Find lag of highest correlation and return relevant information for all tags.
    Note that the operation requires a DataFrame with even temporal spacing, and will interpolate to match the
    smallest spacing.

    Args:
        df (DataFrame): Time series data
        relate_to (Union[int, str]): Column to compare others with
        lag (np.ndarray): NumPy array for time shifting space to search
        interpolator (str): Interpolator to use for shifting the timeseries
    Returns:
        DataFrame: Sorted DataFrame with columns (column, max_correlation, lag) by descending correlation
    """
    # Enforce most common time spacing in main column
    interval_ms = df.timestamp.diff().min()
    relate_to_df = make_even(df[['timestamp', relate_to]], interval_ms, interpolator=interpolator)[relate_to]

    # Round lag timings to integer shifts
    lag_idx = np.unique(lag // interval_ms).astype(int)

    # Interpolate for all levels of lag
    cross_correlations = np.zeros(shape=(lag.shape[0], df.shape[1]))
    for i, l in enumerate(lag_idx):
        cross_correlations[i] = cross_correlate(df, relate_to_df, lag_idx=l)
        # print(df, relate_to_df.shift(-l))
    # print(cross_correlations)

    # Find optimal lag for every feature
    max_corr_idxs = np.nanargmax(np.abs(cross_correlations), axis=0)
    max_lags = lag[max_corr_idxs]

    # Sorted indices by correlation
    max_corrs = np.nanmax(np.abs(cross_correlations), axis=0)
    sorted_idxs = max_corrs.argsort()[::-1]
    # print(max_corr_idxs.shape, max_corrs.shape, sorted_idxs)
    out_df = DataFrame({
        'col': df.columns[sorted_idxs],
        'corr': max_corrs[sorted_idxs],
        'lag': max_lags[sorted_idxs]
    })
    return out_df
