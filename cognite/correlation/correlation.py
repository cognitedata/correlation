from typing import *
import pandas as pd
import numpy as np


def columns_by_correlation(df: pd.DataFrame, relate_to: Union[int, str], lag=0) -> List[Tuple[Union[int, str], float]]:
    correlations = cross_correlate(df, df[relate_to], lag)
    sort_corr: List[int] = sorted(df.columns, key=lambda x: np.abs(correlations)[x], reverse=True)
    return [(col, corr) for col, corr in zip(sort_corr, correlations[sort_corr])]


def cross_correlate(df: pd.DataFrame, relate_to_df: pd.DataFrame, lag_idx=0):
    # Uneven spacing will lead to NaNs, which means less data will be used
    correlations = df.corrwith(relate_to_df.shift(-lag_idx))
    return correlations


def columns_by_max_cross_correlation(df: pd.DataFrame,
                                     relate_to: Union[int, str],
                                     lags: pd.TimedeltaIndex,
                                     return_cross_correlation_df: bool = False)\
                                        -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
    """Find lag of highest correlation and return relevant information for all tags.
    Note that the operation requires a DataFrame with even temporal spacing.

    It is recommended to either have a lot of data in the data frame, or to use a short time frame for the lags,
    as the results are unstable if too few data points overlap in the time shifted time series.

    Args:
        df (DataFrame): Time series data
        relate_to (Union[int, str]): Column to compare others with
        lags (pd.TimedeltaIndex): Pandas sequence of timedeltas for shifting the time series
        return_cross_correlation_df (bool): Whether or not to return the cross correlations for all columns.
            This is a DataFrame containing the cross correlation for all calculated times
    Returns:
        DataFrame: Sorted DataFrame with columns (column, max_correlation, lag) by descending absolute correlation
        DataFrame, optional: Cross correlation for each time lag for each sensor
    """
    diffs = df.index.to_series().diff()
    dmin, dmax = diffs.min(), diffs.max()
    assert dmin == dmax, 'Time series must be evenly spaced'
    assert df.isna().sum().sum() == 0, 'NaN values must be interpolated before calculating cross-correlation'

    # Enforce most common time spacing in main column
    time_interval = dmin
    relate_to_df = df[relate_to]

    # Round lag timings to integer multiples of the minimum shifts
    lags_idx = np.unique(lags // time_interval).astype(int)
    # print('lags_idx:', lags_idx)
    # Interpolate for all levels of lag
    cross_correlations = np.zeros(shape=(lags_idx.shape[0], df.shape[1]))
    for i, l in enumerate(lags_idx):
        correlations = cross_correlate(df, relate_to_df, lag_idx=l)
        cross_correlations[i] = correlations
        # print(df, relate_to_df.shift(-l))
    # print(cross_correlations)

    # Find optimal lag for every feature
    max_corr_idxs = np.nanargmax(np.abs(cross_correlations), axis=0)
    max_lags = lags_idx[max_corr_idxs] * time_interval

    # Sorted indices by correlation
    max_corrs = cross_correlations[(max_corr_idxs, np.arange(max_corr_idxs.shape[0]))]
    sorted_idxs = max_corrs.argsort()[::-1]
    # print(max_corr_idxs.shape, max_corrs.shape, sorted_idxs)
    out_df = pd.DataFrame({
        'col': df.columns[sorted_idxs],
        'corr': max_corrs[sorted_idxs],
        'lag': - max_lags[sorted_idxs]
    })
    if return_cross_correlation_df:
        cross_correlations = pd.DataFrame(cross_correlations)
        cross_correlations.set_index(lags_idx * time_interval, inplace=True)
        cross_correlations.columns = df.columns
        return out_df, cross_correlations
    return out_df
