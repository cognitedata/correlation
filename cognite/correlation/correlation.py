from typing import *

import numpy as np
import pandas as pd


def cross_correlate(df: pd.DataFrame, relate_to_series: pd.Series, lag_idx=0):
    """Calculate cross correlation for a given lag.

    It is recommended to either have a lot of data in the data frame, or to use a short time frame for the lags,
    as the results are unstable if too few data points overlap in the time shifted time series.

    Args:
        df (pandas.Series): Time series data to correlate with some series
        relate_to_series (pandas.Series): Pandas Series with time series data to relate df to. Must have the same
            temporal spacing as df.
        lag_idx (int): How many indices to move the DataFrame in relation to the series.

    Returns:
        pandas.DataFrame: Pandas DataFrame containing the cross correlations of the columns.

    Examples:
        >>> df = pd.DataFrame({'x': (1, 7, 3, 5), 'y': (3, 7, 6, 4)})
        >>> cross_correlate(df, df['x'], lag_idx=1)
        x   -0.981981
        y   -0.960769
    """

    correlations = df.corrwith(relate_to_series.shift(-lag_idx))
    return correlations


def columns_by_max_cross_correlation(
    df: pd.DataFrame, relate_to: Union[int, str], lags: pd.TimedeltaIndex, return_cross_correlation_df: bool = False
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
    """Find lag of highest correlation and return relevant information for all columns of the inputted DataFrame.
    Note that the operation requires a DataFrame with even temporal spacing.

    It is recommended to either have a lot of data in the data frame, or to use a short time frame for the lags,
    as the results are unstable if too few data points overlap in the time shifted time series.

    Args:
        df (pandas.DataFrame): Time series data
        relate_to (Union[int, str]): Name of column to compare others with
        lags (pandas.TimedeltaIndex): Pandas sequence of timedeltas for shifting the time series
        return_cross_correlation_df (bool): Whether or not to return the cross correlations for all columns.
            This is a pandas.DataFrame containing the cross correlation for all calculated lags.
    Returns:
        Union[pandas.DataFrame, Tuple[pandas.DataFrame, pandas.DataFrame]]: Pandas DataFrame containing results of
        calculations, and optionally a DataFrame containing the cross correlations for each column at all calculated
        lags.

    Examples:
        Return maximum correlations and time lags for a simple dataframe.

        >>> df = pd.DataFrame({'datetime': pd.date_range(datetime(2017, 1, 1), datetime(2017, 1, 3), periods=10),
        >>>                    'x': np.sin(np.linspace(0, 2 * np.pi, 10)),
        >>>                    'y': np.sin(np.linspace(1, 2 * np.pi + 1, 10))}).set_index('datetime')
        >>> lags = pd.timedelta_range(timedelta(days=-3), timedelta(), periods=10)
        >>> columns_by_cross_correlation(df, 'x', lags)

    """
    diffs = df.index.to_series().diff()
    dmin, dmax = diffs.min(), diffs.max()
    assert dmin == dmax, "Time series must be evenly spaced"
    assert df.isna().sum().sum() == 0, "NaN values must be interpolated away before calculating cross-correlation"

    # Enforce most common time spacing in main column
    time_interval = dmin
    relate_to_df = df[relate_to]

    # Round lag timings to integer multiples of the minimum shifts
    lags_idx = np.unique(lags // time_interval).astype(int)

    # Interpolate for all levels of lag
    cross_correlations = np.zeros(shape=(lags_idx.shape[0], df.shape[1]))
    for i, l in enumerate(lags_idx):
        correlations = cross_correlate(df, relate_to_df, lag_idx=l)
        cross_correlations[i] = correlations

    # Find optimal lag for every feature
    max_corr_idxs = np.nanargmax(np.abs(cross_correlations), axis=0)
    max_lags = lags_idx[max_corr_idxs] * time_interval

    # Sorted indices by correlation
    max_corrs = cross_correlations[(max_corr_idxs, np.arange(max_corr_idxs.shape[0]))]
    sorted_idxs = np.abs(max_corrs).argsort()[::-1]

    out_df = pd.DataFrame(
        {"col": df.columns[sorted_idxs], "corr": max_corrs[sorted_idxs], "lag": -max_lags[sorted_idxs]}
    )
    if return_cross_correlation_df:
        cross_correlations = pd.DataFrame(cross_correlations)
        cross_correlations.set_index(lags_idx * time_interval, inplace=True)
        cross_correlations.columns = df.columns
        cross_correlations = cross_correlations[df.columns[sorted_idxs]]
        return out_df, cross_correlations
    return out_df
