import datetime
from typing import *

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import ticker


def _time_ticks(x, pos=None):
    td = datetime.timedelta(seconds=int(abs(x)))
    if x < 0:
        return "– " + str(td)
    return str(td)


def _td_to_sec(td: datetime.timedelta):
    return td.days * 24 * 60 * 60 + td.seconds


def plot_cross_correlations(
    cross_correlation_df: pd.DataFrame,
    cols_to_plot: Union[Iterable[str], Iterable[int]] = None,
    separate_plots: bool = True,
    mpl_args: Dict[str, Any] = None,
) -> None:
    """Plot cross-correlations over time lags. The cols_to_plot parameter can either be an iterable of strings of
    columns to plot, or a list of integers of the indices of the sorted list of columns to display.

    Args:
        cross_correlation_df (pd.DataFrame): The DataFrame returned from columns_by_max_correlation (with the optional
            parameter return_cross_correlation_df enabled).
        cols_to_plot (Union[Iterable[str], Iterable[int]]): What columns to plot or, alternatively an iterable of
            indices for which of the ordered columns to plot.
        separate_plots (bool): Whether or not to divide the plots into individual plots, or to keep them in one plot
        mpl_args (Dict[str, Any]): Additional parameters for Matplotlib plotting function
    Returns:
        None

    Examples:
        Plot the cross-correlation for columns numbered 1-4 of the dataframe (0 is excluded to not plot auto-correlation).

        >>> plot_cross_correlation(cross_correlation_df, range(1,5))
    """
    if mpl_args is None:
        mpl_args = {}
    if cols_to_plot is not None:
        # Make indexable
        cols_to_plot = np.array(cols_to_plot)
        assert cols_to_plot.dtype.kind in ("i", "U"), "cols_to_plot must be an iterable of strings or ints"
        if cols_to_plot.dtype.kind == "i":
            assert np.all(
                (cols_to_plot < cross_correlation_df.shape[1]) & (cols_to_plot >= 0)
            ), "When indexing columns, the indices must be within range"
            cols_to_plot = cross_correlation_df.columns[cols_to_plot]
        assert np.all(
            [col in cross_correlation_df.columns for col in cols_to_plot]
        ), "All columns to plot must exist in the DataFrame"
        cross_correlation_df = cross_correlation_df[cols_to_plot]
    df_sec = cross_correlation_df.set_index(cross_correlation_df.index.to_series().apply(_td_to_sec))
    if separate_plots:
        for colname in df_sec.columns:
            # Information about plotted cross-correlation
            col = df_sec[colname]
            argmax = np.nanargmax(np.abs(col.values))
            maxval, maxtime = col.values[argmax], col.index[argmax]
            plt.figure(figsize=(15, 7))
            plt.plot(col, **mpl_args)
            plt.gca().axvline(maxtime, color="r")
            plt.suptitle(colname + "\n")
            plt.title("Max Correlation: {}, time lag: {}".format(str(maxval)[:6], str(_time_ticks(maxtime))))
            plt.gcf().set_facecolor("white")
            formatter = ticker.FuncFormatter(_time_ticks)
            plt.gca().xaxis.set_major_formatter(formatter)
            plt.xticks(rotation=60)
            plt.show(block=False)
    else:
        df_sec.plot(figsize=(20, 10), **mpl_args)
        plt.gcf().set_facecolor("white")
        formatter = ticker.FuncFormatter(_time_ticks)
        plt.gca().xaxis.set_major_formatter(formatter)
        plt.xticks(rotation=60)
        plt.show(block=False)
