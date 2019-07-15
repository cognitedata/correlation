from datetime import timedelta

import numpy as np
import pandas as pd

import cognite.correlation.plot as cplot

n = 1000
ts = np.linspace(0, 2 * np.pi, n)
df = pd.DataFrame({"x": np.sin(ts) ** 2, "y": np.exp(-ts)})
df.set_index(pd.timedelta_range(timedelta(minutes=-60), timedelta(), periods=n), inplace=True)


def test_plot():
    cplot.plot_cross_correlations(df, separate_plots=False)


def test_plots_separate():
    cplot.plot_cross_correlations(df, mpl_args={"marker": "x"})


def test_plots_select_tag():
    cplot.plot_cross_correlations(df, cols_to_plot=["x"])


def test_plots_select_index():
    cplot.plot_cross_correlations(df, cols_to_plot=[1])
