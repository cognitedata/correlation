import cognite.correlation
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

np.random.seed(0)
no_na = pd.DataFrame({
    'x': range(10),
    'y': (*range(10, 1, -1), 3),
    'z': np.random.randint(0, 100, 10)
})

np.random.seed(0)
with_na = pd.DataFrame({
    'x': range(10),
    'y': (*range(10, 1, -1), 3),
    'z': np.random.randint(0, 100, 10),
    'w': np.array([*np.arange(6), np.NaN, *np.arange(3)])
})


def test_cross_correlate():
    np.random.seed(0)
    df = no_na
    corr = cognite.correlation.cross_correlate(df, df['y'])
    assert corr['z'] == 0.04169206868197918


def test_correlation_sort():
    np.random.seed(0)
    df = no_na
    corr_sort = cognite.correlation.columns_by_correlation(df, 'y')
    assert corr_sort[0][0] == 'y'


def test_cross_correlate_nans():
    np.random.seed()
    df = with_na
    corr = cognite.correlation.cross_correlate(df, df['x'])
    assert not corr.isna().any()


def test_correlation_sort_nans():
    np.random.seed(0)
    df = with_na
    corr_sort = cognite.correlation.columns_by_correlation(df, 'y')
    assert corr_sort[0][0] == 'y'


def test_max_cross_correlation():
    return
    drange = pd.date_range(start=datetime(2017, 1, 1), end=datetime(2018, 1, 1), periods=500)
    ddiff = drange - datetime()
    df = pd.DataFrame({
        'datetime': drange,
        'x': np.sin(2 * np.pi * (drange / timedelta(days=60))) + np.random.rand(500) * 0.1,
        'y': np.sin(2 * np.pi * (drange + timedelta(days=30)) / timedelta(days=60)),  # Response after x
        'z': np.sin(2 * np.pi * (drange - timedelta(days=30)) / timedelta(days=60)) + np.random.rand(500) * 0.3,
    })
    print(df.head())
    df.set_index('datetime', inplace=True)
    # Want to find a cause, will only look back in time
    lags = pd.timedelta_range(start=timedelta(days=-50), end=timedelta(days=50), periods=101)
    corr_info = cognite.correlation.columns_by_max_cross_correlation(df, 'y', lags)
    print(corr_info)
    assert np.round(corr_info['corr'][0], decimals=7) == 1
    assert corr_info['lag'][0] == timedelta(0)
