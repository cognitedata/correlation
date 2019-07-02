import cognite.correlation
import pandas as pd
import numpy as np

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
    corr = cognite.correlation.cross_correlate(df, 'y')
    assert corr['z'] == 0.04169206868197918


def test_correlation_sort():
    np.random.seed(0)
    df = no_na
    corr_sort = cognite.correlation.columns_by_correlation(df, 'y')
    assert corr_sort[0][0] == 'y'


def test_cross_correlate_nans():
    np.random.seed()
    df = with_na
    corr = cognite.correlation.cross_correlate(df, 'x')
    assert not corr.isna().any()


def test_correlation_sort_nans():
    np.random.seed(0)
    df = with_na
    corr_sort = cognite.correlation.columns_by_correlation(df, 'y')
    assert corr_sort[0][0] == 'y'
