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
    df = pd.DataFrame({
        'timestamp': np.arange(0, 500),
        'x': np.sin(np.linspace(0, 8*np.pi, num=500)) + np.random.rand(500) * 0.1,
        'y': np.sin(np.linspace(0 + 0.4, 8*np.pi + 0.4, num=500)),  # Response after x
        'z': np.sin(np.linspace(0 + 0.2, 8*np.pi + 0.2, num=500)) + np.random.rand(500) * 0.3,
    })
    # Want to find a cause, will only look back in time
    lags = np.linspace(-100, 99, num=100)
    print(lags)
    print('\n', df.head())
    print(cognite.correlation.columns_by_max_cross_correlation(df, 'y', lags))
