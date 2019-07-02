from cognite.client import CogniteClient


def test_with_cdf():
    client = CogniteClient(client_name='correlation-testing')
    ts_info = client.time_series.list(limit=2).to_pandas()
    print(ts_info.columns)
    weeks = 1000
    tss = client.datapoints.retrieve_dataframe(start='{}w-ago'.format(weeks), end='now',
                                               aggregates=['average'], granularity='1d',
                                               id=ts_info['id'].to_list())
    print(tss)
