from cognite.client import CogniteClient
from typing import *


class DataLoader:
    def __init__(self, client: CogniteClient):
        """
        Class for loading data for use in correlation calculation
        :param client: CogniteClient with API-key and project name specified
        """
        self.client: CogniteClient = client

    def find_time_series(self, include_metadata: bool = True, asset_ids: Optional[List[int]] = None, limit: int = 25):
        return self.client.time_series.list(include_metadata, asset_ids, limit)

    def load_df(self, asset_ids: List[int]):
        """
        Load time series for all assets in list
        :param asset_ids: List of asset ids
        :return: Pandas DataFrame with time series for all assets
        """
        return self.client.time_series.retrieve_multiple(asset_ids).to_pandas()

