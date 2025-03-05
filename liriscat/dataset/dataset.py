import logging
from collections import defaultdict, deque

import torch
from torch.utils import data
import numpy as np
import pandas as pd

class Dataset(object):

    def __init__(self, df, concept_map, metadata, config):
        """
        Args:
            df: Dataframe with columns (user_id, item_id, correct, dimension_id)
            concept_map: dict, concept map {qid: cid}
            metadata : dict of keys {"num_user_id", "num_item_id", "num_dimension_id"}, containing the total number of users, items and concepts
        """
        self._metadata = metadata
        self._concept_map = concept_map
        self._df = df
        self._n_logs = self._df.shape[0]
        self._config = config

        self._np_array = df.to_numpy()
        self._torch_array = torch.from_numpy(self._np_array).to(device=config['device'])
        self._log_tensor = self._generate_log_tensor()  # precompute right away

        self._users_id = df['user_id'].unique() # Ids of the users in this dataset instance (after splitting)
        self._items_id = df['item_id'].unique() # Ids of the items in this dataset instance (after splitting)
        self._concepts_id = df['dimension_id'].unique() # Ids of the categorys in this dataset instance (after splitting)

        assert max(self._users_id) < self.n_users, \
            f'Require item ids renumbered : max user id = {max(self._users_id)}; nb users = {self.n_users}'
        assert max(self._items_id) < self.n_items, \
            f'Require item ids renumbered : max item id = {max(self._items_id)}; nb items = {self.n_items}'
        assert max(self._concepts_id) < self.n_categories, \
            f'Require concept ids renumbered : max concept id = {max(self._concepts_id)}; nb concepts = {self.n_categories}'

    @property
    def n_users(self):
        """
        @return: Total number of users in the dataset (before splitting)
        """
        return self.metadata["num_user_id"]

    @property
    def n_logs(self):
        """
        @return: Total number of users in the dataset (before splitting)
        """
        return self._n_logs

    @property
    def n_items(self):
        """
        @return: Total number of items in the dataset (before splitting)
        """
        return self.metadata["num_item_id"]

    @property
    def n_categories(self):
        """
        @return: Total number of categories
        """
        return self.metadata["num_dimension_id"]

    @property
    def users_id(self):
        """
        @return: Ids of the users in this dataset instance (after splitting)
        """
        return self._users_id

    @property
    def items_id(self):
        """
        @return: Ids of the items in this dataset instance (after splitting)
        """
        return self._items_id

    @property
    def concepts_id(self):
        """
        @return: Ids of the items in this dataset instance (after splitting)
        """
        return self._concepts_id

    @property
    def config(self):
        """
        @return: Total number of users in the dataset (before splitting)
        """
        return self._config

    @property
    def concept_map(self):
        return self._concept_map

    @property
    def metadata(self):
        return self._metadata

    @property
    def df(self):
        return self._df

    @property
    def np_array(self):
        raise Exception("The np_array contains both query and the meta set !")
        return self._np_array

    @property
    def torch_array(self):
        raise Exception("The torch_array contains both query and the meta set !")
        return self._torch_array

    @property
    def log_tensor(self):
        raise Exception("The log_tensor contains both query and the meta set !")
        return self._log_tensor

    def _generate_log_tensor(self):
        tensor_data = torch.zeros((self.n_users, self.n_items), device=self.config['device'])

        sid = self.torch_array[:, 0].int()
        qid = self.torch_array[:, 1].int()
        val = self.torch_array[:, 3].float()

        tensor_data.index_put_((sid, qid), val)
        return tensor_data

class LoaderDataset(Dataset, data.dataset.Dataset):

    def __init__(self, data, concept_map, metadata, config):
        """
        Args:
            data: list, [(sid, qid, score)]
            concept_map: dict, concept map {qid: cid}
            metadata : dict of keys {"num_user_id", "num_item_id", "num_dimension_id"}, containing the total number of users, items and concepts
        """
        super().__init__(data, concept_map, metadata, config)

    def __getitem__(self, item):
        return self.raw_data_array[item]

    def __len__(self):
        return self.n_logs
