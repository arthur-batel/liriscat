import logging
import warnings
from torch.utils import data
import itertools
import numpy as np
import torch
from torch import Tensor
import torch.jit
from typing import List
import itertools
from torch.utils.data import DataLoader



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
        self._query_seed = config['seed']
        self.device = config['device']

        self._users_id = df['user_id'].unique()  # Ids of the users in this dataset instance (after splitting)
        self._questions_id = df['item_id'].unique()  # Ids of the items in this dataset instance (after splitting)
        self._concepts_id = df[
            'dimension_id'].unique()  # Ids of the categorys in this dataset instance (after splitting)
        self._n_meta = self._metadata['min_nb_users_logs'] // 5

        assert max(self._users_id) < self.n_users, \
            f'Require item ids renumbered : max user id = {max(self._users_id)}; nb users = {self.n_users}'
        assert max(self._questions_id) < self.n_questions, \
            f'Require item ids renumbered : max item id = {max(self._questions_id)}; nb items = {self.n_questions}'
        assert max(self._concepts_id) < self.n_categories, \
            f'Require concept ids renumbered : max concept id = {max(self._concepts_id)}; nb categories= {self.n_categories}'
        assert self._n_meta <= self._metadata['min_nb_users_logs'] - self.config["n_query"], \
            f'Some users have not enough logs to perform to submit {self.config["n_query"]} questions: min number of user logs = {self._metadata['min_nb_users_logs']}'

        self._torch_array = torch.from_numpy(df.to_numpy()).to(device=self.device)
        self._log_tensor = self._generate_log_tensor()  # precompute right away
        self._user_dict, self._user_id2idx, self._user_idx2id = self._generate_user_dict()

    @property
    def n_users(self):
        """
        @return: Total number of users in the dataset (before splitting)
        """
        return self.metadata["num_user_id"]

    @property
    def n_actual_users(self):
        """
        @return: Number of users in the current dataset object (after splitting)
        """
        return len(self.users_id)

    @property
    def n_logs(self):
        """
        @return: Total number of users in the dataset (before splitting)
        """
        return self._n_logs

    @property
    def n_questions(self):
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
        return self._questions_id

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
    def n_meta(self):
        """
        @return: Total number of users in the dataset (before splitting)
        """
        return self._n_meta

    @property
    def concept_map(self):
        return self._concept_map

    @property
    def metadata(self):
        return self._metadata

    @property
    def query_seed(self):
        return self._query_seed

    def set_query_seed(self, seed):
        self._query_seed = seed

    @property
    def df(self):
        return self._df

    @property
    def user_id2idx(self):
        return self._user_id2idx

    @property
    def user_idx2id(self):
        return self._user_idx2id

    @property
    def torch_array(self):
        warnings.warn("The torch_array contains both query and the meta set !")
        return self._torch_array

    @property
    def log_tensor(self):
        warnings.warn("The log_tensor contains both query and the meta set !")
        return self._log_tensor

    @property
    def user_dict(self):
        warnings.warn("The user_dict contains both query and the meta set !")
        return self._user_dict

    def _generate_log_tensor(self):
        tensor_data = torch.zeros((self.n_users, self.n_questions), device=self.device)

        sid = self.torch_array[:, 0].int()
        qid = self.torch_array[:, 1].int()
        val = self.torch_array[:, 3].float()

        tensor_data.index_put_((sid, qid), val)
        return tensor_data

    def _generate_user_dict(self) -> None:
        _user_dict = {}
        _user_id2idx = {}
        _user_idx2id = {}

        idx = 0
        for u, row in enumerate(self.log_tensor):
            q_ids = torch.where(row != 0)[0]
            labels = row[q_ids]
            if len(q_ids) > 0 :
                _user_dict[idx] = {'q_ids': q_ids, 'labels': labels}
                _user_id2idx[u] = idx
                _user_idx2id[idx] = u
                idx += 1

        return _user_dict, _user_id2idx, _user_idx2id


class CATDataset(Dataset, data.dataset.Dataset, data.DataLoader):

    def __init__(self, data, concept_map, metadata, config, batch_size, shuffle=True, pin_memory=True):
        """
        Args:
            data: list, [(sid, qid, score)]
            concept_map: dict, concept map {qid: cid}
            metadata : dict of keys {"num_user_id", "num_item_id", "num_dimension_id"}, containing the total number of users, items and concepts
        """
        Dataset.__init__(self,data, concept_map, metadata, config)
        DataLoader.__init__(self, dataset=self, collate_fn=CustomCollate(self), batch_size=batch_size, shuffle=shuffle,
                         pin_memory=pin_memory)

        self._qconcept_len_cache = {}
        self._qconcept_list_cache = {}

        for qid, concepts in self.concept_map.items():

            self._qconcept_len_cache[qid] = len(concepts)

            # Save the concept list. (Remains on CPU as a Python list,
            # or you could turn it into a small GPU tensor if you prefer.)
            self._qconcept_list_cache[qid] = concepts

        self.rng = np.random.default_rng(self.query_seed)

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.user_dict)

    def generate_sample(self, index):
        data = self.user_dict[index]

        observed_index = (self.rng.permutation(data['q_ids'].shape[0]) + index) % data['q_ids'].shape[0]
        meta_index = observed_index[-self.n_meta:]
        query_index = observed_index[:self.config['n_query']]

        query_concepts_nb = torch.tensor(
            [self._qconcept_len_cache[q.item()] for q in data['q_ids'][query_index]],
            device=self.device
        )
        qc = [
            self._qconcept_list_cache[q.item()]  # returns a Python list
            for q in data['q_ids'][query_index]
        ]

        meta_concepts_nb = torch.tensor(
            [self._qconcept_len_cache[q.item()] for q in data['q_ids'][meta_index]],
            device=self.device
        )

        mc = list(
            itertools.chain.from_iterable(
                self._qconcept_list_cache[q.item()] for q in data['q_ids'][meta_index]
            )
        )



        result = (self.user_idx2id[index],  # int
                  data['q_ids'][query_index],  # torch.Tensor
                  data['labels'][query_index],  # torch.Tensor
                  qc,  # list of lists
                  query_concepts_nb,  # torch.Tensor

                  data['q_ids'][meta_index],  # torch.Tensor
                  data['labels'][meta_index],  # torch.Tensor
                  mc,  # list of ints
                  meta_concepts_nb,  # torch.Tensor
                  )
        return result

    def __getitem__(self, index):
        # return the data of the user reindexed in this dataset instance
        # Warning : index is not the user id but the index of the user in the dataset!
        'Generates one sample of data'


        return self.generate_sample(index)

class evalDataset(CATDataset):

    def __init__(self, data, concept_map, metadata, config, batch_size, shuffle=False, pin_memory=True):
        """
        Args:
            data: list, [(sid, qid, score)]
            concept_map: dict, concept map {qid: cid}
            metadata : dict of keys {"num_user_id", "num_item_id", "num_dimension_id"}, containing the total number of users, items and concepts
        """
        super().__init__(data, concept_map, metadata, config, batch_size, shuffle=shuffle, pin_memory=pin_memory)
        self._meta_mask = torch.zeros_like(self.log_tensor, device=self.device, dtype=torch.bool)
        self._precomputed_batch = {}

    @property
    def meta_mask(self):
        return self._meta_mask

    @property
    def meta_tensor(self):
        return self.log_tensor * self.meta_mask

    def split_query_meta(self, query_seed):

        self.set_query_seed(query_seed)
        self._meta_mask = torch.zeros_like(self.log_tensor, device=self.device, dtype=torch.bool)

        for index in range(len(self)):
            sample_tuple = self.generate_sample(index)

            self._meta_mask.index_put_(((self.user_idx2id[index]*torch.ones_like(sample_tuple[5], dtype=torch.long)), sample_tuple[5]), torch.ones_like(sample_tuple[5], dtype=torch.bool))
            self._precomputed_batch[index] = sample_tuple

    def __getitem__(self, index):
        # return the data of the user reindexed in this dataset instance
        # Warning : index is not the user id but the index of the user in the dataset!

        return self._precomputed_batch[index]

class CustomCollate(object):
    def __init__(self, data: CATDataset):
        self.data = data

    def __call__(self, batch):

        env = EnvModule(len(batch), self.data.n_questions, self.data.n_meta, self.data.device)

        # Lists of lists to store the categories associated to each question and meta question
        qc_list = []
        mc_list = []

        for i, (u, qq, ql, qc, qc_nb, mq, ml, mc, mc_nb) in enumerate(batch) :

            ### ----- Query questions
            # Number of query question for the current user
            N = qq.shape[0]
            env.query_len[i] = N
            env.query_users[i] = u

            # Padding with zeros on the right
            env.query_meta_indices[i,N:] = 0

            # Saving logs questions, responses and number of categories
            env.query_questions[i,:N] = qq
            env.query_responses[i, :N] = ql
            env.query_cat_nb[i, :N] = qc_nb

            # Saving the categories associated to each question
            qc_list.append(qc)

            ### ----- Meta questions
            # Saving meta users, questions, responses and number of categories

            env.meta_users[i] = u*torch.ones_like(mq)
            env.meta_questions[i] = mq
            env.meta_responses[i] = ml
            env.meta_cat_nb[i] = mc_nb

            # Saving the categories associated to each meta question
            mc_list.extend(mc)

        # Convert the lists of lists to tensors because meta questions are not varying
        env.meta_categories = torch.tensor(mc_list, device=self.data.device).int()
        env.query_cat_list = qc_list

        return env

class EnvModule(torch.jit.ScriptModule):
    # Declare attributes as class fields with type annotations
    query_meta_indices: Tensor
    row_idx: Tensor
    query_len: Tensor
    query_users: Tensor
    query_questions: Tensor
    query_responses: Tensor
    query_cat_nb: Tensor
    query_cat_list: List[List[List[int]]]
    query_user_ids: Tensor
    query_question_ids: Tensor
    query_labels: Tensor
    query_category_ids: Tensor
    meta_users: Tensor
    meta_questions: Tensor
    meta_responses: Tensor
    meta_cat_nb: Tensor
    meta_categories: Tensor
    device: torch.device

    def __init__(self, batch_size: int, n_questions: int, n_meta: int, device: torch.device):
        super(EnvModule, self).__init__()

        # initialize state variables
        self.query_meta_indices = torch.arange(0, n_questions, device=device, dtype=torch.long).repeat(batch_size, 1)
        self.row_idx = torch.arange(self.query_meta_indices.size(0))

        # Initialize attributes with proper tensors
        self.query_len = torch.zeros(batch_size, dtype=torch.long, device=device)
        self.query_users = torch.zeros(batch_size, dtype=torch.long, device=device)
        self.query_questions = torch.zeros(batch_size, n_questions, dtype=torch.long, device=device)
        self.query_responses = torch.zeros(batch_size, n_questions, dtype=torch.long, device=device)
        self.query_cat_nb = torch.zeros(batch_size, n_questions, dtype=torch.long, device=device)

        self.query_cat_list = []
        self.query_user_ids = torch.tensor([], dtype=torch.long, device=device)
        self.query_question_ids = torch.tensor([], dtype=torch.long, device=device)
        self.query_labels = torch.tensor([], dtype=torch.long, device=device)
        self.query_category_ids = torch.tensor([], dtype=torch.long, device=device)

        self.meta_users = torch.zeros(batch_size, n_meta, dtype=torch.long, device=device)
        self.meta_questions = torch.zeros(batch_size, n_meta, dtype=torch.long, device=device)
        self.meta_responses = torch.zeros(batch_size, n_meta, dtype=torch.long, device=device)
        self.meta_cat_nb = torch.zeros(batch_size, n_meta, dtype=torch.long, device=device)
        self.meta_categories = torch.tensor([], dtype=torch.long, device=device)

        self.device = device

    @property
    def n_meta_logs(self):
        """
        @return: Total number of meta logs (including logs multiplied because of several categories
        """
        return self.meta_cat_nb.sum().item()

    @torch.jit.script_method
    def update(self, actions: Tensor, t: int) -> None:
        indices: List[int] = self.query_meta_indices[self.row_idx, actions].tolist()

        qc_action_list: List[List[int]] = [outer[idx] for outer, idx in zip(self.query_cat_list, indices)]
        flat_qc: List[int] = self.flatten_list(qc_action_list)
        QC = torch.tensor(flat_qc, device=self.device).int()

        # Assuming feedIMPACT_query is TorchScript-compatible or is wrapped properly
        new_user_ids, new_question_ids, new_labels, new_category_ids = self.feedIMPACT_query(
            self.query_questions[self.row_idx, self.query_meta_indices[self.row_idx, actions]],
            self.query_responses[self.row_idx, self.query_meta_indices[self.row_idx, actions]],
            self.query_cat_nb[self.row_idx, self.query_meta_indices[self.row_idx, actions]],
            self.query_users, QC
        )

        tmp = self.query_meta_indices[self.row_idx, t]
        self.query_meta_indices[self.row_idx, t] = self.query_meta_indices[self.row_idx, actions]
        self.query_meta_indices[self.row_idx, actions] = tmp

        # Add the new data to the set of submitted questions
        self.query_user_ids = torch.cat((self.query_user_ids, new_user_ids), dim=0)
        self.query_question_ids = torch.cat((self.query_question_ids, new_question_ids), dim=0)
        self.query_labels = torch.cat((self.query_labels, new_labels), dim=0)
        self.query_category_ids = torch.cat((self.query_category_ids, new_category_ids), dim=0)

    @torch.jit.script_method
    def feedIMPACT_query(self, QQ, QL, QC_NB, U, QC):

        question_ids = torch.repeat_interleave(QQ, QC_NB)
        user_ids = torch.repeat_interleave(U, QC_NB)
        labels = torch.repeat_interleave(QL, QC_NB)

        return user_ids, question_ids, labels, QC

    @torch.jit.script_method
    def feedIMPACT_meta(self):
        MQ = self.meta_questions.reshape(-1)
        ML = self.meta_responses.reshape(-1)
        MC_NB = self.meta_cat_nb.reshape(-1)
        MU = self.meta_users.reshape(-1)

        question_ids = torch.repeat_interleave(MQ, MC_NB)
        user_ids = torch.repeat_interleave(MU, MC_NB)
        labels = torch.repeat_interleave(ML, MC_NB)
        category_ids = self.meta_categories

        return user_ids, question_ids, labels, category_ids

    @torch.jit.ignore
    def flatten_list(self,qc_action_list: List[List[int]]) -> List[int]:
        flat: List[int] = []
        for sublist in qc_action_list:
                flat.extend(sublist)
        return flat





