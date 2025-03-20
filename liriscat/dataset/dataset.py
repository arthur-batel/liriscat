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

        self._cat_tensor, self._cat_mask, self._cat_nb = self._generate_qc_tensor()

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
    def n_query(self):
        """
        @return: Total number of users in the dataset (before splitting)
        """
        return self.config["n_query"]

    @property
    def cat_tensor(self):
        return self._cat_tensor

    @property
    def cat_mask(self):
        return self._cat_mask

    @property
    def cat_nb(self):
        return self._cat_nb

    @property
    def concept_map(self):
        return self._concept_map

    @property
    def cq_max(self):
        "Maximum Number of categories per question in the dataset"
        return self.metadata['max_nb_categories_per_question']

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

    def _generate_qc_tensor(self):
        size = (self.n_questions, self.cq_max)
        _cat_tensor = torch.full(size, -1,dtype=torch.long, device=self.device)
        _cat_len = torch.empty(self.n_questions, dtype=torch.long, device=self.device)
        for qid, concepts in self.concept_map.items():
            _cat_tensor[qid, :len(concepts)] = torch.tensor(concepts, device=self.device)
            _cat_len[qid] = len(concepts)
        _cat_mask = torch.where(_cat_tensor == -1, torch.zeros(size, dtype=torch.bool, device=self.device),
                                    torch.ones(size, dtype=torch.bool, device=self.device))
        return _cat_tensor, _cat_mask, _cat_len


class CATDataset(Dataset, data.dataset.Dataset, data.DataLoader):
    """
    Train dataset
    """

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

        self.rng = np.random.default_rng(self.query_seed)

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.user_dict)

    def generate_sample(self, index):
        data = self.user_dict[index]

        observed_index = (self.rng.permutation(data['q_ids'].shape[0]) + index) % data['q_ids'].shape[0]
        meta_index = observed_index[-self.n_meta:]
        query_index = observed_index[:self.config['n_query']]

        return (self.user_idx2id[index],  # int
                  data['q_ids'][query_index],  # 1D torch.Tensor, size = Q_u
                  data['labels'][query_index],  # 1D torch.Tensor, size = Q_u
                  self.cat_tensor[data['q_ids'][query_index]].flatten(),  # 1D torch.Tensor, size = Q_u x cq_max
                  self.cat_mask[data['q_ids'][query_index]].flatten(),  # 1D torch.Tensor, size = Q_u x cq_max
                  self.cat_nb[data['q_ids'][query_index]],  # 1D torch.Tensor, size = Q_u

                  data['q_ids'][meta_index],  # 1D torch.Tensor, size = M
                  data['labels'][meta_index],  # 1D torch.Tensor, size = M
                  self.cat_tensor[data['q_ids'][meta_index]].flatten(),  # 1D torch.Tensor, size = M x cq_max
                  self.cat_mask[data['q_ids'][meta_index]].flatten(), # 1D torch.Tensor, size = M x cq_max
                  self.cat_nb[data['q_ids'][meta_index]])  # 1D torch.Tensor, size = M


    def __getitem__(self, index):
        # return the data of the user reindexed in this dataset instance
        # Warning : index is not the user id but the index of the user in the dataset!
        'Generates one sample of data'


        return self.generate_sample(index)

class evalDataset(CATDataset):
    """
    valid and test dataset
    """

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
        """
        Split the dataset into a query and a meta set
        """

        self.set_query_seed(query_seed)
        self._meta_mask = torch.zeros_like(self.log_tensor, device=self.device, dtype=torch.bool)

        for index in range(len(self)):
            sample_tuple = self.generate_sample(index)

            self._meta_mask.index_put_(
                (torch.full(sample_tuple[5].shape, self.user_idx2id[index], dtype=torch.long, device=self.device),
                 sample_tuple[5]),
                torch.ones_like(sample_tuple[5], dtype=torch.bool)
            )
            self._precomputed_batch[index] = sample_tuple

    def __getitem__(self, index):
        # return the data of the user reindexed in this dataset instance
        # Warning : index is not the user id but the index of the user in the dataset!

        return self._precomputed_batch[index]

class CustomCollate(object):
    def __init__(self, data: CATDataset):
        self.data = data

    def __call__(self, batch):
        """
        Collate users data into a batch
        """

        env = EnvModule(len(batch), self.data, self.data.device)

        for i, (u, qq, ql, qc, qc_mask, qc_nb, mq, ml, mc, mc_mask, mc_nb) in enumerate(batch) :

            ### ----- Query questions
            # Number of query question for the current user
            N = qq.shape[0]
            env.query_len[i] = N
            env.query_users[i] = u

            # Padding with zeros on the right
            env.query_indices[i, N:] = 0

            # Saving logs questions, responses and number of categories
            env.query_questions[i,:N] = qq
            env.query_responses[i, :N] = ql
            env.query_cat_nb[i, :N] = qc_nb

            # Saving the categories associated to each question and there mask
            env.query_cat[i, :qc.shape[0]] = qc
            env.query_cat_mask[i, :qc.shape[0]] = qc_mask

            ### ----- Meta questions
            # Saving meta users, questions, responses and number of categories
            env.meta_users[i] = torch.full(mq.shape, u, dtype=torch.long, device=self.data.device)
            env.meta_questions[i] = mq
            env.meta_responses[i] = ml
            env.meta_cat_nb[i] = mc_nb

            # Saving the categories associated to each question and there mask
            env.meta_cat[i, :mc.shape[0]] = mc
            env.meta_cat_mask[i, :mc.shape[0]] = mc_mask

        return env

class EnvModule:
    """
    Class managing the Query set, Set of submitted questions and the Meta set (prealocation, storage, update)
    """

    def __init__(self, batch_size: int, data : CATDataset, device: torch.device):

        n_query = data.config['n_query']
        max_nb_cat_per_question = data.metadata['max_nb_categories_per_question']
        data_batch_size = n_query * max_nb_cat_per_question * batch_size

        # initialize state variables
        ## Tensor storing the indices of the questions submitted by the user
        self.query_indices = torch.arange(0, data.n_query, device=device, dtype=torch.long).repeat(batch_size, 1) # tensor of size (batch_size, n_questions)
        self.row_idx = torch.arange(self.query_indices.size(0)) # tensor of size (batch_size)

        ## Index of the current number of submitted logs
        self.current_idx = 0

        # Initialize attributes with proper tensors
        self.query_len = torch.empty(batch_size, dtype=torch.long, device=device)
        self.query_users = torch.empty(batch_size, dtype=torch.long, device=device)
        self.query_questions = torch.empty(batch_size, data.n_query, dtype=torch.long, device=device)
        self.query_responses = torch.empty(batch_size, data.n_query, dtype=torch.long, device=device)
        self.query_cat_nb = torch.empty(batch_size, data.n_query, dtype=torch.long, device=device)
        self.query_cat = torch.empty(batch_size, data.n_query*data.cq_max, dtype=torch.long, device=device)
        self.query_cat_mask = torch.empty(batch_size, data.n_query*data.cq_max, dtype=torch.bool, device=device)

        self.query_user_ids = torch.empty(data_batch_size, dtype=torch.long, device=device)
        self.query_question_ids = torch.empty(data_batch_size, dtype=torch.long, device=device)
        self.query_labels = torch.empty(data_batch_size, dtype=torch.long, device=device)
        self.query_category_ids = torch.empty(data_batch_size, dtype=torch.long, device=device)

        self.meta_users = torch.empty(batch_size, data.n_meta, dtype=torch.long, device=device)
        self.meta_questions = torch.empty(batch_size, data.n_meta, dtype=torch.long, device=device)
        self.meta_responses = torch.empty(batch_size, data.n_meta, dtype=torch.long, device=device)
        self.meta_cat_nb = torch.empty(batch_size, data.n_meta, dtype=torch.long, device=device)
        self.meta_cat = torch.empty(batch_size, data.n_meta*data.cq_max, dtype=torch.long, device=device)
        self.meta_cat_mask = torch.empty(batch_size, data.n_meta*data.cq_max, dtype=torch.bool, device=device)

        self.device = device
        self.cq_max = data.cq_max

    @property
    def n_meta_logs(self):
        """
        @return: Total number of meta logs (including logs multiplied because of several categories
        """
        return self.meta_cat_nb.sum().item()

    def update(self, actions: Tensor, t: int) -> None:
        with torch.no_grad():
            idx = self.current_idx

            new_user_ids, new_question_ids, new_labels = self.generate_IMPACT_query(
                self.query_questions[self.row_idx, self.query_indices[self.row_idx, actions]],
                self.query_responses[self.row_idx, self.query_indices[self.row_idx, actions]],
                self.query_cat_nb[self.row_idx, self.query_indices[self.row_idx, actions]],
                self.query_users)

            start = self.query_indices[self.row_idx, actions] * self.cq_max # (shape: [batch_size])
            offset = torch.arange(self.cq_max, device=self.device).unsqueeze(0) # (shape: [1, cq_max])
            indices = start.unsqueeze(1) + offset # (shape: [batch_size, cq_max])
            new_category_ids = self.query_cat.gather(1, indices)[self.query_cat_mask.gather(1, indices)]

            # Update the tensor of query indices
            tmp = self.query_indices[self.row_idx, t]
            self.query_indices[self.row_idx, t] = self.query_indices[self.row_idx, actions]
            self.query_indices[self.row_idx, actions] = tmp

            # increment the number of submitted logs
            self.current_idx += new_user_ids.shape[0]

            # Add the new data to the set of submitted questions
            self.query_user_ids[idx:self.current_idx] =  new_user_ids
            self.query_question_ids[idx:self.current_idx] =  new_question_ids
            self.query_labels[idx:self.current_idx] = new_labels
            self.query_category_ids[idx:self.current_idx] = new_category_ids


    def feed_IMPACT_query(self):
        return {
            "user_ids":self.query_user_ids[:self.current_idx],
            "question_ids":self.query_question_ids[:self.current_idx],
            "labels":self.query_labels[:self.current_idx],
            "category_ids":self.query_category_ids[:self.current_idx]}


    def generate_IMPACT_query(self, QQ, QL, QC_NB, U):

        question_ids = torch.repeat_interleave(QQ, QC_NB)
        user_ids = torch.repeat_interleave(U, QC_NB)
        labels = torch.repeat_interleave(QL, QC_NB)

        return user_ids, question_ids, labels

    def generate_IMPACT_meta(self):
        MQ = self.meta_questions.reshape(-1)
        ML = self.meta_responses.reshape(-1)
        MC_NB = self.meta_cat_nb.reshape(-1)
        MU = self.meta_users.reshape(-1)

        question_ids = torch.repeat_interleave(MQ, MC_NB)
        user_ids = torch.repeat_interleave(MU, MC_NB)
        labels = torch.repeat_interleave(ML, MC_NB)
        category_ids = self.meta_cat[self.meta_cat_mask]

        return user_ids, question_ids, labels, category_ids

    def flatten_list(self,qc_action_list: List[List[int]]) -> List[int]:
        flat: List[int] = []
        for sublist in qc_action_list:
                flat.extend(sublist)
        return flat

class EnvQueryDataset(Dataset):
    """
    Bridge class transforming the set of submitted question into a torch.utils.data.Dataset
    """
    def __init__(self, query_data):
        # query_data is a dictionary with tensors: user_ids, question_ids, labels, category_ids
        self.user_ids = query_data["user_ids"]
        self.question_ids = query_data["question_ids"]
        self.labels = query_data["labels"]
        self.category_ids = query_data["category_ids"]
        self.length = self.user_ids.shape[0]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return {
            "user_ids": self.user_ids[idx],
            "question_ids": self.question_ids[idx],
            "labels": self.labels[idx],
            "category_ids": self.category_ids[idx],
        }





