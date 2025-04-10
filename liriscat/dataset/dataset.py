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
from IMPACT.dataset import Dataset as IMPACTDataset


class Dataset(IMPACTDataset):

    def __init__(self, df, concept_map, metadata, config):
        """
        Args:
            df: Dataframe with columns (user_id, item_id, correct, dimension_id)
            concept_map: dict, concept map {qid: cid}
            metadata : dict of keys {"num_user_id", "num_item_id", "num_dimension_id"}, containing the total number of users, items and concepts
        """

        super().__init__(df.to_records(index=False, column_dtypes={'user_id': int, 'item_id': int, "dimension_id": int,
                                                                   "correct": float}), concept_map, metadata)

        self._df = df
        self._config = config
        self._query_seed = config['seed']
        self.device = config['device']

        self._questions_id = self.items_id

        self._n_meta = self._metadata['min_nb_users_logs'] // 5

        assert max(self._users_id) < self.n_users, \
            f'Require item ids renumbered : max user id = {max(self._users_id)}; nb users = {self.n_users}'
        assert max(self._questions_id) < self.n_questions, \
            f'Require item ids renumbered : max item id = {max(self._questions_id)}; nb items = {self.n_questions}'
        assert max(self._concepts_id) < self.n_categories, \
            f'Require concept ids renumbered : max concept id = {max(self._concepts_id)}; nb categories= {self.n_categories}'
        assert self._n_meta <= self._metadata['min_nb_users_logs'] - self.config["n_query"], \
            f'Some users have not enough logs to perform to submit {self.config["n_query"]} questions: min number of user logs = {self._metadata["min_nb_users_logs"]}'

        self._torch_array = torch.from_numpy(df.to_numpy()).to(device=self.device)
        self._log_tensor = self._generate_log_tensor()  # precompute right away
        self._user_dict, self._user_id2idx, self._user_idx2id = self._generate_user_dict()
        self._cat_tensor, self._cat_mask, self._cat_nb = self._generate_qc_tensor()

    @property
    def n_actual_users(self):
        """
        @return: Number of users in the current dataset object (after splitting)
        """
        return len(self.users_id)

    @property
    def n_questions(self):
        """
        @return: Total number of items in the dataset (before splitting)
        """
        return self.metadata["num_item_id"]

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
    def cq_max(self):
        "Maximum Number of categories per question in the dataset"
        return self.metadata['max_nb_categories_per_question']

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
        return self._torch_array

    @property
    def user_dict(self):
        return self._user_dict

    def _generate_user_dict(self):
        _user_dict = {}
        _user_id2idx = {}
        _user_idx2id = {}

        idx = 0
        for u, row in enumerate(self.log_tensor):
            q_ids = torch.where(row != 0)[0]
            labels = row[q_ids]
            if len(q_ids) > 0:
                _user_dict[idx] = {'q_ids': q_ids, 'labels': labels}
                _user_id2idx[u] = idx
                _user_idx2id[idx] = u
                idx += 1

        return _user_dict, _user_id2idx, _user_idx2id

    def _generate_qc_tensor(self):
        size = (self.n_questions, self.cq_max)
        _cat_tensor = torch.full(size, -1, dtype=torch.long, device=self.device)
        _cat_len = torch.empty(self.n_questions, dtype=torch.long, device=self.device)
        for qid, concepts in self.concept_map.items():
            _cat_tensor[qid, :len(concepts)] = torch.tensor(concepts, device=self.device)
            _cat_len[qid] = len(concepts)
        _cat_mask = torch.where(_cat_tensor == -1, torch.zeros(size, dtype=torch.bool, device=self.device),
                                torch.ones(size, dtype=torch.bool, device=self.device))
        return _cat_tensor, _cat_mask, _cat_len


class CATDataset(Dataset, data.dataset.Dataset):
    """
    Train dataset
    """

    def __init__(self, data, concept_map, metadata, config):
        """
        Args:
            data: list, [(sid, qid, score)]
            concept_map: dict, concept map {qid: cid}
            metadata : dict of keys {"num_user_id", "num_item_id", "num_dimension_id"}, containing the total number of users, items and concepts
        """
        Dataset.__init__(self, data, concept_map, metadata, config)

        self.rng = np.random.default_rng(self.query_seed)

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.user_dict)

    def generate_sample(self, index):
        data = self.user_dict[index]

        observed_index = (self.rng.permutation(data['q_ids'].shape[0]) + index) % data['q_ids'].shape[0]
        meta_index = observed_index[-self.n_meta:]
        query_index = observed_index[:self.config['n_query']]

        return {'u_idx' :self.user_idx2id[index],  # int
                'qq':data['q_ids'][query_index],  # 1D torch.Tensor, size = Q_u
                'ql':data['labels'][query_index],  # 1D torch.Tensor, size = Q_u
                'qc':self.cat_tensor[data['q_ids'][query_index]].flatten(),  # 1D torch.Tensor, size = Q_u x cq_max
                'qc_mask':self.cat_mask[data['q_ids'][query_index]].flatten(),  # 1D torch.Tensor, size = Q_u x cq_max
                'qc_nb':self.cat_nb[data['q_ids'][query_index]],  # 1D torch.Tensor, size = Q_u

                'mq':data['q_ids'][meta_index],  # 1D torch.Tensor, size = M
                'ml':data['labels'][meta_index],  # 1D torch.Tensor, size = M
                'mc':self.cat_tensor[data['q_ids'][meta_index]].flatten(),  # 1D torch.Tensor, size = M x cq_max
                'mc_mask':self.cat_mask[data['q_ids'][meta_index]].flatten(),  # 1D torch.Tensor, size = M x cq_max
                'mc_nb':self.cat_nb[data['q_ids'][meta_index]]}  # 1D torch.Tensor, size = M

    def __getitem__(self, index):
        # return the data of the user reindexed in this dataset instance
        # Warning : index is not the user id but the index of the user in the dataset!
        'Generates one sample of data'

        return self.generate_sample(index)


class EvalDataset(CATDataset):
    """
    valid and test dataset
    """

    def __init__(self, data, concept_map, metadata, config):
        """
        Args:
            data: list, [(sid, qid, score)]
            concept_map: dict, concept map {qid: cid}
            metadata : dict of keys {"num_user_id", "num_item_id", "num_dimension_id"}, containing the total number of users, items and concepts
        """
        super().__init__(data, concept_map, metadata, config)
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
                (torch.full(sample_tuple['mq'].shape, self.user_idx2id[index], dtype=torch.long, device=self.device),
                 sample_tuple['mq']),
                torch.ones_like(sample_tuple['mq'], dtype=torch.bool)
            )
            self._precomputed_batch[index] = sample_tuple

    def __getitem__(self, index):
        # return the data of the user reindexed in this dataset instance
        # Warning : index is not the user id but the index of the user in the dataset!

        return self._precomputed_batch[index]


class QueryEnv:
    """
    QueryEnv manages the Query set, Set of submitted questions and the Meta set (prealocation, storage, update) :
        - store the data
        - save and update there membership to the three sets
        - transform the data to IMPACT compatible format
    The data of each batch of users overwrites the previous one to optimize GPU memory allocation
    """

    def __init__(self, data: CATDataset, device: torch.device, batch_size: int):
        self.n_query = data.config['n_query']
        self.device = device
        self.cq_max = data.cq_max # Max nb of categories per question

        max_nb_cat_per_question = data.metadata['max_nb_categories_per_question']
        max_data_batch_size = self.n_query * max_nb_cat_per_question * batch_size

        # Initialize attributes
        ## Variable storing the current batch size
        self._current_batch_size = batch_size

        ## Tensors storing all query data (submitted and unsubmitted)
        self._query_len = torch.empty(batch_size, dtype=torch.long, device=device)
        self._query_users_vec = torch.empty(batch_size, dtype=torch.long, device=device)
        self._query_questions = torch.empty(batch_size, data.n_query, dtype=torch.long, device=device)
        self._query_responses = torch.empty(batch_size, data.n_query, dtype=torch.long, device=device)
        self._query_cat_nb = torch.empty(batch_size, data.n_query, dtype=torch.long, device=device)
        self._query_cat = torch.empty(batch_size, data.n_query * data.cq_max, dtype=torch.long, device=device)
        self._query_cat_mask = torch.empty(batch_size, data.n_query * data.cq_max, dtype=torch.bool, device=device)

        ## Tensors storing submitted query data (after expansion of the logs due to multiple categories per question)
        self.sub_user_ids = torch.empty(max_data_batch_size, dtype=torch.long, device=device)
        self.sub_question_ids = torch.empty(max_data_batch_size, dtype=torch.long, device=device)
        self.sub_labels = torch.empty(max_data_batch_size, dtype=torch.long, device=device)
        self.sub_category_ids = torch.empty(max_data_batch_size, dtype=torch.long, device=device)

        ## Tensor storing the indices of the questions submitted to the user
        self._query_indices = torch.arange(0, data.n_query, device=device, dtype=torch.long).repeat(batch_size,
                                                                                                    1)  # tensor of size (batch_size, n_query)
        self._row_idx = torch.arange(batch_size)  # tensor of size (batch_size)

        ## Submitted state
        self._current_charged_log_nb = 0  # Index of the current number of submitted logs to the CDM (different from the number of submitted questions per users). Reinint at each user batch by UserCollate

        ## Tensors storing meta data
        self._meta_users = torch.empty(batch_size, data.n_meta, dtype=torch.long, device=device)
        self._meta_questions = torch.empty(batch_size, data.n_meta, dtype=torch.long, device=device)
        self._meta_responses = torch.empty(batch_size, data.n_meta, dtype=torch.long, device=device)
        self._meta_cat_nb = torch.empty(batch_size, data.n_meta, dtype=torch.long, device=device)
        self._meta_cat = torch.empty(batch_size, data.n_meta * data.cq_max, dtype=torch.long, device=device)
        self._meta_cat_mask = torch.empty(batch_size, data.n_meta * data.cq_max, dtype=torch.bool, device=device)

    @property
    def current_batch_size(self):
        return self._current_batch_size

    @property
    def n_meta_logs(self):
        """
        @return: Total number of meta logs (including logs multiplied because of several categories
        """
        return self._meta_cat_nb.sum().item()

    @property
    def meta_users(self):
        return self._meta_users[:self._current_batch_size, :]

    @property
    def meta_questions(self):
        return self._meta_questions[:self._current_batch_size, :]

    @property
    def meta_responses(self):
        return self._meta_responses[:self._current_batch_size, :]

    @property
    def meta_cat_nb(self):
        return self._meta_cat_nb[:self._current_batch_size, :]

    @property
    def meta_cat(self):
        return self._meta_cat[:self._current_batch_size, :]

    @property
    def meta_cat_mask(self):
        return self._meta_cat_mask[:self._current_batch_size, :]

    @property
    def query_len(self):
        return self._query_len[:self._current_batch_size]

    @property
    def query_users_vec(self):
        return self._query_users_vec[:self._current_batch_size]

    @property
    def query_questions(self):
        return self._query_questions[:self._current_batch_size, :]

    @property
    def query_responses(self):
        return self._query_responses[:self._current_batch_size, :]

    @property
    def query_cat_nb(self):
        return self._query_cat_nb[:self._current_batch_size, :]

    @property
    def query_cat(self):
        return self._query_cat[:self._current_batch_size, :]

    @property
    def query_cat_mask(self):
        return self._query_cat_mask[:self._current_batch_size, :]

    @property
    def query_indices(self):
        return self._query_indices[:self._current_batch_size, :]

    @property
    def row_idx(self):
        return self._row_idx[:self._current_batch_size]

    def loading_new_users(self, current_batch_size: int):
        """
        Limit the access to the only part of tensors which have been refilled in the last batch (to execute at every batch)
        """
        self._current_batch_size = current_batch_size
        self._current_charged_log_nb = 0

    def set_user_query_meta_data(self, user_idx, user_id, n, qq, ql, qc, qc_mask, qc_nb, mq, ml, mc, mc_mask, mc_nb):
        """
        Fill the tensor data with a new user
        """
        # Number of query question for the current user
        self._query_len[user_idx] = n
        self._query_users_vec[user_idx] = user_id

        # Saving logs questions, responses and number of categories
        self._query_questions[user_idx, :n] = qq
        self._query_responses[user_idx, :n] = ql
        self._query_cat_nb[user_idx, :n] = qc_nb

        # Saving the categories associated to each question and their mask
        self._query_cat[user_idx, :qc.shape[0]] = qc
        self._query_cat_mask[user_idx, :qc.shape[0]] = qc_mask

        ### ----- Meta questions
        # Saving meta users, questions, responses and number of categories
        self._meta_users[user_idx] = torch.full(mq.shape, user_id, dtype=torch.long, device=self.device)
        self._meta_questions[user_idx] = mq
        self._meta_responses[user_idx] = ml
        self._meta_cat_nb[user_idx] = mc_nb

        # Saving the categories associated to each question and their mask
        self._meta_cat[user_idx, :mc.shape[0]] = mc
        self._meta_cat_mask[user_idx, :mc.shape[0]] = mc_mask

    def update(self, actions: Tensor, t: int) -> None:
        """
        Move selected questions from query set to submitted set
        """
        actions += t

        with torch.no_grad():
            idx = self._current_charged_log_nb

            new_user_ids, new_question_ids, new_labels = self.generate_IMPACT_query(
                self.query_questions[self.row_idx, self.query_indices[self.row_idx, actions]],
                self.query_responses[self.row_idx, self.query_indices[self.row_idx, actions]],
                self.query_cat_nb[self.row_idx, self.query_indices[self.row_idx, actions]],
                self.query_users_vec)

            start = self.query_indices[self.row_idx, actions] * self.cq_max  # (shape: [batch_size])
            offset = torch.arange(self.cq_max, device=self.device).unsqueeze(0)  # (shape: [1, cq_max])
            indices = start.unsqueeze(1) + offset  # (shape: [batch_size, cq_max])
            new_category_ids = self.query_cat.gather(1, indices)[self.query_cat_mask.gather(1, indices)]

            # Update the tensor of query indices
            tmp = self.query_indices[self.row_idx, t]
            self.query_indices[self.row_idx, t] = self.query_indices[self.row_idx, actions]
            self.query_indices[self.row_idx, actions] = tmp

            # increment the number of submitted logs
            self._current_charged_log_nb += new_user_ids.shape[0]

            # Add the new data to the set of submitted questions
            self.sub_user_ids[idx:self._current_charged_log_nb] = new_user_ids
            self.sub_question_ids[idx:self._current_charged_log_nb] = new_question_ids
            self.sub_labels[idx:self._current_charged_log_nb] = new_labels
            self.sub_category_ids[idx:self._current_charged_log_nb] = new_category_ids

    def get_query_options(self, t):
        """
        Return a dictionary with the tensors (batch_size * (data.n_query -t)) of the query questions for each user
        :param t:
        :return:
        """
        return {
            'query_questions': self.query_questions[
                self.row_idx.unsqueeze(1).expand(-1, self.n_query - t), self.query_indices[:, t:]],
            'query_users': self.query_users_vec,
            'query_len': self.query_len - t,
        }

    def feed_IMPACT_sub(self):
        """
        Return the dictionary of submitted data logs for IMPACT retraining
        :return:
        """
        return {
            "user_ids": self.sub_user_ids[:self._current_charged_log_nb],
            "question_ids": self.sub_question_ids[:self._current_charged_log_nb],
            "labels": self.sub_labels[:self._current_charged_log_nb],
            "category_ids": self.sub_category_ids[:self._current_charged_log_nb]}

    def generate_IMPACT_query(self, QQ, QL, QC_NB, U):
        """
        Extends the query data for multiple categories handling with IMPACT format
        :param QQ:
        :param QL:
        :param QC_NB:
        :param U:
        :return:
        """

        question_ids = torch.repeat_interleave(QQ, QC_NB)
        user_ids = torch.repeat_interleave(U, QC_NB)
        labels = torch.repeat_interleave(QL, QC_NB)

        return user_ids, question_ids, labels

    def generate_IMPACT_meta(self):
        """
        Extends the meta data for multiple categories handling with IMPACT format
        :return:
        """
        MQ = self.meta_questions.reshape(-1)
        ML = self.meta_responses.reshape(-1)
        MC_NB = self.meta_cat_nb.reshape(-1)
        MU = self.meta_users.reshape(-1)

        question_ids = torch.repeat_interleave(MQ, MC_NB)
        user_ids = torch.repeat_interleave(MU, MC_NB)
        labels = torch.repeat_interleave(ML, MC_NB)
        category_ids = self.meta_cat[self.meta_cat_mask]

        return user_ids, question_ids, labels, category_ids


class UserCollate(object):
    def __init__(self, query_env: QueryEnv):
        self.query_env = query_env

    def __call__(self, batch):
        """
        Collate users data into a batch
        """

        self.query_env.loading_new_users(len(batch))

        for i, b in enumerate(batch):

            (u, qq, ql, qc, qc_mask, qc_nb, mq, ml, mc, mc_mask, mc_nb) = b.values()

            ### ----- Query questions
            # Number of query question for the current user
            n = qq.shape[0]
            self.query_env.set_user_query_meta_data(i, u, n, qq, ql, qc, qc_mask, qc_nb, mq, ml, mc, mc_mask, mc_nb)

        return None


class SubmittedDataset(Dataset):
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
