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

    def __init__(self, df, concept_map, metadata, config, nb_modalities):
        """
        Args:
            df: Dataframe with columns (user_id, item_id, correct, dimension_id)
            concept_map: dict, concept map {qid: cid}
            metadata : dict of keys {"num_user_id", "num_item_id", "num_dimension_id"}, containing the total number of users, items and concepts
        """

        super().__init__(df.to_records(index=False, column_dtypes={'user_id': int, 'item_id': int, "dimension_id": int,
                                                                   "correct": float}), concept_map, metadata,nb_modalities)

        self._df = df
        self._config = config
        self._query_seed = config['seed']
        self.device = config['device']

        self._questions_id = self.items_id

        self._n_meta = self._metadata['min_nb_users_logs'] // 5
        self._qu_max = self._metadata['max_nb_questions_per_user']

        assert max(self._users_id) < self.n_users, \
            f'Require item ids renumbered : max user id = {max(self._users_id)}; nb users = {self.n_users}'
        assert max(self._questions_id) < self.n_questions, \
            f'Require item ids renumbered : max item id = {max(self._questions_id)}; nb items = {self.n_questions}'
        assert max(self._concepts_id) < self.n_categories, \
            f'Require concept ids renumbered : max concept id = {max(self._concepts_id)}; nb categories= {self.n_categories}'
        assert self._n_meta <= self._metadata['min_nb_users_logs'] - self.config["n_query"], \
            f'Some users have not enough logs to submit {self.config["n_query"]} questions, the support set is too small: min number of user logs = {self._metadata["min_nb_users_logs"]}'

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
        @return: Total number of questions in the dataset (before splitting)
        """
        return self.metadata["num_item_id"]

    @property
    def config(self):
        """
        @return: Global configuration dictionary
        """
        return self._config

    @property
    def n_meta(self):
        """
        @return: Size of the meta set (nb of questions set aside for each user in order to perform evaluation)
        """
        return self._n_meta

    @property
    def n_query(self):
        """
        @return: Nb of questions to query per user from the support set
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
        """
        @return: Maximum Number of categories per question in the dataset
        """
        return self.metadata['max_nb_categories_per_question']

    @property
    def qu_max(self):
        """
        @return: Maximum Number of questions per user in the dataset
        """
        return self._qu_max

    @property
    def sup_max(self):
        """
        @return: Maximum size of the support set
        """
        return self.qu_max - self.n_meta

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

    def __init__(self, data, concept_map, metadata, config, nb_modalities):
        """
        Args:
            data: list, [(sid, qid, score)]
            concept_map: dict, concept map {qid: cid}
            metadata : dict of keys {"num_user_id", "num_item_id", "num_dimension_id"}, containing the total number of users, items and concepts
        """
        Dataset.__init__(self, data, concept_map, metadata, config, nb_modalities)

        self.rng = np.random.default_rng(self.query_seed)

    def reset_rng(self):
        """
        Reset the random number generator to a new seed
        :param seed: new seed
        """
        self.rng = np.random.default_rng(self.query_seed)

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.user_dict)

    def generate_sample(self, index):
        """
        Method to generate a sample of data from user indexed by 'index'. Randomly split data into Support and Meta sets. Each call of the function with the same parameter return different splitting. However, the process is reproducible.

        :param index: index of a user in the user_dict (index != user id: some user do not have data and are therefore not in the user_dict
        :return: dictionary of support and meta data of the user, in kit for Dataloader collate function
        """
        data = self.user_dict[index]

        observed_index = (self.rng.permutation(data['q_ids'].shape[0]) + index) % data['q_ids'].shape[0]
        meta_index = observed_index[-self.n_meta:]
        query_index = observed_index[:-self.n_meta]

        return {'u_idx' :self.user_idx2id[index],  # int

                # Support set
                'sq':data['q_ids'][query_index],  # 1D torch.Tensor, size = Q_u
                'sl':data['labels'][query_index],  # 1D torch.Tensor, size = Q_u
                'sc':self.cat_tensor[data['q_ids'][query_index]].flatten(),  # 1D torch.Tensor, size = Q_u x cq_max
                'sc_mask':self.cat_mask[data['q_ids'][query_index]].flatten(),  # 1D torch.Tensor, size = Q_u x cq_max
                'sc_nb':self.cat_nb[data['q_ids'][query_index]],  # 1D torch.Tensor, size = Q_u

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

    def __init__(self, data, concept_map, metadata, config,nb_modalities):
        """
        Args:
            data: list, [(sid, qid, score)]
            concept_map: dict, concept map {qid: cid}
            metadata : dict of keys {"num_user_id", "num_item_id", "num_dimension_id"}, containing the total number of users, items and concepts
        """
        super().__init__(data, concept_map, metadata, config, nb_modalities)
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
        Split the dataset into a query and a meta set once and for all
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
    QueryEnv manages the Support set, Query Set, Submitted set and the Meta set (prealocation, storage, update) :
        - store the data : preallocate a container which is then emptied and refilled at every users batch
        - save and update questions membership in Query, Submitted and Meta set
        - transform the data into CDM's compatible formats
    The data of each batch of users overwrites the previous one in the data container to optimize GPU memory allocation
    """

    def __init__(self, data: CATDataset, device: torch.device, batch_size: int):
        """

        :param data: CATDataset object
        :param device: torch.device object
        :param batch_size: Number of users in the batch
        """
        self.n_query = data.config['n_query'] # Nb of questions to query
        self.device = device
        self.cq_max = data.cq_max # Max nb of categories per question

        self.sup_max = data.sup_max
        self.n_meta = data.n_meta # Size of the meta set (nb of questions set aside for each user in order to perform evaluation)
        max_sub_data_batch_size = batch_size * self.n_query * self.cq_max  # maximum size of all the submitted data in this batch


        # Initialize attributes
        ## Variable storing the current batch size
        self._current_batch_size = batch_size

        ## Support container (torch.Tensor) storing all support data (query and submitted). 2D form: (l=users,c=questions). Before expansion for multiple categories/question
        self._support_len = torch.empty(batch_size, dtype=torch.long, device=device) # Shape = U_batch
        self._support_users_vec = torch.empty(batch_size, dtype=torch.long, device=device) # Shape = U_batch
        self._support_questions = torch.empty(batch_size, data.sup_max, dtype=torch.long, device=device) # Shape = U_batch x |max_u Q_sup(u)|
        self._support_responses = torch.empty(batch_size, data.sup_max, dtype=torch.long, device=device) # Shape = U_batch x |max_u Q_sup(u)|
        self._support_cat_nb = torch.empty(batch_size, data.sup_max, dtype=torch.long, device=device) # Shape = U_batch x |max_u Q_sup(u)|
        self._support_cat = torch.empty(batch_size, data.sup_max * data.cq_max, dtype=torch.long, device=device) # Shape = U_batch x |max_u Q_sup(u)| x cq_max
        self._support_cat_mask = torch.empty(batch_size, data.sup_max * data.cq_max, dtype=torch.bool, device=device) # Shape = U_batch x |max_u Q_sup(u)| x cq_max

        ## Submitted questions container (torch.Tensor) storing submitted data. 1D form: (l=users x questions x cat/q). After expansion for multiple categories/question
        self._sub_user_ids = torch.empty(max_sub_data_batch_size, dtype=torch.long, device=device)
        self._sub_question_ids = torch.empty(max_sub_data_batch_size, dtype=torch.long, device=device)
        self._sub_labels = torch.empty(max_sub_data_batch_size, dtype=torch.long, device=device)
        self._sub_category_ids = torch.empty(max_sub_data_batch_size, dtype=torch.long, device=device)

        ## Tensor storing the indices of the questions submitted to the user
        self._support_indices = torch.arange(0, data.sup_max, device=device, dtype=torch.long).repeat(batch_size,
                                                                                                      1)  # tensor of size (batch_size, sup_max)

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
    def current_charged_log_nb(self):
        return self._current_charged_log_nb

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
    def support_len(self):
        return self._support_len[:self._current_batch_size]

    @property
    def support_users_vec(self):
        return self._support_users_vec[:self._current_batch_size]

    @property
    def support_questions(self):
        return self._support_questions[:self._current_batch_size, :]

    @property
    def support_responses(self):
        return self._support_responses[:self._current_batch_size, :]

    @property
    def support_cat_nb(self):
        return self._support_cat_nb[:self._current_batch_size, :]

    @property
    def support_cat(self):
        return self._support_cat[:self._current_batch_size, :]

    @property
    def support_cat_mask(self):
        return self._support_cat_mask[:self._current_batch_size, :]

    @property
    def support_indices(self):
        return self._support_indices[:self._current_batch_size, :]

    @property
    def sub_user_ids(self):
        return self._sub_user_ids[:self._current_charged_log_nb]
    
    @property
    def sub_question_ids(self):
        return self._sub_question_ids[:self._current_charged_log_nb]
    
    @property
    def sub_labels(self):
        return self._sub_labels[:self._current_charged_log_nb]
    
    @property
    def sub_category_ids(self):
        return self._sub_category_ids[:self._current_charged_log_nb]

    @property
    def row_idx(self):
        return self._row_idx[:self._current_batch_size]
    
    def step_support_len(self,t):
        return self.support_len - t

    def load_batch(self, batch):
        """
        Adapt the data container to the new batch of users by limiting the access to the only part of the container which have been refilled (to execute at every batch)
        """

        self._current_batch_size = len(batch)
        self._current_charged_log_nb = 0
        self._support_indices[:self.current_batch_size,:] = torch.arange(0, self.sup_max, device=self.device, dtype=torch.long).repeat(self.current_batch_size,
                                                                                                      1)

        for i, b in enumerate(batch):
            (u, sq, sl, sc, sc_mask, sc_nb, mq, ml, mc, mc_mask, mc_nb) = b.values()

            ### ----- Support questions
            # Number of support questions for the current user
            n = sq.shape[0]
            self._support_len[i] = n
            self._support_users_vec[i] = u

            # Saving logs questions, responses and number of categories
            self._support_questions[i, :n] = sq
            self._support_responses[i, :n] = sl
            self._support_cat_nb[i, :n] = sc_nb

            # Saving the categories associated to each question and their mask
            self._support_cat[i, :sc.shape[0]] = sc
            self._support_cat_mask[i, :sc.shape[0]] = sc_mask

            ### ----- Meta questions
            # Saving meta users, questions, responses and number of categories
            self._meta_users[i] = torch.full(mq.shape, u, dtype=torch.long, device=self.device)
            self._meta_questions[i] = mq
            self._meta_responses[i] = ml
            self._meta_cat_nb[i] = mc_nb

            # Saving the categories associated to each question and their mask
            self._meta_cat[i, :mc.shape[0]] = mc
            self._meta_cat_mask[i, :mc.shape[0]] = mc_mask



    def update(self, actions: Tensor, t: int) -> None:
        """

        Move selected questions from query set to submitted set
        :param actions: Indices in the support set of the questions to be moved -> submitted questions = questions[indices[actions]]. Shape = (batch_size, 1)
        """
        assert (actions < self.step_support_len(t)).all(), "Actions should be in the range [0, support_len-t]"

        actions_indices = self.support_indices[self.row_idx,actions+t]  # Indices in the support set of the questions to be moved : question submitted = questions[actions_indices]

        with torch.no_grad():
            

            new_user_ids, new_question_ids, new_labels = self.generate_IMPACT_query(
                self.support_questions[self.row_idx, actions_indices],
                self.support_responses[self.row_idx, actions_indices],
                self.support_cat_nb[self.row_idx, actions_indices],
                self.support_users_vec)

            start = actions_indices * self.cq_max  # (shape: [batch_size])
            offset = torch.arange(self.cq_max, device=self.device).unsqueeze(0)  # (shape: [1, cq_max])
            indices = start.unsqueeze(1) + offset  # (shape: [batch_size, cq_max])
            new_category_ids = self.support_cat.gather(1, indices)[self.support_cat_mask.gather(1, indices)]

            # Update the tensor of query indices
            tmp = self.support_indices[self.row_idx, t]
            self.support_indices[self.row_idx, t] = actions_indices
            self._support_indices.index_put_((self.row_idx, actions+t), tmp)

            # Add the new data to the set of submitted questions
            idx = self.current_charged_log_nb

            ## increment the number of submitted logs
            self._current_charged_log_nb += new_user_ids.shape[0]

            self._sub_user_ids[idx:self.current_charged_log_nb] = new_user_ids
            self._sub_question_ids[idx:self.current_charged_log_nb] = new_question_ids
            self._sub_labels[idx:self.current_charged_log_nb] = new_labels
            self._sub_category_ids[idx:self.current_charged_log_nb] = new_category_ids

    def get_query_options(self, t):
        """
        Return a dictionary with the tensors (batch_size * (data.n_query -t)) of the query questions for each user
        :param t:
        :return:
        """

        return {
            'support_questions': self.support_questions[
                self.row_idx.unsqueeze(1).expand(-1, self.sup_max - t), self.support_indices[:, t:]],
            'support_users': self.support_users_vec,
            'support_len': self.step_support_len(t)
        }

    def feed_IMPACT_sub(self):
        """
        Return the dictionary of submitted data logs for IMPACT retraining
        :return:
        """
        return {
            "user_ids": self.sub_user_ids,
            "question_ids": self.sub_question_ids,
            "labels": self.sub_labels,
            "category_ids": self.sub_category_ids}

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

        return batch


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
