import logging
from collections import defaultdict, deque

import torch
import warnings
from torch.utils import data
import numpy as np
import pandas as pd
import random
import itertools
import numpy as np
from numba import njit, typed


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

        assert max(self._users_id) < self.n_users, \
            f'Require item ids renumbered : max user id = {max(self._users_id)}; nb users = {self.n_users}'
        assert max(self._questions_id) < self.n_questions, \
            f'Require item ids renumbered : max item id = {max(self._questions_id)}; nb items = {self.n_questions}'
        assert max(self._concepts_id) < self.n_categories, \
            f'Require concept ids renumbered : max concept id = {max(self._concepts_id)}; nb categories= {self.n_categories}'
        assert self._metadata['min_nb_users_logs'] // 5 >= self.config["n_query"], \
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


class CATDataset(Dataset, data.dataset.Dataset):

    def __init__(self, data, concept_map, metadata, config):
        """
        Args:
            data: list, [(sid, qid, score)]
            concept_map: dict, concept map {qid: cid}
            metadata : dict of keys {"num_user_id", "num_item_id", "num_dimension_id"}, containing the total number of users, items and concepts
        """
        super().__init__(data, concept_map, metadata, config)

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.user_dict)

    def __getitem__(self, index):
        # return the data of the user reindexed in this dataset instance
        # Warning : index is not the user id but the index of the user in the dataset!
        'Generates one sample of data'
        data = self.user_dict[index]

        rng = np.random.default_rng(index + self.query_seed)  # create a generator with a fixed seed
        observed_index = rng.permutation(data['q_ids'].shape[0])
        meta_index = observed_index[-self.config['n_query']:]
        query_index = observed_index[:-self.config['n_query']]

        query_concepts_nb = torch.tensor(
            [len(self.concept_map[question.item()]) for question in data['q_ids'][query_index]], device=self.device)
        qc = [self.concept_map[question.item()] for question in data['q_ids'][query_index]]

        meta_concepts_nb = torch.tensor(
            [len(self.concept_map[question.item()]) for question in data['q_ids'][meta_index]], device=self.device)
        mc = list(itertools.chain.from_iterable(
            self.concept_map[question.item()] for question in data['q_ids'][meta_index]
        ))

        result = (self.user_idx2id[index],# int
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

def feedIMPACT(QQ,QL,QC_NB,U,QC, device):

    question_ids = torch.repeat_interleave(QQ, QC_NB)
    user_ids = torch.repeat_interleave(U, QC_NB)
    labels = torch.repeat_interleave(QL, QC_NB)

    return user_ids, question_ids, labels, QC

def feedIMPACT_meta(MQ,ML,MC_NB,MU,MC):
    MQ = MQ.reshape(-1)
    ML = ML.reshape(-1)
    MC_NB = MC_NB.reshape(-1)
    MU = MU.reshape(-1)

    question_ids = torch.repeat_interleave(MQ, MC_NB)
    user_ids = torch.repeat_interleave(MU, MC_NB)
    labels = torch.repeat_interleave(ML, MC_NB)
    category_ids = MC

    return user_ids, question_ids, labels, category_ids

def ll2tensor(ll, device, dtype=torch.float):
    total_length = sum(len(lst) for lst in ll)
    result = torch.empty(total_length, dtype=dtype, device=device)
    pos = 0
    for lst in ll:
        l = len(lst)
        # Convert the current list to a tensor and copy it into the preallocated result.
        result[pos:pos + l] = torch.tensor(lst, dtype=dtype, device=device)
        pos += l
    return result

def remove(ll, ll_sub,remove_indices):
    # Convert removal indices to a simple list of integers
    rem_idxs = remove_indices.tolist()
    new_ll = []

    for i, sublist in enumerate(ll):
        rem_index = rem_idxs[i]
        # Save the removed element
        ll_sub[i].append(sublist[rem_index])
        # Build the new sublist without the removed element
        new_ll.append([v for j, v in enumerate(sublist) if j != rem_index])
    return new_ll, ll_sub

def remove_from_list(ll, ll_sub, remove_indices):
    # Convert removal indices to a simple list of integers
    rem_idxs = remove_indices
    new_ll = []
    for i, sublist in enumerate(ll):
        rem_index = rem_idxs[i]
        # Save the removed element
        ll_sub[i].extend(sublist[rem_index])
        # Build the new sublist without the removed element
        new_ll.append([v for j, v in enumerate(sublist) if j != rem_index])
    return new_ll, ll_sub

class CustomCollate(object):
    def __init__(self, data: CATDataset):
        self.data = data

    def __call__(self, batch):

        I = torch.arange(0, self.data.n_questions, device=self.data.device, dtype=torch.long).repeat(len(batch),1)
        Lengths = torch.zeros(size=(len(batch),), device=self.data.device, dtype=torch.long)

        QU = torch.zeros(size=(len(batch),), device=self.data.device, dtype=torch.long)
        QQ = torch.zeros(size=(len(batch), self.data.n_questions), device=self.data.device, dtype=torch.long)
        QL = torch.zeros(size=(len(batch), self.data.n_questions), device=self.data.device, dtype=torch.long)
        QC_NB = torch.zeros(size=(len(batch), self.data.n_questions), device=self.data.device, dtype=torch.long)

        MU = torch.zeros(size=(len(batch), self.data.config['n_query']), device=self.data.device, dtype=torch.long)
        MQ = torch.zeros(size=(len(batch), self.data.config['n_query']), device=self.data.device, dtype=torch.long)
        ML = torch.zeros(size=(len(batch), self.data.config['n_query']), device=self.data.device, dtype=torch.long)
        MC_NB = torch.zeros(size=(len(batch), self.data.config['n_query']), device=self.data.device, dtype=torch.long)

        qc_list = []
        mc_list = []

        for i, (u, qq, ql, qc, qc_nb, mq, ml, mc, mc_nb) in enumerate(batch) :
            N = qq.shape[0]
            Lengths[i] = N
            QU[i] = u

            I[i,N:] = 0

            QQ[i,:N] = qq
            QL[i, :N] = ql
            QC_NB[i, :N] = qc_nb

            qc_list.append(qc)

            MU[i] = u*torch.ones_like(mq)
            MQ[i] = mq
            ML[i] = ml
            MC_NB[i] = mc_nb

            mc_list.extend(mc)

        MC = torch.tensor(mc_list, device=self.data.device).int()

        return Lengths,I,QU,QQ,QL,QC_NB,MQ,ML,MC_NB, qc_list, MC, MU

class IMPACTCollate(object):
    def __init__(self, data: CATDataset):
        self.data = data

    def __call__(self, batch):
        qq_list = []
        ql_list = []
        qc_list = []
        qu_list = []
        mq_list = []
        ml_list = []
        mc_list = []
        mu_list = []

        for qq, ql, qc, qu, mq, ml, mc, mu in batch:
            qq_list.append(qq)
            ql_list.append(ql)
            qc_list.append(qc)
            qu_list.append(qu)
            mq_list.append(mq)
            ml_list.append(ml)
            mc_list.append(mc)
            mu_list.append(mu)

        qq_tensor = torch.cat(qq_list)
        ql_tensor = torch.cat(ql_list)
        qc_tensor = torch.cat(qc_list)
        qu_tensor = torch.cat(qu_list)
        mq_tensor = torch.cat(mq_list)
        ml_tensor = torch.cat(ml_list)
        mc_tensor = torch.cat(mc_list)
        mu_tensor = torch.cat(mu_list)

        return qq_tensor, ql_tensor, qc_tensor, qu_tensor, mq_tensor, ml_tensor, mc_tensor, mu_tensor
