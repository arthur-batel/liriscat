import functools
from collections import defaultdict

from torch.masked import MaskedTensor
import torch.nn as nn
from IMPACT.model.IMPACT import resp_to_mod

import torch.utils.data as data

from IMPACT.model.abstract_model import AbstractModel

from IMPACT.dataset import *
from IMPACT.model import IMPACT
import torch.nn.functional as F
from torch.utils.data import DataLoader

from liriscat import dataset
from liriscat import utils

import warnings
import torch
import logging

warnings.filterwarnings(
    "ignore",
    message=r".*The PyTorch API of MaskedTensors is in prototype stage and will change in the near future. Please open a Github issue for features requests and see our documentation on the torch.masked module for further information about the project.*",
    category=UserWarning
)



class CATIMPACT(IMPACT) :

    def __init__(self, **config):
        super().__init__(**config)
    def init_model(self, train_data: dataset.Dataset, valid_data: dataset.Dataset):
        super().init_model(train_data,valid_data)

    def get_regularizer_with_pior(self):
        A = (self.model.users_emb.weight - self.model.prior_mean) 
        S = self.model.prior_cov_inv
        SA_T = torch.matmul(A, S)  
        
        return torch.bmm(SA_T.unsqueeze(1), A.unsqueeze(2)).sum()

    def get_params(self):
        return self.model.state_dict()



    def init_test(self, test_data):
        self.model.R = test_data.log_tensor
        self.model.ir_idx = resp_to_mod(self.model.R, self.model.nb_modalities)
        self.model.ir_idx = self.model.ir_idx.to(self.config['device'], non_blocking=True)
        self.init_users_prior(test_data)

    def init_users_prior(self, test_data):
        train_valid_users = torch.tensor(list(set(range(test_data.n_users)) - test_data.users_id),
                                         device=self.config['device'])
        train_valid_users = train_valid_users.to(dtype=torch.long)
        user_embeddings = self.model.users_emb(train_valid_users).float()
        
        ave = user_embeddings.mean(dim=0)
        std = user_embeddings.std(dim=0)

        E = torch.normal(ave.expand(test_data.n_actual_users, -1), std.expand(test_data.n_actual_users, -1)/2)
        E = E - E.mean(dim=0) + ave

        self.model.users_emb.weight.data[list(test_data.users_id), :] = E.to(self.config['device'])

        cov_matrix = torch.cov(user_embeddings.T).to(dtype=torch.float)

        self.model.prior_cov_inv = torch.inverse(cov_matrix)

        self.model.prior_mean = ave.unsqueeze(0)
        self.model.get_regularizer = functools.partial(self.get_regularizer_with_pior)


    def _compute_loss(self, users_id, items_id, concepts_id, labels):
        device = self.config['device']
        beta = 0.5

        lambda_param = self.config['lambda']

        u_emb, im_emb_prime, i0_emb_prime, in_emb_prime, W_t = self.model.get_embeddings(users_id, items_id,
                                                                                         concepts_id)

        L1, L2, L3 = self.loss(u_emb=u_emb, im_emb_prime=im_emb_prime, i0_emb_prime=i0_emb_prime,
                                 in_emb_prime=in_emb_prime, W_t=W_t,
                                 modalities_idx=self.model.ir_idx[users_id, items_id],
                                 nb_mod_max_plus_sent=self.model.nb_mod_max_plus_sent,
                                 diff_mask=self.model.diff_mask[items_id],
                                 diff_mask2=self.model.diff_mask2[items_id],
                                 users_id=users_id, items_id=items_id,
                                 concepts_id=concepts_id, R=self.model.R, users_emb=self.model.users_emb.weight)

        R = self.model.get_regularizer()

        # Stack losses into a tensor
        losses = torch.stack([L1, L3])  # Shape: (4,)

        # Update statistics and compute weights
        weights = self.L_W.compute_weights(losses)

        # Compute total loss
        total_loss = 1*L1 + lambda_param * R

        return total_loss


    def get_KLI(self, query_data) :

        preds = self.model(query_data)


