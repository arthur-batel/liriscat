import functools
from collections import defaultdict

from torch.masked import MaskedTensor
import torch.nn as nn
from IMPACT.model.IMPACT import resp_to_mod

import torch.utils.data as data

from IMPACT.model.abstract_model import AbstractModel

from IMPACT.dataset import *
from IMPACT.model import IMPACT, IMPACTModel_low_mem, IMPACTModel, custom_loss_low_mem, custom_loss
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
        self.initialized_users_prior = False
        
    def init_CDM_model(self, train_data: dataset.Dataset, valid_data: dataset.Dataset):
        super().init_model(train_data,valid_data)

        # Replacement of pretrained users embeddings with randomly generated ones
        self.model.train_valid_users = torch.tensor(list(train_data.users_id.union(valid_data.users_id)), device=self.config['device'])
        self.model.R_train = train_data.log_tensor + valid_data.log_tensor

    def get_params(self):
        return self.model.state_dict()

    def init_test(self, test_data):
        self.model.R = test_data.log_tensor
        self.model.ir_idx = resp_to_mod(self.model.R, self.model.nb_modalities)
        self.model.ir_idx = self.model.ir_idx.to(self.config['device'], non_blocking=True)    

    def init_users_prior(self, train_data, valid_data):
        """
        Set the regularization term based on the posterior distribution learned on train and valid users.
        """
        train_valid_users = self.model.train_valid_users.to(dtype=torch.long)
        
        # NO data leak to test dataset because we only look at the train and valid users 
        user_embeddings = self.model.users_emb(train_valid_users).float().detach()
        
        ave = user_embeddings.mean(dim=0)

        self.model.cov_matrix = torch.cov(user_embeddings.T).to(dtype=torch.float)

        self.model.prior_cov_inv = torch.inverse(self.model.cov_matrix)
        self.model.prior_mean = ave.unsqueeze(0)
        
        self.initialized_users_prior = True

    def set_regularizer_with_prior(self):
        self.get_regularizer = self.get_regularizer_with_prior

    def get_regularizer_with_prior(self,unique_users, unique_items, user_emb):    
        A = (user_emb[unique_users] - self.model.prior_mean)  # [nb_users, d_in]
        S = self.model.prior_cov_inv
        SA_T = torch.matmul(A, S)  

        return torch.bmm(SA_T.unsqueeze(1), A.unsqueeze(2)).sum()

    def _loss_function(self, users_id, items_id, concepts_id, labels):
        return self._compute_loss(users_id, items_id, concepts_id, labels)

    def _compute_loss(self, users_id, items_id, concepts_id, labels, learning_users_emb):

        im_emb_prime = self.get_modalities_emb(items_id)

        nb_items_modalities = self.model.nb_modalities[items_id]

        L1, L3 = custom_l1(learning_users_emb=learning_users_emb,
            reference_users_emb=self.model.users_emb.weight,
            im_emb_prime=im_emb_prime,
            modalities_idx=self.model.ir_idx[users_id, items_id],
            nb_mod_max_plus_sent=self.model.nb_mod_max_plus_sent,
            diff_mask=self.model.diff_mask[items_id],
            users_id=users_id,
            train_valid_users=self.model.train_valid_users, 
            items_id=items_id,
            concepts_id=concepts_id,
            labels=labels,
            R=self.model.R_train,
            nb_items_modalities=nb_items_modalities)

        unique_users =  torch.unique(users_id)
        unique_items = torch.unique(items_id)

        R = self.get_regularizer(unique_users, unique_items, learning_users_emb)

        return L1, L3, R
    
    def get_regularizer(self,unique_users, unique_items, learning_users_emb):
        im_idx = self.model.im_idx[unique_items]  # [batch_size, nb_mod]
        i0_idx = self.model.i0_idx[unique_items]  # [batch_size, nb_mod]
        in_idx = self.model.in_idx[unique_items] 
        return learning_users_emb[unique_users].norm().pow(2) + self.model.item_response_embeddings.weight[im_idx].norm().pow(
            2) + self.model.item_response_embeddings.weight[i0_idx].norm().pow(
            2)+ self.model.item_response_embeddings.weight[in_idx].norm().pow(
            2)
    
    def forward(self, users_id, items_id, concepts_id, users_emb):
        # I
        im_idx = self.model.im_idx[items_id]
        im_emb = self.model.item_response_embeddings(im_idx)

        # E
        u_emb = users_emb[users_id]

        # p_uim
        diff = u_emb.unsqueeze(1) - im_emb
        p_uim = torch.sum(diff ** 2, dim=2)

        return mod_to_resp(torch.argmin(p_uim + self.model.mask[items_id, :], dim=1), self.model.nb_modalities[items_id])

    
    def get_modalities_emb(self, items_id):

        # Compute item-response indices
        im_idx = self.model.im_idx[items_id]  # [batch_size, nb_mod]
        im_emb_prime = self.model.item_response_embeddings(im_idx)  # [batch_size, nb_mod, embedding_dim]

        return im_emb_prime

@torch.jit.script
def custom_l1(learning_users_emb: torch.Tensor,
              reference_users_emb: torch.Tensor, # fixed users embeddings
            im_emb_prime: torch.Tensor,
            modalities_idx: torch.Tensor,
            nb_mod_max_plus_sent: int,
            diff_mask: torch.Tensor,
            users_id: torch.Tensor,
            train_valid_users: torch.Tensor,
            items_id: torch.Tensor,
            concepts_id: torch.Tensor,
            labels: torch.Tensor,
            R: torch.Tensor,
            nb_items_modalities: torch.Tensor):
    
    u_emb = learning_users_emb[users_id]


    diff = u_emb.unsqueeze(1) - im_emb_prime


    p_uir = -torch.sum(diff ** 2, dim=2)  # [batch_size, nb_mod_max_plus_sent]

    ##### L1
    device = p_uir.device

    # Compute differences between adjacent modalities
    diffs = p_uir[:, :-1] - p_uir[:, 1:]  # shape = [batch_size, nb_mod_max_plus_sent-1]

    # Compute loss terms for responses greater and less than r
    greater_mask = torch.arange(nb_mod_max_plus_sent - 1, device=device).unsqueeze(0) >= modalities_idx.unsqueeze(
        1)  # nb_mod_max_plus_sent - 1 : start at 0 (torch.arange ok), sentinels (included)
    less_mask = ~greater_mask

    L1 = torch.where(diff_mask == 1, F.softplus((less_mask.int() - greater_mask.int()) * diffs),
                     torch.zeros_like(diff_mask)).mean(dim=1).mean()
    
    ##### L3
    R_t = R[train_valid_users][:, items_id].t() # responses to compare to
    b = (labels.unsqueeze(1) - R_t)

    b_diag = b.abs()

    u_mask = (b_diag > 0.0) & (R_t >= 1.0) & (b_diag <= (2/nb_items_modalities).unsqueeze(1))   # b_diag > 0 : not exactly similar responses for which we cannot say anything; R_t >= 1.0 : comparison with not null responses only

    indices = torch.nonzero(u_mask)
    u_base_idx = indices[:, 0]
    u_comp_idx = indices[:, 1] # responses to compare to
    
    u_base_emb = learning_users_emb[users_id[u_base_idx], concepts_id[u_base_idx]]
    u_comp_emb = reference_users_emb[train_valid_users[u_comp_idx], concepts_id[u_base_idx]]
    
    sign_b = b[u_base_idx, u_comp_idx].sign()
    diff_emb = u_base_emb - u_comp_emb

    L3 = F.softplus(- sign_b * diff_emb).mean()

    return L1, L3  # L3 is not used in this implementation, returning 0

@torch.jit.script
def mod_to_resp(indexes: torch.Tensor, nb_modalities: torch.Tensor):
    indexes = indexes - 1  # sentinels remove -> [0,nb_modalities-1]
    responses = indexes / (nb_modalities - 1)  # -> [0,1]
    responses = responses + 1  # -> [1,2]
    return responses