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
        
    def init_model(self, train_data: dataset.Dataset, valid_data: dataset.Dataset):
        super().init_model(train_data,valid_data)

        # Replacement of pretrained users embeddings with randomly generated ones
        self.init_users_prior(train_data, valid_data)
            
    def freeze_test_users(self, train_data, valid_data):
        self.model.users_emb.weight.requires_grad_(False)    # freeze all
        self.model.users_emb.weight[list(train_data.users_id)].requires_grad_(True)
        self.model.users_emb.weight[list(valid_data.users_id)].requires_grad_(True)

    def unfreeze_test_freeze_train_valid_users(self, test_data):
        self.model.users_emb.weight.requires_grad_(False)    # freeze all
        self.model.users_emb.weight[list(test_data.users_id)].requires_grad_(True)

    def get_regularizer_with_pior(self,unique_users, unique_items):    
        A = (self.model.users_emb.weight[unique_users] - self.model.prior_mean) 
        S = self.model.prior_cov_inv
        SA_T = torch.matmul(A, S)  
        
        return torch.bmm(SA_T.unsqueeze(1), A.unsqueeze(2)).sum()

    def get_params(self):
        return self.model.state_dict()


    def init_test(self, test_data):
        self.model.R = test_data.log_tensor
        self.model.ir_idx = resp_to_mod(self.model.R, self.model.nb_modalities)
        self.model.ir_idx = self.model.ir_idx.to(self.config['device'], non_blocking=True)    

    def init_users_prior(self, train_data, valid_data):
        """
        Initialize users embedding and set the regularization term based on the posterior distribution learned on train and valid users.
        """
        train_valid_users = torch.tensor(list(train_data.users_id.union(valid_data.users_id)),device=self.config['device'])
        train_valid_users = train_valid_users.to(dtype=torch.long)
        
        # NO data leak to test dataset because we only look at the train and valid users 
        user_embeddings = self.model.users_emb(train_valid_users).float().detach()
        
        ave = user_embeddings.mean(dim=0)
        std = user_embeddings.std(dim=0)

        E = torch.normal(ave.expand(train_data.n_users, -1), std.expand(train_data.n_users, -1)/2)
        E = E - E.mean(dim=0) + ave
        with torch.no_grad():
            self.model.users_emb.weight.data = E.to(self.config['device'])

        cov_matrix = torch.cov(user_embeddings.T).to(dtype=torch.float)

        self.model.prior_cov_inv = torch.inverse(cov_matrix)
        self.model.prior_mean = ave.unsqueeze(0)
        self.model.get_regularizer = functools.partial(self.get_regularizer_with_pior)
        
        self.initialized_users_prior = True

    def _compute_loss(self, users_id, items_id, concepts_id=None, labels=None):

        lambda_param = self.config['lambda']

        u_emb = self.get_users_emb(users_id)
        
        im_emb_prime = self.get_modalities_emb(items_id)

        L1 = custom_l1(u_emb=u_emb,
            im_emb_prime=im_emb_prime,
            modalities_idx=self.model.ir_idx[users_id, items_id],
            nb_mod_max_plus_sent=self.model.nb_mod_max_plus_sent,
            diff_mask=self.model.diff_mask[items_id])
        
        unique_users =  torch.unique(users_id)
        unique_items = torch.unique(items_id)
        R = self.model.get_regularizer(unique_users, unique_items)

        # Compute total loss
        total_loss = 1*L1 + lambda_param * R

        return total_loss
    
    
    def get_modalities_emb(self, items_id):

        # Compute item-response indices
        im_idx = self.model.im_idx[items_id]  # [batch_size, nb_mod]
        im_emb_prime = self.model.item_response_embeddings(im_idx)  # [batch_size, nb_mod, embedding_dim]

        return im_emb_prime
    
    def get_users_emb(self, users_id):
        u_emb = self.model.users_emb(users_id)

        return u_emb



    def get_KLI(self, query_data) :

        preds = self.model(query_data)

@torch.jit.script
def custom_l1(u_emb: torch.Tensor,
            im_emb_prime: torch.Tensor,
            modalities_idx: torch.Tensor,
            nb_mod_max_plus_sent: int,
            diff_mask: torch.Tensor):


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

   
    return L1


