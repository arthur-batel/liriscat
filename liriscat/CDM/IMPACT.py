import functools
from collections import defaultdict

from torch.masked import MaskedTensor
import torch.nn as nn

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


        self.params_optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config['inner_lr'],foreach=False )

        self.params_scaler = torch.amp.GradScaler(self.config['device'])

    def initialize_test_users(self, test_data):
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


    def get_regularizer_with_pior(self):
        A = (self.model.users_emb.weight - self.model.prior_mean) 
        S = self.model.prior_cov_inv
        SA_T = torch.matmul(A, S)  
        
        return torch.bmm(SA_T.unsqueeze(1), A.unsqueeze(2)).sum()

    def get_params(self):
        return self.model.state_dict()

    def update_params(self,user_ids, question_ids, labels, categories) :
        logging.debug("- Update params : ")
        
        for t in range(self.config['num_inner_epochs']) :

            self.params_optimizer.zero_grad()

            with torch.amp.autocast('cuda'):
                loss = self._compute_loss(user_ids, question_ids, categories, labels)

            self.params_scaler.scale(loss).backward()
            self.params_scaler.step(self.params_optimizer)
            self.params_scaler.update()

    def update_users(self,query_data, meta_data, meta_labels) :
        logging.debug("- Update users ")
        m_user_ids, m_question_ids, m_category_ids = meta_data

        data = dataset.SubmittedDataset(query_data)
        dataloader = DataLoader(data, batch_size=2048, shuffle=True, num_workers=0)

        user_params_optimizer = torch.optim.Adam(self.model.users_emb.parameters(),
                                                      lr=self.config[
                                                          'inner_user_lr'],foreach=False )  # todo : Decide How to use a scheduler

        user_params_scaler = torch.amp.GradScaler(self.config['device'])

        n_batches = len(dataloader)

        for t in range(self.config['num_inner_users_epochs']) :

            sum_loss_0 = 0
            sum_loss_1 = 0
            sum_acc_0 = 0
            sum_acc_1 = 0
            sum_meta_acc = 0
            sum_meta_loss = 0

            for batch in dataloader:
                user_ids = batch["user_ids"]
                question_ids = batch["question_ids"]
                labels = batch["labels"]
                category_ids = batch["category_ids"]

                self.model.eval()
                
                preds = self.model(user_ids, question_ids, category_ids)
                sum_acc_0 += utils.micro_ave_accuracy(labels, preds)

                meta_preds = self.model(m_user_ids, m_question_ids, m_category_ids)

                self.model.train()

                user_params_optimizer.zero_grad()

                with torch.amp.autocast('cuda'):
                    loss = self._compute_loss(user_ids, question_ids, labels, category_ids)
                    sum_loss_0 += loss.item()


                # loss.backward()
                # user_params_optimizer.step()
                user_params_scaler.scale(loss).backward()
                user_params_scaler.step(user_params_optimizer)
                user_params_scaler.update()

                with torch.amp.autocast('cuda'):
                    loss2 = self._compute_loss(user_ids, question_ids, labels, category_ids)
                    sum_loss_1 += loss2.item()

                

                self.model.eval()

                preds = self.model(user_ids, question_ids, category_ids)

                self.model.train()
                sum_acc_1 += utils.micro_ave_accuracy(labels, preds)
                sum_meta_acc += utils.micro_ave_accuracy(meta_labels, meta_preds)
                
            with torch.amp.autocast('cuda'):
                meta_loss = self._compute_loss(m_user_ids, m_question_ids, meta_labels, m_category_ids)
                sum_meta_loss += meta_loss.item()

                
            logging.debug(
                f'inner epoch {t} - query loss_0 : {sum_loss_0/n_batches:.5f} '
                f'- query loss_1 : {sum_loss_1/n_batches:.5f} '
                f'- query acc 0 : {sum_acc_0/n_batches:.5f} '
                f'- query acc 1 : {sum_acc_1/n_batches:.5f} '
                f'- meta acc 0 : {sum_meta_acc/n_batches:.5f}'
                f'- meta loss 1 : {sum_meta_loss:.5f}'
            )


    def get_KLI(self, query_data) :

        preds = self.model(query_data)


