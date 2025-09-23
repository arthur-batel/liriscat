import functools
from collections import defaultdict

from torch.masked import MaskedTensor
import torch.nn as nn
from IMPACT.model.IMPACT import resp_to_mod

import torch.utils.data as data

from IMPACT.dataset import *
from IMPACT.model import IMPACT
import torch.nn.functional as F
from torch.utils.data import DataLoader

from micat import dataset
from micat import utils

import warnings
import torch
import logging

warnings.filterwarnings(
    "ignore",
    message=r".*The PyTorch API of MaskedTensors is in prototype stage and will change in the near future. Please open a Github issue for features requests and see our documentation on the torch.masked module for further information about the project.*",
    category=UserWarning
)

# coding: utf-8
# 2021/4/1 @ WangFei

import logging
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score, root_mean_squared_error
from EduCDM import CDM


class PosLinear(nn.Linear):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        weight = 2 * F.relu(1 * torch.neg(self.weight)) + self.weight
        return F.linear(input, weight, self.bias)


class Net(nn.Module):

    def __init__(self, knowledge_n, exer_n, student_n):
        self.knowledge_dim = knowledge_n
        self.exer_n = exer_n
        self.emb_num = student_n
        self.stu_dim = self.knowledge_dim
        self.prednet_input_len = self.knowledge_dim
        self.prednet_len1, self.prednet_len2 = 512, 256  # changeable

        super(Net, self).__init__()

        # prediction sub-net
        self.student_emb = nn.Embedding(self.emb_num, self.stu_dim)
        self.k_difficulty = nn.Embedding(self.exer_n, self.knowledge_dim)
        self.e_difficulty = nn.Embedding(self.exer_n, 1)
        self.prednet_full1 = PosLinear(self.prednet_input_len, self.prednet_len1)
        self.drop_1 = nn.Dropout(p=0.5)
        self.prednet_full2 = PosLinear(self.prednet_len1, self.prednet_len2)
        self.drop_2 = nn.Dropout(p=0.5)
        self.prednet_full3 = PosLinear(self.prednet_len2, 1)

        # initialize
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)

    @property
    def users_emb(self):
        return self.student_emb


class NCDM(CDM):
    '''Neural Cognitive Diagnosis Model'''

    @property
    def name(self):
        return "NCDM"

    @property
    def model(self):
        return self.ncdm_net

    def __init__(self, knowledge_n, exer_n, student_n, config):
        super().__init__()
       
        self.ncdm_net = Net(knowledge_n, exer_n, student_n)
        self.config = config

        if self.config['load_params']:
            self.load()

    

class CATNCDM(NCDM) :

    def __init__(self, **config):
        self.config=config
        self.initialized_users_prior = False

        
    def init_CDM_model(self, train_data: dataset.Dataset, valid_data: dataset.Dataset):
        super().__init__(train_data.n_categories,train_data.n_items, train_data.n_users, self.config)

        self.knowledge_emb = torch.nn.Embedding(train_data.n_items, train_data.n_categories, device = self.config['device'])
        for item_idx, knowledges in train_data.concept_map.items():
            self.knowledge_emb.weight.data[item_idx][np.array(knowledges)] = 1.0

        # Replacement of pretrained users embeddings with randomly generated ones
        self.model.train_valid_users = torch.tensor(list(train_data.users_id.union(valid_data.users_id)), device=self.config['device'])

        self.model.to(self.config['device'])
    
    def get_params(self):
        return self.model.state_dict()

    def init_test(self, test_data):
        pass  

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

    def get_regularizer(self,unique_users, unique_items, learning_users_emb):

        return learning_users_emb[unique_users].norm().pow(2)

    def forward(self, users_id, items_id, concepts_id, users_emb):

        stu_emb = users_emb[users_id]  # [batch_size, knowledge_dim]
        stat_emb = torch.sigmoid(stu_emb)
        k_difficulty = torch.sigmoid(self.model.k_difficulty(items_id))
        e_difficulty = torch.sigmoid(self.model.e_difficulty(items_id))  # * 10
        # prednet
        input_x = e_difficulty * (stat_emb - k_difficulty) * self.knowledge_emb(concepts_id)
        input_x = self.model.drop_1(torch.sigmoid(self.model.prednet_full1(input_x)))
        input_x = self.model.drop_2(torch.sigmoid(self.model.prednet_full2(input_x)))
        output_1 = self.model.prednet_full3(input_x)
        output_2 = torch.sigmoid(output_1)

        return output_2.view(-1)+1.0

    def _compute_loss(self, users_id, items_id, concepts_id, labels, learning_users_emb):

        stu_emb = learning_users_emb[users_id]  # [batch_size, knowledge_dim]
        stat_emb = torch.sigmoid(stu_emb)
        k_difficulty = torch.sigmoid(self.model.k_difficulty(items_id))
        e_difficulty = torch.sigmoid(self.model.e_difficulty(items_id))  # * 10
        # prednet
        input_x = e_difficulty * (stat_emb - k_difficulty) * self.knowledge_emb(concepts_id)
        input_x = self.model.drop_1(torch.sigmoid(self.model.prednet_full1(input_x)))
        input_x = self.model.drop_2(torch.sigmoid(self.model.prednet_full2(input_x)))
        output_1 = self.model.prednet_full3(input_x)
        output_2 = torch.sigmoid(output_1).view(-1)

        binary_labels = (labels == 2).float()

        with torch.amp.autocast('cuda',enabled=False):
            L1 = nn.BCELoss()(output_2.float(),binary_labels)

        unique_users = torch.unique(users_id)
        unique_items = torch.unique(items_id)

        R = self.get_regularizer(unique_users, unique_items, learning_users_emb)

        return L1, None, R

    def train(self, train_data, test_data=None, epoch=10, lr=0.002, silence=False):
        self.ncdm_net.train()
        loss_function = nn.BCELoss()
        optimizer = optim.Adam(self.ncdm_net.parameters(), lr=lr)

        self.best_valid_rmse = float('inf')
        self.best_epoch = 0
        best_valid_params = None

        train_loader = DataLoader(train_data, batch_size=self.config['batch_size'], shuffle=True)

        for epoch_i in range(epoch):
            epoch_losses = []
            batch_count = 0
            for batch_data in tqdm(train_loader, "Epoch %s" % epoch_i):
                batch_count += 1

                users_id, items_id, y, concepts_id = batch_data[:,0].int(), batch_data[:,1].int(), batch_data[:,2], batch_data[:,3].int()
                
                loss,_,_ = self._compute_loss(users_id, items_id, concepts_id, y, self.model.student_emb.weight)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_losses.append(loss.mean().item())

            print("[Epoch %d] average loss: %.6f" % (epoch_i, float(np.mean(epoch_losses))))
            if test_data is not None:
                rmse, accuracy = self.eval(test_data)
                print("[Epoch %d] rmse: %.6f, accuracy: %.6f" % (epoch_i, rmse, accuracy))
                if epoch_i - self.best_epoch > self.config['patience']:
                    break
                if rmse < self.best_valid_rmse:
                    self.best_valid_rmse = rmse
                    self.best_epoch = epoch_i
                    best_valid_params = self.model.state_dict()

        self.ncdm_net.load_state_dict(best_valid_params)

        if self.config['save_params']:
            self.save()

    def eval(self, test_data):

        test_loader = DataLoader(test_data, batch_size=self.config['batch_size'], shuffle=False)
        
        self.ncdm_net.eval()
        y_true, y_pred = [], []
        for batch_data in tqdm(test_loader, "Evaluating"):
            users_id, items_id, y, concepts_id = batch_data[:,0].int(), batch_data[:,1].int(), batch_data[:,2], batch_data[:,3].int()
            
            pred: torch.Tensor = self.forward(users_id, items_id, concepts_id, self.model.student_emb.weight)
            y_pred.extend(pred.detach().cpu().tolist())
            y_true.extend(y.tolist())

        return root_mean_squared_error(y_true, y_pred), np.mean((np.array(y_true)==np.array(y_pred).round()))

    def save(self) :
        path = self.config['params_path'] + self.config['dataset_name'] +'_'+ self.name + '_fold_' + str(self.config['i_fold']) + '_seed_' + str(
            self.config['seed'])
        torch.save(self.model.state_dict(), path+".pt")

    def load(self):
        path = self.config['params_path'] + self.config['dataset_name']+ '_'+ self.name + '_fold_' + str(self.config['i_fold']) + '_seed_' + str(
            self.config['seed'])

        print(path)
        self.model.load_state_dict(torch.load(path + '.pt'))
  
    
    