import functools
from collections import defaultdict

from torch.masked import MaskedTensor
import torch.nn as nn

import torch.utils.data as data

from IMPACT.model.abstract_model import AbstractModel
from CAT.model import IRTModel
from IMPACT.dataset import *
from IMPACT.model import IMPACT
import torch.nn.functional as F
from torch.utils.data import DataLoader

from liriscat import dataset

import warnings
import torch

warnings.filterwarnings(
    "ignore",
    message=r".*The PyTorch API of MaskedTensors is in prototype stage and will change in the near future. Please open a Github issue for features requests and see our documentation on the torch.masked module for further information about the project.*",
    category=UserWarning
)



class CATIMPACT(IMPACT) :

    def __init__(self, **config):
        super().__init__(**config)

    def init_model(self, train_data: Dataset, valid_data: Dataset):
        super().init_model(train_data,valid_data)

        self.user_params_optimizer = torch.optim.Adam(self.model.users_emb.parameters(),
                                                      lr=self.config['inner_user_lr'])  # todo : Decide How to use a scheduler
        self.params_optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config['inner_lr'])

        self.params_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.user_params_optimizer, patience=2,
                                                                           factor=0.5)

        self.user_params_scaler = torch.amp.GradScaler(self.config['device'])
        self.params_scaler = torch.amp.GradScaler(self.config['device'])

    def initialize_test_users(self, test_data):
        train_valid_users = torch.tensor(list(set(range(test_data.n_users)) - set(test_data.users_id.tolist())),
                                         device=self.config['device'])
        ave = self.model.users_emb(train_valid_users).mean(dim=0)
        std = self.model.users_emb(train_valid_users).std(dim=0)
        self.model.users_emb.weight.data[test_data.users_id, :] = torch.normal(
            ave.expand(test_data.n_actual_users, -1), std.expand(test_data.n_actual_users, -1))

    def get_params(self):
        return self.model.state_dict()

    def update_params(self,user_ids, question_ids, labels, categories) :
        for _ in range(self.config['num_inner_epochs']) :
            self.params_optimizer.zero_grad()

            with torch.amp.autocast('cuda'):
                loss = self._compute_loss(user_ids, question_ids, categories, labels)

            self.params_scaler.scale(loss).backward()
            self.params_scaler.step(self.params_optimizer)
            self.params_scaler.update()

    def update_users(self,query_data) :
        data = dataset.SubmittedDataset(query_data)
        dataloader = DataLoader(data, batch_size=2048, shuffle=True)

        for _ in range(self.config['num_inner_users_epochs']) :
            self.user_params_optimizer.zero_grad()

            for batch in dataloader:
                user_ids = batch["user_ids"]
                question_ids = batch["question_ids"]
                labels = batch["labels"]
                category_ids = batch["category_ids"]

                with torch.amp.autocast('cuda'):
                    loss = self._compute_loss(user_ids, question_ids, category_ids, labels)

                self.user_params_scaler.scale(loss).backward()
                self.user_params_scaler.step(self.user_params_optimizer)
                self.user_params_scaler.update()


class CATIRT(IRTModel) :

    def __init__(self, **config):
        super().__init__(**config)