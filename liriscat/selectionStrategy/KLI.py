import logging

import torch.nn
from liriscat.selectionStrategy import AbstractSelectionStrategy
from torch import nn
import warnings


class KLI(AbstractSelectionStrategy):
    def __init__(self, metadata,**config):
        super().__init__('KLI', metadata, **config)
        self.model = KLIModel()
        logging.info(self.name)

    def select_action(self,t,env):

        self.CDM.get_KLI(env.get_query_options(t))

        # Generate random indices efficiently
        remove_indices = torch.randint(t, env.query_len.max(), size=env.query_len.shape, device=self.device)
        remove_indices = torch.remainder(remove_indices, env.query_len - t) + t

        return remove_indices

    def _loss_function(self,user_ids, question_ids, categories, labels):
        return None

    def update_params(self,user_ids, question_ids, labels, categories):
        # self.optimizer.zero_grad()
        #
        # with torch.amp.autocast('cuda'):
        #     loss = self._loss_function(user_ids, question_ids, categories, labels)
        #
        # self.scaler.scale(loss).backward()
        # self.scaler.step(self.optimizer)
        # self.scaler.update()

        pass

    def get_params(self):
        warnings.warn('get_params() Notimplemented')
        return None

    def _save_model_params(self):
        warnings.warn('_save_model_params() Notimplemented')
        self.get_params()
        return None

## ----------------- Helper functions ----------------- ##

class KLIModel(nn.Module):
    def __init__(self):
        super(KLIModel, self).__init__()

        self.rd_seed = torch.nn.parameter.Parameter()