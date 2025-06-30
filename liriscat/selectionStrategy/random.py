import logging

import torch.nn
from liriscat.selectionStrategy import AbstractSelectionStrategy
from torch import nn
import warnings


class Random(AbstractSelectionStrategy):
    def __init__(self, metadata,**config):
        super().__init__('Random', metadata, **config)
        self.model = RandomModel()
        logging.info(self.name)
        self.trainable = False
        self.config = config
        
    def select_action(self,options_dict):

        # Generate random indices efficiently
        remove_indices = torch.randint(high=options_dict['support_len'].max(), size=options_dict['support_len'].shape, device=self.device, generator=self._rng)
        remove_indices = torch.remainder(remove_indices, options_dict['support_len'])

        return remove_indices

    def _loss_function(self, users_id, items_id, concepts_id, labels,users_emb):
        return torch.tensor([0.1], requires_grad = True)


    def get_params(self):
        warnings.warn('get_params() Notimplemented')
        return None

    def _save_model_params(self):
        warnings.warn('_save_model_params() Notimplemented')
        self.get_params()
        return None

## ----------------- Helper functions ----------------- ##

class RandomModel(nn.Module):
    def __init__(self):
        super(RandomModel, self).__init__()

        self.rd_seed = torch.nn.parameter.Parameter()

    def forward(self, x):
        pass