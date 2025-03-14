import torch.nn
from liriscat.selectionStrategy import AbstractSelectionStrategy
from torch import nn
import warnings


class Random(AbstractSelectionStrategy):
    def __init__(self, **config):
        super().__init__('Random', **config)
        self.model = RandomModel()
        print(self.model)

    def select_action(self,t,I,Lengths):

        # Generate random indices efficiently
        remove_indices = torch.randint(t, Lengths.max(), size=Lengths.shape, device=self.device)
        remove_indices = torch.remainder(remove_indices, Lengths-t) +t

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

class RandomModel(nn.Module):
    def __init__(self):
        super(RandomModel, self).__init__()

        self.rd_seed = torch.nn.parameter.Parameter()