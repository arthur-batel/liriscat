import functools
import itertools
import warnings
from abc import ABC, abstractmethod

import numba
import numpy as np
from itertools import chain
from stat import S_IREAD

from sklearn.metrics import roc_auc_score
from torch.cuda import device

from liriscat import utils
from liriscat import dataset
from liriscat import CDM
import torch
import logging
from torch.utils import data
from tqdm import tqdm


class AbstractSelectionStrategy(ABC):
    def __init__(self, name: str = None, **config):
        super().__init__()

        utils.set_seed(config['seed'])

        self.config = config
        self.device = config['device']
        self._name = name
        self.CDM = None
        self.model = None
        self.state = None
        self._trained = False
        self.fold = 0

        if self.config['verbose_early_stopping']:
            # Decide on the early stopping criterion
            match self.config['esc']:
                case 'error':
                    self._train_method = self._verbose_train_early_stopping_error
        else:
            match self.config['esc']:
                case 'error':
                    self._train_method = self._train_early_stopping_error

        self.pred_metrics = config['pred_metrics'] if config['pred_metrics'] else ['rmse', 'mae']
        self.pred_metric_functions = {
            'rmse': utils.root_mean_squared_error,
            'mae': utils.mean_absolute_error,
            'r2': utils.r2,
            'mi_acc': utils.micro_ave_accuracy,
            'mi_prec': utils.micro_ave_precision,
            'mi_rec': utils.micro_ave_recall,
            'mi_f1': utils.micro_ave_f1,
            'mi_auc': utils.micro_ave_auc,
        }
        assert set(self.pred_metrics).issubset(self.pred_metric_functions.keys())

        self.profile_metrics = config['profile_metrics'] if config['profile_metrics'] else ['pc-er', 'doa']
        self.profile_metric_functions = {
            'pc-er': utils.compute_pc_er,
            'doa': utils.compute_doa,
            'rm': utils.compute_rm,
        }
        assert set(self.profile_metrics).issubset(self.profile_metric_functions.keys())

        match config['valid_metric']:
            case 'rmse':
                self.valid_metric = utils.root_mean_squared_error
                self.metric_sign = 1  # Metric to minimize :1; metric to maximize :-1
            case 'mae':
                self.valid_metric = utils.mean_absolute_error
                self.metric_sign = 1
            case 'mi_acc':
                self.valid_metric = utils.micro_ave_accuracy
                self.metric_sign = -1

        match config['CDM']:
            case 'impact':
                self.CDM = CDM.IMPACT(**config)

    @property
    def name(self):
        return f'{self._name}_cont_model'

    @name.setter
    def name(self, new_value):
        self._name = new_value

    @abstractmethod
    def _loss_function(self, user_ids, question_ids, categories, labels):
        raise NotImplementedError

    @abstractmethod
    def update_params(self, user_ids, question_ids, labels, categories):
        raise NotImplementedError

    @abstractmethod
    def select_action(self, t, env):
        raise NotImplementedError

    @abstractmethod
    def get_params(self):
        raise NotImplementedError

    @abstractmethod
    def _save_model_params(self):
        raise NotImplementedError

    def evaluation_state(func):
        """
        Temporary set the model state to "eval"
        """

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Extract 'self' from the first positional argument
            self = args[0] if args else None
            if self is None:
                raise ValueError("Decorator 'evaluation_state' requires to be used on instance methods.")

            # Store the previous state
            prev_state = getattr(self, 'state', None)

            try:
                # Set the state to 'eval' before method execution
                self.state = "eval"
                # Call the actual method
                self.CDM.model.eval()
                self.model.eval()
                with torch.no_grad(), torch.amp.autocast('cuda'):
                    result = func(*args, **kwargs)
            finally:
                # Restore the previous state after method execution
                self.CDM.model.train()
                self.model.train()
                self.state = prev_state

            return result

        return wrapper

    def evaluation_param(func):
        """
        Temporary change precomputed self.model.R and self.model.ir_idx params
        """

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Extract 'self' from the first positional argument
            self = args[0] if args else None
            if self is None:
                raise ValueError("Decorator 'evaluation_param' requires to be used on instance methods.")

            # Store the previous state
            prev_state = (self.CDM.model.R, self.CDM.model.ir_idx)

            try:
                # Set the state to 'eval' before method execution
                self.CDM.model.R = self.CDM.model.R_valid
                self.CDM.model.ir_idx = self.CDM.model.ir_idx_valid
                # Call the actual method
                result = func(*args, **kwargs)
            finally:
                # Restore the previous state after method execution
                self.CDM.model.R = prev_state[0]
                self.CDM.model.ir_idx = prev_state[1]

            return result

        return wrapper

    @evaluation_param
    @evaluation_state
    def evaluate_valid(self, valid_loader: dataset.CATDataset):
        """
        Evaluate the model on the given data using the given metrics.
        """
        loss_list = []
        pred_list = []
        label_list = []

        for batch_users_env in valid_loader:

            # Prepare the meta set
            m_user_ids, m_question_ids, m_labels, m_category_ids = batch_users_env.generate_IMPACT_meta()

            for t in range(self.config['n_query']):

                # Select the action (question to submit)
                actions = self.select_action(t, batch_users_env)

                batch_users_env.update(actions, t)

            with torch.enable_grad():
                self.CDM.model.train()
                self.CDM.update_users(batch_users_env.feed_IMPACT_query())
                self.CDM.model.eval()

            preds = self.CDM.model(m_user_ids, m_question_ids, m_category_ids)

            total_loss = self.CDM._compute_loss(m_user_ids, m_question_ids, m_labels.int(), m_category_ids)
            loss_list.append(total_loss.detach())

            pred_list.append(preds)
            label_list.append(m_labels)

            pred_tensor = torch.cat(pred_list)
            label_tensor = torch.cat(label_list)
            mean_loss = torch.mean(torch.stack(loss_list))

        return mean_loss, self.valid_metric(pred_tensor, label_tensor)

    @evaluation_param
    @evaluation_state
    def evaluate_test(self, test_data: dataset.CATDataset):
        """CATDataset
        Evaluate the model on the given data using the given metrics.
        """

        test_loader = data.DataLoader(test_data, collate_fn=dataset.CustomCollate(test_data), batch_size=10000,
                                      shuffle=False, pin_memory=False)

        pred_list = {t : [] for t in range(self.config['n_query'])}
        label_list = {t : [] for t in range(self.config['n_query'])}
        emb_tensor = torch.zeros(size = (test_data.n_actual_users, self.config['n_query'], test_data.n_categories), device=self.device)

        log_idx = 0
        for batch_users_env in test_loader:

            # Prepare the meta set
            m_user_ids, m_question_ids, m_labels, m_category_ids = batch_users_env.generate_IMPACT_meta()

            for t in range(self.config['n_query']):

                # Select the action (question to submit)
                actions = self.select_action(t, batch_users_env)

                batch_users_env.update(actions, t)

                with torch.enable_grad():
                    self.CDM.model.train()
                    self.CDM.update_users(batch_users_env.feed_IMPACT_query())
                    self.CDM.model.eval()

                preds = self.CDM.model(m_user_ids, m_question_ids, m_category_ids)

                pred_list[t].append(preds)
                label_list[t].append(m_labels)
                emb_tensor[log_idx:log_idx+batch_users_env.query_users.shape[0],t,:] = self.CDM.get_user_emb()[batch_users_env.query_users,:]

            log_idx += batch_users_env.query_users.shape[0]

        # Compute metrics in one pass using a dictionary comprehension
        results_pred = {t : {metric: self.pred_metric_functions[metric](torch.cat(pred_list[t]), torch.cat(label_list[t])).cpu().item()
                   for metric in self.pred_metrics} for t in range(self.config['n_query'])}

        results_profiles = {t : {metric: self.profile_metric_functions[metric](emb_tensor[:,t,:], test_data)
                   for metric in self.profile_metrics} for t in range(self.config['n_query'])}

        return results_pred, results_profiles

    def train(self, train_loader: dataset.CATDataset, valid_loader: dataset.evalDataset):
        """Train the model."""

        lr = self.config['learning_rate']
        batch_size = self.config['batch_size']
        device = self.config['device']

        torch.cuda.empty_cache()

        match self.config['CDM']:
            case 'impact':
                self.CDM.init_model(train_loader, valid_loader)
                self.CDM.model.to(device, non_blocking=True)
                if hasattr(torch, "compile"):
                    self.CDM.model = torch.compile(self.CDM.model)

        self.model.to(device, non_blocking=True)
        if hasattr(torch, "compile"):
            self.model = torch.compile(self.model)

        logging.info('train on {}'.format(device))
        logging.info("-- START Training --")

        self.best_epoch = 0
        self.best_valid_loss = float('inf')
        self.best_valid_metric = self.metric_sign * float('inf')

        self.best_S_params = self.get_params()
        self.best_CDM_params = self.CDM.get_params()

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        # Reduce the learning rate when a metric has stopped improving
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=2, factor=0.5)
        self.scaler = torch.amp.GradScaler(self.device)

        self.model.train()
        self._train_method(train_loader, valid_loader)

        self._trained = True

        logging.info("-- END Training --")

        if self.config['save_params']:
            self._save_model_params(temporary=False)
            logging.info("Params saved")
        self.fold += 1

    def _train_early_stopping_error(self, train_loader, valid_loader):

        epochs = self.config['num_epochs']
        eval_freq = self.config['eval_freq']
        patience = self.config['patience']

        valid_loader.split_query_meta(self.config['seed']) # split valid query qnd meta set one and for all epochs

        for _, ep in tqdm(enumerate(range(epochs + 1)), total=epochs, disable=self.config['disable_tqdm']):

            train_loader.set_query_seed(ep) # changes the meta and query set for each epoch

            for batch_users_env in train_loader:

                m_user_ids, m_question_ids, m_labels, m_category_ids = batch_users_env.generate_IMPACT_meta()

                for t in range(self.config['n_query']):

                    actions = self.select_action(t, batch_users_env)

                    batch_users_env.update(actions, t)

                    self.CDM.update_users(batch_users_env.feed_IMPACT_query())
                    self.update_params(m_user_ids, m_question_ids, m_labels, m_category_ids)

                self.CDM.update_params(m_user_ids, m_question_ids, m_labels, m_category_ids)

            # Early stopping
            if (ep + 1) % eval_freq == 0:
                with torch.no_grad(), torch.amp.autocast('cuda'):
                    valid_loss, valid_metric = self.evaluate_valid(valid_loader)

                    logging.info(f'{valid_metric}, {valid_loss}')

                    with warnings.catch_warnings(record=True) as w:
                        warnings.simplefilter("always")

                    # Checking loss improvement
                    if self.metric_sign * self.best_valid_metric > self.metric_sign * valid_metric:  # (self.best_valid_metric - valid_rmse) / abs(self.best_valid_metric) > 0.001:
                        self.best_epoch = ep
                        self.best_valid_metric = valid_metric
                        self.best_model_params = self.model.state_dict()

                        self.scheduler.step(valid_loss)

                    if ep - self.best_epoch >= patience:
                        break

        self.model.load_state_dict(self.best_model_params)

