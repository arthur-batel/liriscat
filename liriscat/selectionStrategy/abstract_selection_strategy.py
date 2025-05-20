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
from torch.utils.data import DataLoader


from liriscat import utils
from liriscat import dataset
from liriscat import CDM
import torch
import logging
from torch.utils import data
from tqdm import tqdm

from liriscat.dataset import UserCollate, QueryEnv


class AbstractSelectionStrategy(ABC):
    def __init__(self, name: str = None, metadata=None, **config):
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
        self.trainable = True
        self._rng = torch.Generator(device=config['device']).manual_seed(config['seed'])

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
                self.CDM = CDM.CATIMPACT(**config)
            case 'irt':
                irt_config = utils.convert_config_to_EduCAT(config, metadata)
                self.CDM = CDM.CATIRT(**irt_config)

    @property
    def name(self):
        return f'{self._name}_cont_model'

    @name.setter
    def name(self, new_value):
        self._name = new_value

    @abstractmethod
    def _loss_function(self, user_ids, question_ids, categories, labels):
        raise NotImplementedError

    def update_S_params(self, user_ids, question_ids, labels, category_ids):
        with torch.enable_grad():
            logging.debug("- Update S params ")
            self.model.train()

            self.S_optimizer.zero_grad()

            loss = self._loss_function(user_ids, question_ids, labels, category_ids)
            
            self.S_scaler.scale(loss).backward()
            self.S_scaler.step(self.S_optimizer)
            self.S_scaler.update()

            self.model.eval()

    def update_CDM_params(self, user_ids, question_ids, labels, category_ids):
        with torch.enable_grad():
            logging.debug("- Update CDM params ")
            self.model.train()

            for t in range(self.config['num_inner_epochs']) :

                self.CDM_optimizer.zero_grad()
    
                loss = self.CDM._compute_loss(user_ids, question_ids, category_ids, labels )
                
                self.CDM_scaler.scale(loss).backward()
                self.CDM_scaler.step(self.CDM_optimizer)
                self.CDM_scaler.update()

            self.model.eval()

    def update_users(self,query_data, meta_data, meta_labels) :
        with torch.enable_grad():
            logging.debug("- Update users ")
            self.CDM.model.train()
            m_user_ids, m_question_ids, m_category_ids = meta_data
    
            data = dataset.SubmittedDataset(query_data)
            dataloader = DataLoader(data, batch_size=2048, shuffle=True, num_workers=0)
    
            user_params_optimizer = torch.optim.Adam(self.CDM.model.users_emb.parameters(),
                                                          lr=self.CDM.config[
                                                              'inner_user_lr'])  # todo : Decide How to use a scheduler
    
            user_params_scaler = torch.amp.GradScaler(self.CDM.config['device'])
    
            n_batches = len(dataloader)
    
            for t in range(self.CDM.config['num_inner_users_epochs']) :
    
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
    
                    self.CDM.model.eval()
                    with torch.no_grad() :                    
                        preds = self.CDM.model(user_ids, question_ids, category_ids)
                        sum_acc_0 += utils.micro_ave_accuracy(labels, preds)

                        meta_preds = self.CDM.model(m_user_ids, m_question_ids, m_category_ids)
    
                    self.CDM.model.train()
                    user_params_optimizer.zero_grad()
                    
                    with torch.amp.autocast('cuda'):
                        loss = self.CDM._compute_loss(user_ids, question_ids, labels, category_ids)
                        sum_loss_0 += loss.item()

                    user_params_scaler.scale(loss).backward()
                    user_params_scaler.step(user_params_optimizer)
                    user_params_scaler.update()

                    self.CDM.model.eval()
                    with torch.no_grad(), torch.amp.autocast('cuda'):
                        loss2 = self.CDM._compute_loss(user_ids, question_ids, labels, category_ids)
                        sum_loss_1 += loss2.item()
    
                    preds = self.CDM.model(user_ids, question_ids, category_ids)
    
                    sum_acc_1 += utils.micro_ave_accuracy(labels, preds)
                    sum_meta_acc += utils.micro_ave_accuracy(meta_labels, meta_preds)
                    
                    with torch.no_grad(),torch.amp.autocast('cuda') :
                        meta_loss = self.CDM._compute_loss(m_user_ids, m_question_ids, meta_labels, m_category_ids)
                        sum_meta_loss += meta_loss.item()
                    
                logging.debug(
                    f'inner epoch {t} - query loss_0 : {sum_loss_0/n_batches:.5f} '
                    f'- query loss_1 : {sum_loss_1/n_batches:.5f} '
                    f'- query acc 0 : {sum_acc_0/n_batches:.5f} '
                    f'- query acc 1 : {sum_acc_1/n_batches:.5f} '
                    f'- meta acc 0 : {sum_meta_acc/n_batches:.5f}'
                    f'- meta loss 1 : {sum_meta_loss:.5f}'
                )
            self.CDM.model.eval()

    @abstractmethod
    def select_action(self, options_dict):
        #return the indice of the question to submit
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
                self.model.eval() # todo : putting in eval mode again
                with torch.no_grad(), torch.amp.autocast('cuda'):
                    result = func(*args, **kwargs)
            finally:
                # Restore the previous state after method execution
                self.CDM.model.train()
                self.model.train() # todo : putting in eval mode again
                self.state = prev_state

            return result

        return wrapper

    @evaluation_state
    def evaluate_valid(self, valid_loader: DataLoader, valid_query_env: QueryEnv):
        """
        Evaluate the model on the given data using the given metrics.
        """
        loss_list = []
        pred_list = []
        label_list = []

        logging.debug("-- evaluate valid --")

        for batch in valid_loader:

            valid_query_env.load_batch(batch)

            # Prepare the meta set
            m_user_ids, m_question_ids, m_labels, m_category_ids = valid_query_env.generate_IMPACT_meta()

            for t in range(self.config['n_query']):

                # Select the action (question to submit)
                actions = self.select_action(valid_query_env.get_query_options(t))

                valid_query_env.update(actions, t)

                self.update_users(valid_query_env.feed_IMPACT_sub(),(m_user_ids, m_question_ids, m_category_ids),m_labels)


            preds = self.CDM.model(m_user_ids, m_question_ids, m_category_ids)
            total_loss = self.CDM._compute_loss(m_user_ids, m_question_ids, m_labels.int(), m_category_ids)

            loss_list.append(total_loss.detach())
            pred_list.append(preds)
            label_list.append(m_labels)

        pred_tensor = torch.cat(pred_list)
        label_tensor = torch.cat(label_list)
        mean_loss = torch.mean(torch.stack(loss_list))

        return mean_loss, self.valid_metric(pred_tensor, label_tensor)
    

    @evaluation_state
    def evaluate_test(self, test_dataset: dataset.EvalDataset):
        """CATDataset
        Evaluate the model on the given data using the given metrics.
        """
        logging.debug("-- evaluate test --")
        
        test_dataset.split_query_meta(self.config['seed'])

        assert self.CDM.initialized_users_prior, \
            f'Users\' embedding and CDM regularization need to be initialized with the aposteriori distribution.'
        self.CDM.init_test(test_dataset)

        test_query_env = QueryEnv(test_dataset, self.device, self.config['valid_batch_size'])
        test_loader = data.DataLoader(test_dataset, collate_fn=dataset.UserCollate(test_query_env), batch_size=self.config['valid_batch_size'],
                                      shuffle=False, pin_memory=False, num_workers=0)

        pred_list = {t : [] for t in range(self.config['n_query'])}
        label_list = {t : [] for t in range(self.config['n_query'])}
        emb_tensor = torch.zeros(size = (test_dataset.n_actual_users, self.config['n_query'], test_dataset.n_categories), device=self.device)

        log_idx = 0
        for batch in test_loader:

            test_query_env.load_batch(batch)

            # Prepare the meta set
            m_user_ids, m_question_ids, m_labels, m_category_ids = test_query_env.generate_IMPACT_meta()

            for t in tqdm(range(self.config['n_query']), total=self.config['n_query'], disable=self.config['disable_tqdm']):

                # Select the action (question to submit)
                options = test_query_env.get_query_options(t)
                actions = self.select_action(options)
                test_query_env.update(actions, t)

                self.update_users(test_query_env.feed_IMPACT_sub(),(m_user_ids, m_question_ids, m_category_ids),m_labels)

                with torch.no_grad() :
                    preds = self.CDM.model(m_user_ids, m_question_ids, m_category_ids)

                pred_list[t].append(preds)
                label_list[t].append(m_labels)
                emb_tensor[log_idx:log_idx+test_query_env.current_batch_size, t, :] = self.CDM.get_user_emb()[test_query_env.support_users_vec, :]

            log_idx += test_query_env.current_batch_size

        # Compute metrics in one pass using a dictionary comprehension
        results_pred = {t : {metric: self.pred_metric_functions[metric](torch.cat(pred_list[t]), torch.cat(label_list[t])).cpu().item()
                   for metric in self.pred_metrics} for t in range(self.config['n_query'])}

        results_profiles = {t : {metric: self.profile_metric_functions[metric](emb_tensor[:,t,:], test_dataset)
                   for metric in self.profile_metrics} for t in range(self.config['n_query'])}

        return results_pred, results_profiles

    def init_models(self, train_dataset: dataset.CATDataset, valid_dataset: dataset.EvalDataset):

        logging.debug("-- Initialize CDM and Selection strategy--")
        
        match self.config['CDM']:
            case 'impact':
                self.CDM.init_model(train_dataset, valid_dataset)

                if hasattr(torch, "compile"):
                    print("compiling CDM model")
                    self.CDM.model = torch.compile(self.CDM.model)

        if hasattr(torch, "compile") and not getattr(self, "model_compiled", False):
            print("compiling selection model")
            if isinstance(self.model, torch.nn.Module):
                self.model = torch.compile(self.model)
                self.model_compiled = True
            else:
                print("Selection model already compiled, skipping recompilation")
        else:
            print("Selection model already compiled, skipping recompilation")

        if hasattr(self.model, "to"):
            self.model.to(self.config['device'])
        else:
            print("Warning: self.model is a function and cannot be moved to device")

    def train(self, train_dataset: dataset.CATDataset, valid_dataset: dataset.EvalDataset):

        lr = self.config['learning_rate']
        batch_size = self.config['batch_size']
        device = self.config['device']

        torch.cuda.empty_cache()

        self.init_models(train_dataset, valid_dataset)

        logging.info('train on {}'.format(device))
        logging.info("-- START Training --")

        self.best_epoch = 0
        self.best_valid_loss = float('inf')
        self.best_valid_metric = self.metric_sign * float('inf')

        self.best_S_params = self.get_params()
        self.best_CDM_params = self.CDM.get_params()

        if self.trainable :
            self.S_optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
            self.S_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.S_optimizer, patience=2, factor=0.5)
            self.S_scaler = torch.amp.GradScaler(self.device)
        else : 
            self.S_optimizer = DummyOptimizer()
            self.S_scheduler = DummyScheduler()
            self.S_scaler = DummyScaler()
            
        self.CDM_optimizer = torch.optim.Adam(self.CDM.model.parameters(), lr=self.config['inner_lr'])
        self.CDM_scaler = torch.amp.GradScaler(self.config['device'])

        # self.meta_optimizer = torch.optim.Adam(self.CDM.model.parameters(), lr=self.config['inner_lr'])
        # self.CDM_scaler = torch.amp.GradScaler(self.config['device'])

        self.model.train()
        self.CDM.model.train()

        self._train_method(train_dataset, valid_dataset)

        self._trained = True

        logging.info("-- END Training --")

        if self.config['save_params']:
            self._save_model_params(temporary=False)
            logging.info("Params saved")
        self.fold += 1

    def _train_early_stopping_error(self, train_dataset, valid_dataset):

        epochs = self.config['num_epochs']
        eval_freq = self.config['eval_freq']
        patience = self.config['patience']

        valid_dataset.split_query_meta(self.config['seed']) # split valid query qnd meta set one and for all epochs

        train_query_env = QueryEnv(train_dataset, self.device, self.config['batch_size'])
        valid_query_env = QueryEnv(valid_dataset, self.device, self.config['valid_batch_size'])

        train_loader = DataLoader(dataset=train_dataset, collate_fn=UserCollate(train_query_env), batch_size=self.config['batch_size'],
                                  shuffle=True, pin_memory=False, num_workers=0)
        valid_loader = DataLoader(dataset=valid_dataset, collate_fn=UserCollate(valid_query_env),
                                  batch_size=self.config['valid_batch_size'],
                                  shuffle=False, pin_memory=False, num_workers=0)

        for _, ep in tqdm(enumerate(range(epochs + 1)), total=epochs, disable=self.config['disable_tqdm']):

            logging.debug(f'------- Epoch : {ep}')

            train_dataset.set_query_seed(ep) # changes the meta and query set for each epoch

            for u_batch, batch in enumerate(train_loader): # UserCollate directly load the data into the query environment
                
                logging.debug(f'----- User batch : {u_batch}')

                train_query_env.load_batch(batch)

                # Prepare the meta set
                m_user_ids, m_question_ids, m_labels, m_category_ids = train_query_env.generate_IMPACT_meta()

                for i_query in range(self.config['n_query']):

                    logging.debug(f'--- Query nb : {i_query}')

                    actions = self.select_action(train_query_env.get_query_options(i_query))
                    train_query_env.update(actions, i_query)

                    self.update_users(train_query_env.feed_IMPACT_sub(),(m_user_ids, m_question_ids, m_category_ids),m_labels )
                    
                self.update_S_params(m_user_ids, m_question_ids, m_labels, m_category_ids)
                #self.update_CDM_params(m_user_ids, m_question_ids, m_labels, m_category_ids)

            # Early stopping
            if (ep + 1) % eval_freq == 0:
                with torch.no_grad(), torch.amp.autocast('cuda'):
                    valid_loss, valid_metric = self.evaluate_valid(valid_loader, valid_query_env)

                    logging.info(f'valid_metric : {valid_metric}, valid_loss : {valid_loss}')

                    with warnings.catch_warnings(record=True) as w:
                        warnings.simplefilter("always")

                    # Checking loss improvement
                    if self.metric_sign * self.best_valid_metric > self.metric_sign * valid_metric:  # (self.best_valid_metric - valid_rmse) / abs(self.best_valid_metric) > 0.001:
                        self.best_epoch = ep
                        self.best_valid_metric = valid_metric
                        self.best_model_params = self.model.state_dict()

                        self.S_scheduler.step(valid_loss)

                    if ep - self.best_epoch >= patience:
                        break

        self.model.load_state_dict(self.best_model_params)

    def reset_rng(self):
        """
        Reset the random number generator to a new seed
        :param seed: new seed
        """
        self._rng = torch.Generator(device=self.config['device']).manual_seed(self.config['seed'])

class DummyOptimizer:
    def zero_grad(self): pass
    def step(self):      pass

class DummyScheduler:
    def step(self,loss): pass

class DummyScaler:
    def __init__(self): self._enabled = False
    def scale(self, loss): return loss
    def step(self, optimizer): pass
    def update(self): pass