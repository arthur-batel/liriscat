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

        # Reproductibility 
        utils.set_seed(config['seed'])
        self._rng = torch.Generator(device=config['device']).manual_seed(config['seed'])

        # Configuration
        self.config = config
        self.device = config['device']
        self._name = name

        # States
        self.state = None
        self._trained = False
        self.fold = 0
        self.trainable = True

        # Algorithmic components
        self.CDM = None
        self.model = None
        self.inner_loop = None
        self._train_method = None

        # Parameters
        self.meta_params = torch.nn.parameter.Parameter(torch.empty(1,metadata['num_dimension_id'])) # Parameters of the meta trainer

        # Metrics
        self.pred_metrics = config['pred_metrics'] if config['pred_metrics'] else ['rmse', 'mae']
        self.profile_metrics = config['profile_metrics'] if config['profile_metrics'] else ['pc-er', 'doa']
        self.valid_metric = None
        self.metric_sign = None

        # Initialization of algorithmic components, parameters and metrics
        ## Parameters init
        torch.nn.init.normal_(self.meta_params).to(self.config['device'])

        ## Algrithmic component init
        if self.config['verbose_early_stopping']:
            # Decide on the early stopping criterion
            match self.config['esc']:
                case 'error':
                    self._train_method = self._verbose_train_early_stopping_error
        else:
            match self.config['esc']:
                case 'error':
                    self._train_method = self._train_early_stopping_error

        match config['CDM']:
            case 'impact':
                self.CDM = CDM.CATIMPACT(**config)
            case 'irt':
                irt_config = utils.convert_config_to_EduCAT(config, metadata)
                self.CDM = CDM.CATIRT(**irt_config)

        match config['meta_trainer']:
            case 'GAP':
                self.inner_loop = self.GAP_inner_loop
            case 'Adam':
                self.inner_loop = self.Adam_inner_loop
            case 'none':
                self.inner_loop = None
            case _:
                raise ValueError(f"Unknown meta trainer: {config['meta_trainer']}")

        ## Metric init
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

    @property
    def name(self):
        return f'{self._name}_cont_model'

    @name.setter
    def name(self, new_value):
        self._name = new_value

    @abstractmethod
    def _loss_function(self, users_id, items_id, concepts_id, labels):
        raise NotImplementedError

    def update_S_params(self, users_id, items_id, labels, concepts_id):
        with torch.enable_grad():
            logging.debug("- Update S params ")
            self.model.train()

            self.S_optimizer.zero_grad()

            loss = self._loss_function(users_id, items_id, labels, concepts_id)
            
            self.S_scaler.scale(loss).backward()
            self.S_scaler.step(self.S_optimizer)
            self.S_scaler.update()

            self.model.eval()

    def update_CDM_params(self, users_id, items_id, labels, concepts_id):
        with torch.enable_grad():
            logging.debug("- Update CDM params ")
            self.model.train()

            for t in range(self.config['num_inner_epochs']) :

                self.CDM_optimizer.zero_grad()
    
                loss = self.CDM._compute_loss(users_id, items_id, concepts_id, labels )
                
                self.CDM_scaler.scale(loss).backward()
                self.CDM_scaler.step(self.CDM_optimizer)
                self.CDM_scaler.update()

            self.model.eval()

    def GAP_inner_loop(self, query_data, meta_data):

        def take_step(users_id, questions_id, labels, categories_id):

            with torch.amp.autocast('cuda'):
                loss = self.CDM._compute_loss(users_id, questions_id, labels, categories_id)

                grads = torch.autograd.grad(loss, self.CDM.model.users_emb().parameters(), create_graph=False)
                preconditioner = (torch.nn.Softplus(beta=2)(self.meta_params).to("cuda").repeat(grads[0].shape[0],1) * grads[0])
            
                with torch.no_grad():
                    self.CDM.model.users_emb.weight.data -= self.config["inner_user_lr"] * preconditioner

                return loss


        self.abstract_inner_loop(query_data, meta_data, take_step)

    def Adam_inner_loop(self, query_data, meta_data):

        def take_step(users_id, questions_id, labels, categories_id):
            self.user_params_optimizer.zero_grad()
            if self.user_params_scaler is not None:
                with torch.amp.autocast(device_type='cuda'):
                    loss = self.CDM._compute_loss(users_id, questions_id)
                self.user_params_scaler.scale(loss).backward()
                self.user_params_scaler.step(self.user_params_optimizer)
                self.user_params_scaler.update()
            else:
                loss = self.CDM._compute_loss(users_id, questions_id)
                loss.backward()
                self.user_params_optimizer.step()
            return loss

        self.abstract_inner_loop(query_data, meta_data, take_step)


    def abstract_inner_loop(self,query_data, meta_data, optimizer):
        """
            User parameters update (theta) on the query set
        """
        logging.debug("- Update users ")

        with torch.enable_grad():


            sub_data = dataset.SubmittedDataset(query_data)
            sub_dataloader = DataLoader(sub_data, batch_size=2048, shuffle=True, num_workers=0)
            n_batches = len(sub_dataloader)
    
            for t in range(self.CDM.config['num_inner_users_epochs']) :
    
                sum_loss_0 = sum_loss_1 = sum_acc_0 = sum_acc_1 = sum_meta_acc = sum_meta_loss = 0
    
                for batch in sub_dataloader:
                    users_id, items_id, labels, concepts_id = batch["user_ids"], batch["question_ids"], batch["labels"], batch["category_ids"]
    
                    self.CDM.model.eval()
                    with torch.no_grad() , torch.amp.autocast('cuda'):                    
                        preds = self.CDM.model(users_id, items_id, concepts_id)
                        sum_acc_0 += utils.micro_ave_accuracy(labels, preds)
                        meta_preds = self.CDM.model(users_id=meta_data['users_id'], items_id=meta_data['questions_id'], concepts_id=meta_data['categories_id'])
                        loss2 = self.CDM._compute_loss(users_id=users_id, items_id=items_id)
                        sum_loss_1 += loss2.item()
    
                    self.CDM.model.train()
                    loss = optimizer(users_id, items_id, labels, concepts_id)
                    sum_loss_0 += loss.item()
                    self.CDM.model.eval()

                    with torch.no_grad(), torch.amp.autocast('cuda'):
                        preds = self.CDM.model(users_id, items_id, concepts_id)
                        sum_acc_1 += utils.micro_ave_accuracy(labels, preds)
                        sum_meta_acc += utils.micro_ave_accuracy(meta_data['labels'], meta_preds)
                        meta_loss = self.CDM._compute_loss(users_id=meta_data['users_id'], items_id=meta_data['questions_id'])
                        sum_meta_loss += meta_loss.item()
                    
                logging.debug(
                    f'- query loss_1 : {sum_loss_1/n_batches:.5f} '
                    f'- query acc 0 : {sum_acc_0/n_batches:.5f} '
                    f'- query acc 1 : {sum_acc_1/n_batches:.5f} '
                    f'- meta acc 0 : {sum_meta_acc/n_batches:.5f}'
                    f'- meta loss 1 : {sum_meta_loss:.5f}'
                )

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
            meta_data = valid_query_env.generate_IMPACT_meta()

            for t in range(self.config['n_query']):

                # Select the action (question to submit)
                actions = self.select_action(valid_query_env.get_query_options(t))

                valid_query_env.update(actions, t)

                self.inner_loop(valid_query_env.feed_IMPACT_sub(),meta_data)

            preds = self.CDM.model(users_id=meta_data['users_id'], items_id=meta_data['questions_id'], concepts_id=meta_data['categories_id'])
            total_loss = self.CDM._compute_loss(users_id=meta_data['users_id'],items_id=meta_data['questions_id'],concepts_id=meta_data['categories_id'], labels=meta_data['labels'].int())

            loss_list.append(total_loss.detach())
            pred_list.append(preds)
            label_list.append(meta_data['labels'])

        pred_tensor = torch.cat(pred_list)
        label_tensor = torch.cat(label_list)
        mean_loss = torch.mean(torch.stack(loss_list))

        return mean_loss, self.valid_metric(pred_tensor, label_tensor)
    

    @evaluation_state
    def evaluate_test(self, test_dataset: dataset.EvalDataset):
        """CATDataset
        Evaluate the model on the given data using the given metrics.
        """
        logging.debug("-- Evaluate test --")

        # Device cleanup
        torch.cuda.empty_cache()
        logging.info('train on {}'.format(self.config['device']))
        
        # Initialize testing config
        match self.config['meta_trainer']:
            case 'GAP':
                pass
            case 'Adam':
                self.user_params_optimizer = torch.optim.Adam(self.CDM.model.users_emb.parameters(),
                                                    lr=self.CDM.config[
                                                        'inner_user_lr'])  # todo : Decide How to use a scheduler
                self.user_params_scaler = torch.amp.GradScaler(self.CDM.config['device'])
            case 'none':
                pass
            case _:
                raise ValueError(f"Unknown meta trainer: {self.config['meta_trainer']}")

        assert self.CDM.initialized_users_prior, \
            f'Users\' embedding and CDM regularization need to be initialized with the aposteriori distribution.'
        
        # Updating CDM to test dataset
        self.CDM.init_test(test_dataset)

        # Dataloaders preperation
        test_dataset.split_query_meta(self.config['seed'])
        test_query_env = QueryEnv(test_dataset, self.device, self.config['valid_batch_size'])
        test_loader = data.DataLoader(test_dataset, collate_fn=dataset.UserCollate(test_query_env), batch_size=self.config['valid_batch_size'],
                                      shuffle=False, pin_memory=False, num_workers=0)

        # Saving metric structures
        pred_list = {t : [] for t in range(self.config['n_query'])}
        label_list = {t : [] for t in range(self.config['n_query'])}
        emb_tensor = torch.zeros(size = (test_dataset.n_actual_users, self.config['n_query'], test_dataset.n_categories), device=self.device)

        # Test
        log_idx = 0
        for batch in test_loader:

            test_query_env.load_batch(batch)

            # Prepare the meta set
            meta_data = test_query_env.generate_IMPACT_meta()

            for t in tqdm(range(self.config['n_query']), total=self.config['n_query'], disable=self.config['disable_tqdm']):

                # Select the action (question to submit)
                options = test_query_env.get_query_options(t)
                actions = self.select_action(options)
                test_query_env.update(actions, t)

                self.inner_loop(test_query_env.feed_IMPACT_sub(),meta_data)

                with torch.no_grad() :
                    preds = self.CDM.model(meta_data['users_id'], meta_data['questions_id'], meta_data['categories_id'])

                pred_list[t].append(preds)
                label_list[t].append(meta_data['labels'])
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

        logging.info("-- START Training --")

        # Device cleanup
        torch.cuda.empty_cache()
        logging.info('train on {}'.format(self.config['device']))

        # Initialize training config
        lr = self.config['learning_rate']

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

        self.meta_optimizer = torch.optim.Adam([self.meta_params], lr=0.001) #todo : wrap in a correct module
        self.meta_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.meta_optimizer, patience=2, factor=0.5)
        self.meta_scaler = torch.amp.GradScaler(self.config['device'])

        match self.config['meta_trainer']:
            case 'GAP':
                pass
            case 'Adam':
                self.user_params_optimizer = torch.optim.Adam(self.CDM.model.users_emb.parameters(),
                                                    lr=self.CDM.config[
                                                        'inner_user_lr'])  # todo : Decide How to use a scheduler
                self.user_params_scaler = torch.amp.GradScaler(self.CDM.config['device'])
            case 'none':
                pass
            case _:
                raise ValueError(f"Unknown meta trainer: {self.config['meta_trainer']}")
            
        # Initialize early stopping parameters
        self.best_epoch = 0
        self.best_valid_loss = float('inf')
        self.best_valid_metric = self.metric_sign * float('inf')

        self.best_S_params = self.get_params()
        self.best_CDM_params = self.CDM.get_params()

        # Training mode
        self.model.train()
        self.CDM.model.train()

        # Train
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
                meta_data = train_query_env.generate_IMPACT_meta()

                for i_query in range(self.config['n_query']):

                    logging.debug(f'--- Query nb : {i_query}')

                    actions = self.select_action(train_query_env.get_query_options(i_query))
                    train_query_env.update(actions, i_query)

                    self.inner_loop(train_query_env.feed_IMPACT_sub(),meta_data)
                    
                self.update_S_params(users_id=meta_data['users_id'],
                                     items_id=meta_data['questions_id'],
                                     labels=meta_data['labels'],
                                     concepts_id=meta_data['categories_id'])
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