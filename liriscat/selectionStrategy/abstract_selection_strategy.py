import functools
import itertools
import warnings
from abc import ABC, abstractmethod
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
            'rmse': root_mean_squared_error,
            'mae': mean_absolute_error,
            'r2': r2,
            'mi_acc': micro_ave_accuracy,
            'mi_prec': micro_ave_precision,
            'mi_rec': micro_ave_recall,
            'mi_f1': micro_ave_f1,
            'mi_auc': micro_ave_auc,
        }
        assert set(self.pred_metrics).issubset(self.pred_metric_functions.keys())

        self.profile_metrics = config['profile_metrics'] if config['profile_metrics'] else ['pc-er', 'doa']
        self.profile_metric_functions = {
            'pc-er': compute_pc_er,
            'doa': compute_doa,
            'rm': compute_rm,
        }
        assert set(self.profile_metrics).issubset(self.profile_metric_functions.keys())

        match config['valid_metric']:
            case 'rmse':
                self.valid_metric = root_mean_squared_error
                self.metric_sign = 1  # Metric to minimize :1; metric to maximize :-1
            case 'mae':
                self.valid_metric = mean_absolute_error
                self.metric_sign = 1
            case 'mi_acc':
                self.valid_metric = micro_ave_accuracy
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
            m_user_ids, m_question_ids, m_labels, m_category_ids = batch_users_env.feedIMPACT_meta()

            for t in range(self.config['n_query']):

                # Select the action (questio to submit)
                actions = self.select_action(t, batch_users_env)

                batch_users_env.update(actions, t)

            with torch.enable_grad():
                self.CDM.model.train()
                self.CDM.update_users(batch_users_env.query_user_ids, batch_users_env.query_question_ids, batch_users_env.query_labels, batch_users_env.query_category_ids)
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
        """
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
            m_user_ids, m_question_ids, m_labels, m_category_ids = batch_users_env.feedIMPACT_meta()

            for t in range(self.config['n_query']):


                # Select the action (question to submit)
                actions = self.select_action(t, batch_users_env)

                batch_users_env.update(actions, t)

                with torch.enable_grad():
                    self.CDM.model.train()
                    self.CDM.update_users(batch_users_env.query_user_ids, batch_users_env.query_question_ids,
                                          batch_users_env.query_labels, batch_users_env.query_category_ids)
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

    def train(self, train_data: dataset.CATDataset, valid_data: dataset.CATDataset):
        """Train the model."""

        lr = self.config['learning_rate']
        batch_size = self.config['batch_size']
        device = self.config['device']

        torch.cuda.empty_cache()

        match self.config['CDM']:
            case 'impact':
                self.CDM.init_model(train_data, valid_data)
                self.CDM.model.to(device, non_blocking=True)

        self.model.to(device, non_blocking=True)

        logging.info('train on {}'.format(device))
        logging.info("-- START Training --")

        self.best_epoch = 0
        self.best_valid_loss = float('inf')
        self.best_valid_metric = self.metric_sign * float('inf')

        self.best_S_params = self.get_params()
        self.best_CDM_params = self.CDM.get_params()

        train_loader = data.DataLoader(train_data, collate_fn=dataset.CustomCollate(train_data), batch_size=batch_size,
                                       shuffle=True, pin_memory=False)
        valid_loader = data.DataLoader(valid_data, collate_fn=dataset.CustomCollate(valid_data), batch_size=10000,
                                       shuffle=False, pin_memory=False)

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

        if hasattr(torch, "compile"):
            self.model = torch.compile(self.model)

        for _, ep in tqdm(enumerate(range(epochs + 1)), total=epochs, disable=self.config['disable_tqdm']):

            for batch_users_env in train_loader:

                m_user_ids, m_question_ids, m_labels, m_category_ids = batch_users_env.feedIMPACT_meta()

                for t in range(self.config['n_query']):

                    actions = self.select_action(t, batch_users_env)

                    batch_users_env.update(actions, t)

                    self.CDM.update_users(batch_users_env.query_user_ids, batch_users_env.query_question_ids, batch_users_env.query_labels, batch_users_env.query_category_ids)
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


def compute_pc_er(emb, test_data):
    U_resp_sum = torch.zeros(size=(test_data.n_users, test_data.n_categories)).to(test_data.raw_data_array.device,
                                                                                  non_blocking=True)
    U_resp_nb = torch.zeros(size=(test_data.n_users, test_data.n_categories)).to(test_data.raw_data_array.device,
                                                                                 non_blocking=True)

    data_loader = data.DataLoader(test_data, batch_size=1, shuffle=False)
    for data_batch in data_loader:
        user_ids = data_batch[:, 0].long()
        item_ids = data_batch[:, 1].long()
        labels = data_batch[:, 2]
        dim_ids = data_batch[:, 3].long()

        U_resp_sum[user_ids, dim_ids] += labels
        U_resp_nb[user_ids, dim_ids] += torch.ones_like(labels)

    U_ave = U_resp_sum / U_resp_nb

    return pc_er(test_data.n_categories, U_ave, emb).cpu().item()


@torch.jit.script
def pc_er(concept_n: int, U_ave: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
    """
    Compute the average correlation across dimensions between U_ave and emb.
    Both U_ave and emb should be [num_user, concept_n] tensors.
    concept_n: number of concepts
    U_ave: FloatTensor [num_user, concept_n]
    emb: FloatTensor [num_user, concept_n]

    Returns:
        c: A 0-dim tensor (scalar) representing the average correlation.
    """

    # Initialize counters as tensors to allow JIT compatibility
    c = torch.tensor(0.0, device=U_ave.device)
    s = torch.tensor(0.0, device=U_ave.device)

    for dim in range(concept_n):
        mask = ~torch.isnan(U_ave[:, dim])
        U_dim_masked = U_ave[:, dim][mask].unsqueeze(1)
        Emb_dim_masked = emb[:, dim][mask].unsqueeze(1)

        corr = torch.corrcoef(torch.concat([U_dim_masked, Emb_dim_masked], dim=1).T)[0][1]

        if not torch.isnan(corr):
            c += corr
            s += 1
    c /= s
    return c


def compute_rm(emb: torch.Tensor, test_data):
    concept_array, concept_lens = utils.preprocess_concept_map(test_data.concept_map)
    return utils.compute_rm_fold(emb.cpu().numpy(), test_data.raw_data, concept_array, concept_lens)


def compute_doa(emb: torch.Tensor, test_data):
    return np.mean(utils.evaluate_doa(emb.cpu().numpy(), test_data.log_tensor.cpu().numpy(), test_data.metadata,
                                      test_data.concept_map))


@torch.jit.script
def root_mean_squared_error(y_true, y_pred):
    """
    Compute the rmse metric (Regression)

    Args:
        y_true (Tensor): Ground truth labels.
        y_pred (Tensor): Predicted labels.

    Returns:
        Tensor: The rmse metric.
    """
    return torch.sqrt(torch.mean(torch.square(y_true - y_pred)))


@torch.jit.script
def mean_absolute_error(y_true, y_pred):
    """
    Compute the mae metric (Regression)

    Args:
        y_true (Tensor): Ground truth labels.
        y_pred (Tensor): Predicted labels.

    Returns:
        Tensor: The mae metric.
    """
    return torch.mean(torch.abs(y_true - y_pred))


@torch.jit.script
def r2(gt, pd):
    """
    Compute the r2 metric (Regression)

    Args:
        y_true (Tensor): Ground truth labels.
        y_pred (Tensor): Predicted labels.

    Returns:
        Tensor: The r2 metric.
    """

    mean = torch.mean(gt)
    sst = torch.sum(torch.square(gt - mean))
    sse = torch.sum(torch.square(gt - pd))

    r2 = 1 - sse / sst

    return r2


@torch.jit.script
def micro_ave_accuracy(y_true, y_pred):
    """
    Compute the micro-averaged accuracy (Classification)

    Args:
        y_true (Tensor): Ground truth labels.
        y_pred (Tensor): Predicted labels.

    Returns:
        Tensor: The micro-averaged precision.
    """
    return torch.mean((y_true == y_pred).float())


@torch.jit.script
def micro_ave_precision(y_true, y_pred):
    """
    Compute the micro-averaged precision (Binary classification)

    Args:
        y_true (Tensor): Ground truth labels.
        y_pred (Tensor): Predicted labels.

    Returns:
        Tensor: The micro-averaged precision.
    """
    true_positives = torch.sum((y_true == 2) & (y_pred == 2)).float()
    predicted_positives = torch.sum(y_pred == 2).float()
    return true_positives / predicted_positives


@torch.jit.script
def micro_ave_recall(y_true, y_pred):
    """
    Compute the micro-averaged recall (Binary classification)

    Args:
        y_true (Tensor): Ground truth labels.
        y_pred (Tensor): Predicted labels.

    Returns:
        Tensor: The micro-averaged recall.
    """
    true_positives = torch.sum((y_true == 2) & (y_pred == 2)).float()
    actual_positives = torch.sum(y_true == 2).float()
    return true_positives / actual_positives


@torch.jit.script
def micro_ave_f1(y_true, y_pred):
    """
    Compute the micro-averaged f1 (Binary classification)

    Args:
        y_true (Tensor): Ground truth labels.
        y_pred (Tensor): Predicted labels.

    Returns:
        Tensor: The micro-averaged f1.
    """
    precision = micro_ave_precision(y_true, y_pred)
    recall = micro_ave_recall(y_true, y_pred)
    return 2 * (precision * recall) / (precision + recall)


def micro_ave_auc(y_true, y_pred):
    """
    Compute the micro-averaged roc-auc (Binary classification)

    Args:
        y_true (Tensor): Ground truth labels.
        y_pred (Tensor): Predicted labels.

    Returns:
        Tensor: The micro-averaged roc-auc.
    """
    y_true = y_true.cpu().int().numpy()
    y_pred = y_pred.cpu().int().numpy()
    roc_auc = roc_auc_score(y_true.ravel(), y_pred.ravel(), average='micro')
    return torch.tensor(roc_auc)
