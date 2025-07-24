import functools
import itertools
import warnings
from abc import ABC, abstractmethod
from torch import nn
from liriscat.meta_models.meta_models import clone_state_dict, kl_divergence_gaussians, zero_grad
from liriscat.selectionStrategy.CovWeighting import CoVWeightingLoss
from torch.func import functional_call
import torch.nn.functional as F
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

def inverse_softplus(x):
    """
    Compute the inverse of the softplus function.
    
    For numerical stability, uses different approaches based on input magnitude:
    - For large x: log(x) (since softplus(y) ≈ y for large y)
    - For small x: log(exp(x) - 1) (direct inverse)
    
    Args:
        x: Input tensor
        
    Returns:
        Inverse softplus of x
    """
    return torch.where(x > 20, torch.log(x), torch.log(torch.expm1(x)))




class AbstractSelectionStrategy(ABC):
    def __init__(self, name: str = None, metadata=None, **config):
        logging.debug(f'------- Abstract model __init__()')
        super().__init__()

        # Reproductibility 
        utils.set_seed(config['seed'])
        self._rng = torch.Generator(device=config['device']).manual_seed(config['seed'])

        # Configuration
        self.config = config
        self.metadata = metadata
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
        self.inner_step = None
        self._train_method = None

        # Metrics
        self.pred_metrics = config['pred_metrics'] if config['pred_metrics'] else ['rmse', 'mae']
        self.profile_metrics = config['profile_metrics'] if config['profile_metrics'] else ['pc-er', 'doa']
        self.valid_metric = None
        self.metric_sign = None

        # Initialization of algorithmic components, parameters and metrics
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

        match self.config['CDM']:
            case 'impact':
                self.CDM = CDM.CATIMPACT(**self.config)
            case 'irt':
                irt_config = utils.convert_config_to_EduCAT(self.config, metadata)
                self.CDM = CDM.CATIRT(**irt_config)

        logging.debug(f'----- Meta trainer init : {self.config["meta_trainer"]}')
        match self.config['meta_trainer']:

            case 'GAP':
                pass
            case 'BETA-CD':
                pass
            case 'Approx_GAP':
                pass
            case 'MAML':
                pass                
            case 'Adam':
                self.inner_step = self.Adam_inner_step
                self.meta_params = None
                self.weights = torch.tensor([1.0, 1.0], device=self.config['device'])  # Initial weights for Adam inner step
            case 'none':
                self.inner_step = None
            case _:
                raise ValueError(f"Unknown meta trainer: {config['meta_trainer']}")
        
        self.L_W = torch.jit.script(CoVWeightingLoss(device=config['device']))

        ## Metric init
        self.pred_metric_functions = {
            'rmse': utils.root_mean_squared_error,
            'mae': utils.mean_absolute_error,
            'r2': utils.r2,
            'mi_acc': utils.micro_ave_accuracy,
            'mi_prec': utils.micro_ave_precision,
            'mi_rec': utils.micro_ave_recall,
            'mi_f_b': utils.micro_f_beta,
            'mi_auc': utils.micro_ave_auc,
            'ma_prec': utils.macro_precision,
            'ma_rec': utils.macro_recall,
            'ma_f_b': utils.macro_f_beta
        }
        assert set(self.pred_metrics).issubset(self.pred_metric_functions.keys())

        self.profile_metric_functions = {
            'pc-er': utils.compute_pc_er,
            'doa': utils.compute_doa,
            'rm': utils.compute_rm,
            'meta_doa': utils.compute_meta_doa,
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
    def _loss_function(self, users_id, items_id, concepts_id, labels,users_emb):
        raise NotImplementedError
    

    def update_S_params(self, loss):
        with torch.enable_grad():
            logging.debug("- Update S params ")
            self.model.train()
            
            self.S_scaler.scale(loss).backward()
            self.S_scaler.step(self.S_optimizer)
            self.S_scaler.update()

            self.S_optimizer.zero_grad()

            self.model.eval()

    def update_meta_params(self, loss):
        with torch.enable_grad():
            logging.debug("- Update meta params ")
            self.model.train()
            self.meta_optimizer.zero_grad()
            self.meta_scaler.scale(loss).backward()

            if self.config['debug'] : 
                if hasattr(self, 'meta_params') and self.meta_params is not None and self.meta_params.grad is not None:
                    logging.info(f"- meta_params gn: {[self.meta_params.grad[i,:].norm().item() for i in range(self.meta_params.shape[0])]}")
                if hasattr(self, 'cross_cond') and self.cross_cond is not None and self.cross_cond.grad is not None:
                    logging.info(f"- cross_cond gn: {[self.cross_cond.grad[i,:].norm().item() for i in range(self.cross_cond.shape[0])]}")
                if hasattr(self, 'meta_lambda') and self.meta_lambda is not None and self.meta_lambda.grad is not None:
                    logging.info(f"- meta_lambda gn: {self.meta_lambda.grad.norm().item()}")
                if hasattr(self, 'learning_users_emb') and self.learning_users_emb is not None and self.learning_users_emb.grad is not None:
                    logging.info(f"- learning_users_emb gn: {self.learning_users_emb.grad.norm().item()}")
                if hasattr(self, 'inner_lrs') and self.inner_lrs is not None and self.inner_lrs.grad is not None:
                    logging.info(f"- inner_lrs gn: {self.inner_lrs.grad.norm().item()}")
                if hasattr(self, 'user_mean') and self.user_mean is not None and self.user_mean.grad is not None:
                    logging.info(f"- user_mean gn: {self.user_mean.grad.norm().item()}")
                if hasattr(self, 'user_log_std') and self.user_log_std is not None and self.user_log_std.grad is not None:
                    logging.info(f"- user_log_std gn: {self.user_log_std.grad.norm().item()}")



            if self.config['debug'] : 
                if hasattr(self, 'meta_params') and self.meta_params is not None and self.meta_params.grad is not None:
                    logging.info(f"- meta_params norm: {[self.meta_params[i,:].norm().item() for i in range(self.meta_params.shape[0])]}")
                if hasattr(self, 'cross_cond') and self.cross_cond is not None and self.cross_cond.grad is not None:
                    logging.info(f"- cross_cond norm: {[self.cross_cond[i,:].norm().item() for i in range(self.cross_cond.shape[0])]}")
                if hasattr(self, 'meta_lambda') and self.meta_lambda is not None and self.meta_lambda.grad is not None:
                    logging.info(f"- meta_lambda norm: {self.meta_lambda.norm().item()}")
                if hasattr(self, 'learning_users_emb') and self.learning_users_emb is not None and self.learning_users_emb.grad is not None:
                    logging.info(f"- learning_users_emb norm: {self.learning_users_emb.norm().item()}")
                if hasattr(self, 'inner_norms') and self.inner_lrs is not None and self.inner_lrs.grad is not None:
                    logging.info(f"- inner_lrs norm: {self.inner_lrs.norm().item()}")
                if hasattr(self, 'user_mean') and self.user_mean is not None and self.user_mean is not None:
                    logging.info(f"- user_mean norm: {self.user_mean.norm().item()}")
                if hasattr(self, 'user_log_std') and self.user_log_std is not None and self.user_log_std is not None:
                    logging.info(f"- user_log_std norm: {self.user_log_std.norm().item()}")
                if hasattr(self, 'kl_weight') and self.kl_weight is not None and self.kl_weight.grad is not None:
                    logging.info(f"- kl_weight norm: {self.kl_weight.norm().item()}")

            # Print current learning rates
            for param_group in self.meta_optimizer.param_groups:
                logging.info(f"- {param_group['name']}: {param_group['lr']}")
            # Add gradient clipping for numerical stability
            if hasattr(self, 'meta_params') and self.meta_params is not None:
                torch.nn.utils.clip_grad_norm_(self.meta_params, max_norm=10.0)
            if hasattr(self, 'cross_cond') and self.cross_cond is not None:
                torch.nn.utils.clip_grad_norm_(self.cross_cond, max_norm=10.0)
            if hasattr(self, 'meta_lambda') and self.meta_lambda is not None:
                torch.nn.utils.clip_grad_norm_(self.meta_lambda, max_norm=1.0)
            if hasattr(self, 'inner_lrs') and self.inner_lrs is not None:
                torch.nn.utils.clip_grad_norm_(self.inner_lrs, max_norm=10.0)
                
            self.meta_scaler.step(self.meta_optimizer)
            self.meta_scaler.update()
            
            self.model.eval()

    def update_CDM_params(self, loss):
        with torch.enable_grad():
            logging.debug("- Update CDM params ")
            self.model.train()

            self.CDM_scaler.scale(loss).backward()
            self.CDM_scaler.step(self.CDM_optimizer)
            self.CDM_scaler.update()

            self.CDM_optimizer.zero_grad()

            self.model.eval()

    def GAP_inner_step(self, users_id, questions_id, labels, categories_id, learning_users_emb=None):
        """
        Meta-learning style inner step: returns updated user embeddings tensor (does not update model in-place).
        Args:
            users_id, questions_id, labels, categories_id: batch data
            users_emb: optional tensor to use as starting point (default: model's current embeddings)
        Returns:
            updated_users_emb: tensor of updated user embeddings (requires_grad)
            loss: loss on the query set
        """
        # 2. Forward pass: compute loss using the copied embeddings
        #    (Assume CDM._compute_loss can take a users_emb argument, else you need to adapt your model)
        L1, L3, R = self.CDM._compute_loss(users_id=users_id, items_id=questions_id, concepts_id=categories_id, labels=labels, learning_users_emb=learning_users_emb)

        # 3. Compute gradients w.r.t. the copied embeddings
        grads_L1 = torch.autograd.grad(L1, learning_users_emb, create_graph=False)
        grads_L3 = torch.autograd.grad(L3, learning_users_emb, create_graph=False)
        grads_R = torch.autograd.grad(R, learning_users_emb, create_graph=False)
        grads_R = (torch.nn.utils.clip_grad_norm_(grads_R[0], max_norm=10.0),)

        P_L1 = F.softplus(self.meta_params[0,:])
        P_L3 = F.softplus(self.cross_cond[0,:])

        w_L1_norm = F.softplus(self.meta_params[1,0])
        w_L3_norm = F.softplus(self.cross_cond[1,0])

        P1 = P_L1+w_L1_norm*F.sigmoid(grads_L3[0].norm())
        P3 = P_L3+w_L3_norm*F.sigmoid(grads_L1[0].norm()+F.softplus(self.meta_lambda)*grads_R[0].norm())

        updated_users_emb = learning_users_emb - P1 * (grads_L1[0]+F.softplus(self.meta_lambda)*grads_R[0]) - P3 * grads_L3[0] 

        return updated_users_emb
    
    def MAML_inner_step(self, users_id, questions_id, labels, categories_id, learning_users_emb=None):
        """
        MAML inner step with manual gradient application to preserve computational graph
        """
        L1, L3, R = self.CDM._compute_loss(users_id=users_id, items_id=questions_id, concepts_id=categories_id, labels=labels, learning_users_emb=learning_users_emb)
        loss = L1 + L3 + self.config['lambda'] * R
        
        # Compute gradients with respect to learning_users_emb
        grads = torch.autograd.grad(loss, learning_users_emb, create_graph=False)[0]
        
        # Manual gradient update (equivalent to optimizer step)
        updated_learning_users_emb = learning_users_emb - self.config['inner_user_lr'] * grads
        
        return updated_learning_users_emb

    def Approx_GAP_inner_step(self, users_id, questions_id, labels, categories_id, learning_users_emb=None):
        """
        Meta-learning style inner step: returns updated user embeddings tensor (does not update model in-place).
        Args:
            users_id, questions_id, labels, categories_id: batch data
            users_emb: optional tensor to use as starting point (default: model's current embeddings)
        Returns:
            updated_users_emb: tensor of updated user embeddings (requires_grad)
            loss: loss on the query set
        """
        # 2. Forward pass: compute loss using the copied embeddings
        #    (Assume CDM._compute_loss can take a users_emb argument, else you need to adapt your model)
        L1, L3, R = self.CDM._compute_loss(users_id=users_id, items_id=questions_id, concepts_id=categories_id, labels=labels, learning_users_emb=learning_users_emb)
        
        # 3. Compute gradients w.r.t. the copied embeddings
        grads_L1 = torch.autograd.grad(L1, learning_users_emb, create_graph=False)
        grads_L3 = torch.autograd.grad(L3, learning_users_emb, create_graph=False)
        grads_R = torch.autograd.grad(R, learning_users_emb, create_graph=False)

        prec_L1 = torch.nn.Softplus()(self.meta_params[0,:]).repeat(self.metadata["num_user_id"], 1)

        updated_users_emb = learning_users_emb - prec_L1 * (grads_L1[0] + grads_L3[0]) - self.config['lambda'] * grads_R[0] 

        return updated_users_emb
    
    def Adam_inner_step(self,users_id, questions_id, labels, categories_id, users_emb):

        self.user_params_optimizer.zero_grad()

        L1, L3, R = self.CDM._compute_loss(users_id=users_id, items_id=questions_id, concepts_id=categories_id, labels=labels, learning_users_emb=users_emb)
        loss = L1 + L3 + self.config['lambda'] * R
        loss.backward()
        self.user_params_optimizer.step()
        return users_emb

    def inner_loop(self,query_data, users_emb):
        """
            User parameters update (theta) on the query set
        """
        logging.debug("- Update users ")

        with torch.enable_grad():

            sub_data = dataset.SubmittedDataset(query_data)
            sub_dataloader = DataLoader(sub_data, batch_size=2048, shuffle=True, pin_memory=self.config['pin_memory'], num_workers=self.config['num_workers'])

            for k in range(self.config['num_inner_users_epochs']) :
                   
                for  batch in sub_dataloader:
                    users_id, items_id, labels, concepts_id, _ = batch["user_ids"], batch["question_ids"], batch["labels"], batch["category_ids"], batch["nb_modalities"]
    
                    self.CDM.model.train()
                    users_emb = self.inner_step(users_id, items_id, labels, concepts_id, users_emb)
                    self.CDM.model.eval()

            return users_emb

    def beta_cd_inner_loop(self,query_data, users_emb):
        """
            User parameters update (theta) on the query set
        """
        logging.debug("- Update users (BETA-CD inner loop) ")

        with torch.enable_grad():

            sub_data = dataset.SubmittedDataset(query_data)
            sub_dataloader = DataLoader(sub_data, batch_size=2048, shuffle=True, pin_memory=self.config['pin_memory'], num_workers=self.config['num_workers'])

            q_params = users_emb
            lambda_mean = nn.Parameter(torch.tile(self.user_mean.clone().detach(), (self.metadata["num_user_id"], 1)).requires_grad_(True))
            lambda_std = nn.Parameter(torch.tile(self.user_log_std.clone().detach(), (self.metadata["num_user_id"], 1)).requires_grad_(True))
            p_params = [lambda_mean, lambda_std]

            for k in range(self.config['num_inner_users_epochs']) :
                   
                for  batch in sub_dataloader:
                    users_id, items_id, labels, concepts_id, _ = batch["user_ids"], batch["question_ids"], batch["labels"], batch["category_ids"], batch["nb_modalities"]
    
                    self.CDM.model.train()

                    grads_accum = [torch.zeros_like(q_params[i]) for i in range(len(q_params))]

                    for _ in range(self.num_sample):
                        learning_users_emb = q_params[0] + torch.randn_like(q_params[0], device=q_params[0].device) * torch.exp(q_params[1])
                        L1, L3, _ = self.CDM._compute_loss(users_id=users_id, items_id=items_id, concepts_id=concepts_id, labels=labels, learning_users_emb=learning_users_emb)

                        kl_loss = kl_divergence_gaussians(p=p_params,q=q_params)
                        loss = L1 + L3 + self.kl_weight * kl_loss

                        zero_grad(q_params)
                        grads = torch.autograd.grad(loss, q_params, retain_graph=True)
        
                        grads_accum[0] = grads_accum[0] + grads[0]/ self.num_sample 
                        grads_accum[1] = grads_accum[1] + grads[1]/ self.num_sample

                    q_params[0] = q_params[0] - self.inner_lrs[k].abs() * grads_accum[0]
                    q_params[1] = q_params[1] - self.inner_lrs[k].abs() * grads_accum[1]

                    self.CDM.model.eval()

            return q_params

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
    def evaluate_valid(self, valid_loader: DataLoader, valid_query_env: QueryEnv, learning_users_emb: nn.Parameter):
        """
        Evaluate the model on the given data using the given metrics.
        """
        rmse_list, loss_list = [], []

        logging.debug("-- evaluate valid --")

        match self.config['meta_trainer']:
            case 'GAP':
                learning_users_emb = nn.Parameter(torch.tile(self.learning_users_emb, (self.metadata["num_user_id"], 1)).to(self.config['device']).requires_grad_(True))
            case 'BETA-CD':
                lambda_mean = nn.Parameter(torch.tile(self.user_mean, (self.metadata["num_user_id"], 1)).requires_grad_(True))
                lambda_std = nn.Parameter(torch.tile(self.user_log_std, (self.metadata["num_user_id"], 1)).requires_grad_(True))
                learning_users_emb = [lambda_mean, lambda_std]
            case 'MAML':
                learning_users_emb = nn.Parameter(torch.tile(self.learning_users_emb, (self.metadata["num_user_id"], 1)).to(self.config['device']).requires_grad_(True))
            case default:
                orig_emb = learning_users_emb.detach().clone()

        for batch in valid_loader:

            valid_query_env.load_batch(batch)

            # Prepare the meta set
            meta_data = valid_query_env.generate_IMPACT_meta()

            for i_query in range(self.config['n_query']):

                # Select the action (question to submit)
                actions = self.select_action(valid_query_env.get_query_options(i_query))

                valid_query_env.update(actions, i_query)

                learning_users_emb = self.inner_loop(valid_query_env.feed_IMPACT_sub(), learning_users_emb)

            total_rmse, total_loss = 0.0, 0.0
            if self.config['meta_trainer'] == 'BETA-CD':
                q_params = learning_users_emb
                for _ in range(self.num_sample):
                    users_emb = q_params[0] + torch.randn_like(q_params[0], device=q_params[0].device) * torch.exp(q_params[1])
                    L1, L3, _ = self.CDM._compute_loss(users_id=meta_data['users_id'],items_id=meta_data['questions_id'],concepts_id=meta_data['categories_id'], labels=meta_data['labels'],learning_users_emb=users_emb)
                    total_loss += L1 + L3 
                    preds = self.CDM.forward(users_id=meta_data['users_id'], items_id=meta_data['questions_id'], concepts_id=meta_data['categories_id'], users_emb=users_emb)
                    total_rmse += utils.root_mean_squared_error(y_pred=preds, y_true=meta_data['labels'], nb_modalities=meta_data['nb_modalities'])
                
                total_loss /= self.num_sample 
                total_rmse /= self.num_sample
            else:
                L1, L3, _ = self.CDM._compute_loss(users_id=meta_data['users_id'], items_id=meta_data['questions_id'], concepts_id=meta_data['categories_id'], labels=meta_data['labels'], learning_users_emb=learning_users_emb)
                losses = torch.stack([L1, L3])  # Shape: (4,)
                total_loss = torch.dot(self.weights, losses) 

                preds = self.CDM.forward(users_id=meta_data['users_id'], items_id=meta_data['questions_id'], concepts_id=meta_data['categories_id'], users_emb=learning_users_emb)
                total_rmse = utils.root_mean_squared_error(y_pred=preds, y_true=meta_data['labels'], nb_modalities=meta_data['nb_modalities'])

            # Check for NaN and skip if found
            logging.info(f"rmse : {total_rmse.item()}")
            rmse_list.append(total_rmse/(len(batch)*1e-3))
            loss_list.append(total_loss/(len(batch)*1e-3))

        mean_rmse = torch.mean(torch.stack(rmse_list))
        mean_loss = torch.mean(torch.stack(loss_list))

        if self.config['meta_trainer'] not in ['GAP','BETA-CD', 'MAML' ]:
            learning_users_emb.copy_(orig_emb)

        return mean_rmse, mean_loss
    
    @evaluation_state
    def evaluate_test(self, test_dataset: dataset.EvalDataset,train_dataset: dataset.EvalDataset,valid_dataset: dataset.EvalDataset):
        """CATDataset
        Evaluate the model on the given data using the given metrics.
        """
        logging.debug("-- Evaluate test --")

        # Device cleanup
        torch.cuda.empty_cache()
        logging.info('train on {}'.format(self.config['device']))

        # Users embeddings initialization
        mvn = torch.distributions.MultivariateNormal(loc=self.CDM.model.prior_mean.squeeze(0), covariance_matrix=self.CDM.model.cov_matrix)
        learning_users_emb = nn.Parameter(mvn.sample((self.CDM.model.users_emb.weight.shape[0],)).requires_grad_(True))

        # Initialize testing config
        match self.config['meta_trainer']:
            case 'GAP':
                #mvn = torch.distributions.MultivariateNormal(loc=self.CDM.model.prior_mean, covariance_matrix=self.CDM.model.cov_matrix)
                #learning_users_emb = nn.Parameter(mvn.sample((test_dataset.n_users,)).squeeze(-2).requires_grad_(True))
                learning_users_emb = nn.Parameter(torch.tile(self.learning_users_emb, (self.metadata["num_user_id"], 1)).to(self.config['device']).requires_grad_(True))
            case 'BETA-CD':
                lambda_mean = nn.Parameter(torch.tile(self.user_mean, (self.metadata["num_user_id"], 1)).requires_grad_(True))
                lambda_std = nn.Parameter(torch.tile(self.user_log_std, (self.metadata["num_user_id"], 1)).requires_grad_(True))
                learning_users_emb = [lambda_mean, lambda_std]
            case 'MAML':
                learning_users_emb = nn.Parameter(torch.tile(self.learning_users_emb, (self.metadata["num_user_id"], 1)).to(self.config['device']).requires_grad_(True))
            case 'Approx_GAP':
                self.user_params_optimizer = torch.optim.Adam(
                    [learning_users_emb],
                    lr=self.config['inner_user_lr']
                )
            case 'Adam':
                self.user_params_optimizer = torch.optim.Adam(
                    [learning_users_emb],
                    lr=self.config['inner_user_lr']
                )
            case 'none':
                pass
            case _:
                raise ValueError(f"Unknown meta trainer: {self.config['meta_trainer']}")


        # Updating CDM to test dataset
        self.CDM.init_test(test_dataset)

        # Dataloaders preperation
        test_dataset.split_query_meta(self.config['seed'])
        test_query_env = QueryEnv(test_dataset, self.device, self.config['valid_batch_size'])
        test_loader = data.DataLoader(test_dataset, collate_fn=dataset.UserCollate(test_query_env), batch_size=self.config['valid_batch_size'],
                                      shuffle=False, pin_memory=self.config['pin_memory'], num_workers=self.config['num_workers'])

        # Saving metric structures
        pred_list = {t : [] for t in range(self.config['n_query'])}
        label_list = {t : [] for t in range(self.config['n_query'])}
        nb_modalities_list = {t : [] for t in range(self.config['n_query'])}
        emb_tensor = torch.zeros(size = (self.config['n_query'],test_dataset.n_users, test_dataset.n_categories), device=self.device)

        # Test
        log_idx = 0
        for batch in test_loader:

            test_query_env.load_batch(batch)

            # Prepare the meta set
            meta_data = test_query_env.generate_IMPACT_meta()
            nb_modalities = test_dataset.nb_modalities[meta_data['questions_id']]

            for i_query in tqdm(range(self.config['n_query']), total=self.config['n_query'], disable=self.config['disable_tqdm']):

                # Select the action (question to submit)
                options = test_query_env.get_query_options(i_query)
                actions = self.select_action(options)
                test_query_env.update(actions, i_query)

                learning_users_emb = self.inner_loop(test_query_env.feed_IMPACT_sub(),learning_users_emb)

                if self.config['meta_trainer'] == 'BETA-CD':
                    with torch.no_grad() :
                        preds = torch.zeros(size=(meta_data['users_id'].shape[0],), device=self.device)
                        q_params = learning_users_emb
                        for _ in range(self.num_sample):
                            users_emb = q_params[0] + torch.randn_like(q_params[0], device=q_params[0].device) * torch.exp(q_params[1])
                            preds += self.CDM.forward(users_id=meta_data['users_id'], items_id=meta_data['questions_id'], concepts_id=meta_data['categories_id'], users_emb=users_emb)
                        preds /= self.num_sample
                else:
                    with torch.no_grad() :
                        preds = self.CDM.forward(users_id=meta_data['users_id'], items_id=meta_data['questions_id'], concepts_id=meta_data['categories_id'], users_emb=learning_users_emb)

                pred_list[i_query].append(preds)
                label_list[i_query].append(meta_data['labels'])
                nb_modalities_list[i_query].append(nb_modalities)

                if self.config['meta_trainer'] == 'BETA-CD':
                    # Store the updated user embeddings
                    emb_tensor[i_query, :, :] = q_params[0]
                else:
                    emb_tensor[i_query, :, :] = learning_users_emb

            log_idx += test_query_env.current_batch_size

        # Compute metrics in one pass using a dictionary comprehension
        results_pred = {t : {metric: self.pred_metric_functions[metric](torch.cat(pred_list[t]), torch.cat(label_list[t]), torch.cat(nb_modalities_list[t])).cpu().item()
                   for metric in self.pred_metrics} for t in range(self.config['n_query'])}

        results_profiles = {
            t: {
                metric: self.profile_metric_functions[metric](emb_tensor[t,:, :], test_dataset, train_dataset, valid_dataset)
                for metric in self.profile_metrics
            }
            for t in range(self.config["n_query"])
        }

        return results_pred, results_profiles

    def init_models(self, train_dataset: dataset.CATDataset, valid_dataset: dataset.EvalDataset):

        logging.debug("------- Initialize CDM and Selection strategy")
        
        match self.config['CDM']:
            case 'impact':
                self.CDM.init_CDM_model(train_dataset, valid_dataset)

                if hasattr(torch, "compile"):
                    logging.info("compiling CDM model")
                    self.CDM.model = torch.compile(self.CDM.model)

        self.CDM.init_users_prior(train_dataset, valid_dataset)

        if hasattr(torch, "compile") and not getattr(self, "model_compiled", False):
            logging.info("compiling selection model")
            if isinstance(self.model, torch.nn.Module):
                self.model = torch.compile(self.model)
                self.model_compiled = True
            else:
                logging.info("Selection model already compiled, skipping recompilation")
        else:
            logging.info("Selection model already compiled, skipping recompilation")


        if hasattr(self.model, "to"):
            self.model.to(self.config['device'])
        else:
            logging.info("Warning: self.model is a function and cannot be moved to device")



    def train(self, train_dataset: dataset.CATDataset, valid_dataset: dataset.EvalDataset):

        logging.info("------- START Training")

        # Device cleanup
        torch.cuda.empty_cache()
        logging.info('train on {}'.format(self.config['device']))

        # Put models in training mode
        self.model.train()
        self.CDM.model.train()
        
        # Users embeddings initialization
        mvn = torch.distributions.MultivariateNormal(loc=self.CDM.model.prior_mean.squeeze(0), covariance_matrix=self.CDM.model.cov_matrix)
        learning_users_emb = nn.Parameter(mvn.sample((self.CDM.model.users_emb.weight.shape[0],)).requires_grad_(True))


        # Initialize training optimizers and schedulers
        lr = self.config['learning_rate']

        if self.trainable :
            self.S_optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
            self.S_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.S_optimizer, patience=2, factor=0.5)
            self.S_scaler = torch.amp.GradScaler(self.device)
        else : 
            self.S_optimizer = DummyOptimizer()
            self.S_scheduler = DummyScheduler()
            self.S_scaler = DummyScaler()
            
        #self.CDM_optimizer = torch.optim.Adam(self.CDM.model.parameters(), lr=self.config['inner_lr'])
        #self.CDM_scaler = torch.amp.GradScaler(self.config['device'])

        match self.config['meta_trainer']:

            case 'GAP':
                self.inner_step = self.GAP_inner_step
                self.CDM.set_regularizer_with_prior()

                # Meta parameters declaration
                self.weights = torch.tensor([1.0, 1.0], device=self.config['device'])
                self.learning_users_emb = nn.Parameter(torch.empty(self.metadata['num_dimension_id']), requires_grad=True)#.to(device=self.config['device'])
                self.meta_params = torch.nn.Parameter(torch.empty(2, self.metadata['num_dimension_id']))
                self.cross_cond = torch.nn.Parameter(torch.empty(2, self.metadata['num_dimension_id']),
                    requires_grad=True)
                self.meta_lambda = torch.nn.Parameter(
                    inverse_softplus(torch.tensor(self.config['lambda'], device=self.device, dtype=torch.float32).clone()),
                    requires_grad=True
                )

                # Meta parameters initialization
                torch.nn.init.normal_(self.meta_params, mean=0.0, std=0.5)
                self.meta_params.data = self.meta_params.data.to(self.config['device'])
                with torch.no_grad():
                    self.meta_params.data[0,:] = self.meta_params.data[0,:] + inverse_softplus(torch.tensor(self.config['inner_user_lr']*500))
                    self.meta_params.data[1,:] = self.meta_params.data[1,:] + 20

                torch.nn.init.normal_(self.cross_cond, mean=0.0, std=0.5)
                self.cross_cond.data = self.cross_cond.data.to(self.config['device'])
                with torch.no_grad():
                    self.cross_cond.data[0,:] = self.cross_cond.data[0,:]+inverse_softplus(torch.tensor(self.config['inner_user_lr']*500))
                    self.cross_cond.data[1,:] = self.cross_cond.data[1,:]+20

                with torch.no_grad():
                    self.learning_users_emb.data = self.CDM.model.prior_mean.data.clone()

                # Optimizers declaration
                self.meta_optimizer = torch.optim.Adam(
                    [
                        {'params': self.meta_params,  'lr': self.config.get('meta_lr', 0.5), "name": "meta_params"},
                        {'params': self.cross_cond,  'lr': self.config.get('meta_lr', 0.5), "name": "cross_cond"},
                        {'params': self.meta_lambda,  'lr': self.config.get('meta_lr', 0.5), "name": "meta_lambda"},
                        {'params': self.learning_users_emb,  'lr': self.config.get('learning_users_emb_lr', 0.0005), "name": "learning_users_emb"},
                    ]
                )
                self.meta_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.meta_optimizer, patience=2, factor=0.5)
                self.meta_scaler = torch.amp.GradScaler(self.config['device'])

            case 'MAML':
                self.inner_step = self.MAML_inner_step

                # Meta parameters declaration
                self.weights = torch.tensor([1.0, 1.0], device=self.config['device'])
                self.learning_users_emb = nn.Parameter(torch.empty(self.metadata['num_dimension_id']), requires_grad=True)#.to(device=self.config['device'])

                torch.nn.init.normal_(self.learning_users_emb, mean=0.0, std=0.5)

                # Meta optimizers declaration
                self.meta_optimizer = torch.optim.Adam(
                    [
                        {'params': self.learning_users_emb,  'lr': self.config.get('meta_lr', 0.001), "name": "learning_users_emb"},
                    ]
                )
                self.meta_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.meta_optimizer, patience=2, factor=0.5)
                self.meta_scaler = torch.amp.GradScaler(self.config['device'])     
                

            case 'BETA-CD':
                self.inner_loop = self.beta_cd_inner_loop
                self.weights = torch.tensor([1.0, 1.0], device=self.config['device'])

                self.num_sample = 10
                
                # Meta parameters declaration
                self.user_mean = torch.nn.Parameter(
                    (torch.randn(self.metadata['num_dimension_id']).to(self.device)-0.05)*0.001
                    ,requires_grad=True
                )
                self.user_log_std = torch.nn.Parameter(
                    torch.randn(self.metadata['num_dimension_id']).to(self.device)-4
                    ,requires_grad=True
                )
                self.kl_weight = torch.nn.Parameter(torch.Tensor([self.config.get('kl_weight', 1e-3)]).to(self.device), requires_grad=True)
                self.inner_lrs = torch.nn.Parameter(torch.Tensor([0.05]* self.config['num_inner_users_epochs']).to(self.device), requires_grad=True)
                
                # Optimizers declaration
                self.meta_optimizer = torch.optim.Adam(
                    [
                        {'params': self.user_mean,  'lr': self.config.get('meta_lr', 0.05), 'name': 'user_mean'},
                        {'params': self.user_log_std,  'lr': self.config.get('meta_lr', 0.05), 'name': 'user_log_std'},
                        {'params': self.inner_lrs,  'lr': self.config.get('meta_lr', 0.05), 'name': 'inner_lrs'},
                    ]
                )
                self.meta_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.meta_optimizer, patience=2, factor=0.5)
                self.meta_scaler = torch.amp.GradScaler(self.config['device'])


            case 'Approx_GAP':
                self.inner_step = self.Approx_GAP_inner_step

                # Meta parameters declaration
                self.meta_params = torch.nn.Parameter(torch.empty(1, self.metadata['num_dimension_id']))
                torch.nn.init.normal_(self.meta_params, mean=0.0, std=0.5)
                self.meta_params.data = self.meta_params.data.to(self.config['device'])
                self.meta_params.data[0,:] = self.meta_params.data[0,:] + inverse_softplus(torch.tensor(self.config['inner_user_lr']*500))
                self.weights = torch.tensor([1.0, 1.0], device=self.config['device'])

                # Optimizers declaration
                self.meta_optimizer = torch.optim.Adam(
                    [
                        {'params': self.meta_params,  'lr': self.config.get('meta_lr', 0.5), 'name': 'meta_params'},
                    ]
                )
                self.meta_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.meta_optimizer, patience=2, factor=0.5)
                self.meta_scaler = torch.amp.GradScaler(self.config['device'])

            case 'MAML':
                logging.warning("The prior needs to be canged !")
            case 'none':
                pass
            case _:
                raise ValueError(f"Unknown meta trainer: {self.config['meta_trainer']}")
            
        # Initialize early stopping parameters
        self.best_epoch = 0
        self.best_valid_rmse = float('inf')

        self.best_S_params = self.get_params()
        self.best_CDM_params = self.CDM.get_params()        

        # Train
        self._train_method(train_dataset, valid_dataset,learning_users_emb)

        self._trained = True

        logging.info("-- END Training --")

        if self.config['save_params']:
            self._save_model_params(temporary=False)
            logging.info("Params saved")
        self.fold += 1

    def _train_early_stopping_error(self, train_dataset, valid_dataset,learning_users_emb):

        epochs = self.config['num_epochs']
        eval_freq = self.config['eval_freq']
        patience = self.config['patience']

        valid_dataset.split_query_meta(self.config['seed']) # split valid query qnd meta set one and for all epochs

        train_query_env = QueryEnv(train_dataset, self.device, self.config['batch_size'])
        valid_query_env = QueryEnv(valid_dataset, self.device, self.config['valid_batch_size'])

        train_loader = DataLoader(dataset=train_dataset, collate_fn=UserCollate(train_query_env), batch_size=self.config['batch_size'],
                                  shuffle=True, pin_memory=self.config['pin_memory'], num_workers=self.config['num_workers'])
        valid_loader = DataLoader(dataset=valid_dataset, collate_fn=UserCollate(valid_query_env),
                                  batch_size=self.config['valid_batch_size'],
                                  shuffle=False, pin_memory=self.config['pin_memory'], num_workers=self.config['num_workers'])
        
        if not hasattr(self, 'learning_users_emb') and self.config['meta_trainer'] != 'BETA-CD':
            orig_emb = learning_users_emb.detach().clone()

        for _, ep in tqdm(enumerate(range(epochs + 1)), total=epochs, disable=self.config['disable_tqdm']):

            logging.debug(f'------- Epoch : {ep}')

            train_dataset.set_query_seed(ep) # changes the meta and query set for each epoch
            
            if hasattr(self, 'learning_users_emb'):
                learning_users_emb = torch.tile(self.learning_users_emb, (train_dataset.n_users, 1)).to(self.config['device'])
            elif self.config['meta_trainer'] == 'BETA-CD':
                lambda_mean = torch.tile(self.user_mean, (train_dataset.n_users, 1))
                lambda_std = torch.tile(self.user_log_std, (train_dataset.n_users, 1))
                learning_users_emb = [lambda_mean, lambda_std]
            else:
                learning_users_emb = orig_emb.clone().detach().requires_grad_(True)

            if self.config['meta_trainer'] == 'MAML':
                learning_users_emb = torch.tile(self.learning_users_emb, (train_dataset.n_users, 1)).to(self.config['device'])

            mean_meta_loss = 0.0

            for u_batch, batch in enumerate(train_loader): # UserCollate directly load the data into the query environment
                
                logging.info(f'----- User batch : {u_batch}')

                train_query_env.load_batch(batch)

                # Prepare the meta set
                meta_data = train_query_env.generate_IMPACT_meta()

                for i_query in range(self.config['n_query']):

                    logging.debug(f'--- Query nb : {i_query}')

                    actions = self.select_action(train_query_env.get_query_options(i_query))
                    train_query_env.update(actions, i_query)

                    i_query_data = train_query_env.feed_IMPACT_sub()
                
                    learning_users_emb = self.inner_loop(i_query_data,learning_users_emb)

                if self.config['meta_trainer'] == 'BETA-CD':
                    q_params = learning_users_emb
                    meta_loss = 0.0
                    for _ in range(self.num_sample):
                        users_emb = q_params[0] + torch.randn_like(q_params[0], device=q_params[0].device) * torch.exp(q_params[1])
                        L1, L3, _ = self.CDM._compute_loss(users_id=meta_data['users_id'],items_id=meta_data['questions_id'],concepts_id=meta_data['categories_id'], labels=meta_data['labels'],learning_users_emb=users_emb)
                        meta_loss += L1 
                    meta_loss /= self.num_sample
                    mean_meta_loss += meta_loss / len(batch)
                else:
                    L1, L3, _ = self.CDM._compute_loss(users_id=meta_data["users_id"], items_id=meta_data["questions_id"], concepts_id=meta_data["categories_id"], labels=meta_data["labels"], learning_users_emb=learning_users_emb)
                    losses = torch.stack([L1, L3])  # Shape: (4,)
                    
                    meta_loss = torch.dot(self.weights, losses) 
                    mean_meta_loss += meta_loss / len(batch)
                    
            if self.config['meta_trainer'] != 'Adam':
                self.update_meta_params(mean_meta_loss)
            #self.update_S_params(meta_loss)
            #self.update_CDM_params(meta_loss)
            if not hasattr(self, 'learning_users_emb') and self.config['meta_trainer'] != 'BETA-CD':
                with torch.no_grad():
                    learning_users_emb.copy_(orig_emb)
                learning_users_emb.requires_grad_(True)

            # Early stopping
            if (ep + 1) % eval_freq == 0:
                with torch.no_grad(), torch.amp.autocast('cuda'):
                    valid_rmse, valid_loss = self.evaluate_valid(valid_loader, valid_query_env,learning_users_emb)

                    logging.info(f'valid_rmse : {valid_rmse}')
                    logging.info(f'valid_loss : {valid_loss}')

                    with warnings.catch_warnings(record=True) as w:
                        warnings.simplefilter("always")

                    self.S_scheduler.step(valid_loss)
                    self.meta_scheduler.step(valid_loss)


                    #TODO : add CDM scheduler

                    # Checking loss improvement
                    if  self.best_valid_rmse > valid_rmse:  # (self.best_valid_metric - valid_rmse) / abs(self.best_valid_metric) > 0.001:
                        self.best_epoch = ep
                        self.best_valid_rmse = valid_rmse
                        self.best_model_params = {'state_dict': self.model.state_dict()}
                        if hasattr(self, 'meta_params') :
                            self.best_model_params['meta_params'] = self.meta_params.detach().clone()
                        if hasattr(self, 'meta_mean') :
                            self.best_model_params['meta_mean'] = self.meta_mean.detach().clone()
                        if hasattr(self, 'meta_lambda') :
                            self.best_model_params['meta_lambda'] = self.meta_lambda.detach().clone()
                        if hasattr(self, 'cross_cond') :
                            self.best_model_params['cross_cond'] = self.cross_cond.detach().clone()
                        if hasattr(self, 'learning_users_emb') :
                            self.best_model_params['learning_users_emb'] = self.learning_users_emb.detach().clone()
                        if hasattr(self, 'user_mean') :
                            self.best_model_params['user_mean'] = self.user_mean.detach().clone()
                        if hasattr(self, 'user_log_std') :
                            self.best_model_params['user_log_std'] = self.user_log_std.detach().clone()
                        if hasattr(self, 'inner_lrs') :
                            self.best_model_params['inner_lrs'] = self.inner_lrs.detach().clone()
                        if hasattr(self, 'kl_weight') :
                            self.best_model_params['kl_weight'] = self.kl_weight.detach().clone()

                    if ep - self.best_epoch >= patience or torch.max(torch.tensor([param_group['lr'] for param_group in self.meta_optimizer.param_groups])) < 1e-4:
                        break

        self.model.load_state_dict(self.best_model_params['state_dict'])

        if hasattr(self, 'meta_params') :
            self.meta_params = self.best_model_params['meta_params'].requires_grad_()
        if hasattr(self, 'meta_mean') :
            self.meta_mean = self.best_model_params['meta_mean'].requires_grad_()
        if hasattr(self, 'meta_lambda') :
            self.meta_lambda = self.best_model_params['meta_lambda'].requires_grad_()
        if hasattr(self, 'cross_cond') :
            self.cross_cond = self.best_model_params['cross_cond'].requires_grad_()
        if hasattr(self, 'learning_users_emb') :
            self.learning_users_emb = self.best_model_params['learning_users_emb'].requires_grad_()
        if hasattr(self, 'user_mean') :
            self.user_mean = self.best_model_params['user_mean'].requires_grad_()
        if hasattr(self, 'user_log_std') :
            self.user_log_std = self.best_model_params['user_log_std'].requires_grad_()
        if hasattr(self, 'inner_lrs') :
            self.inner_lrs = self.best_model_params['inner_lrs'].requires_grad_()
        if hasattr(self, 'kl_weight') :
            self.kl_weight = self.best_model_params['kl_weight'].requires_grad_()


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


