import functools
import itertools
import warnings
from abc import ABC, abstractmethod
from torch import nn
from liriscat.selectionStrategy.CovWeighting import CoVWeightingLoss

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
                self.inner_step = self.GAP_inner_step
                self.meta_params = torch.nn.Parameter(torch.empty(7, metadata['num_dimension_id']))
                torch.nn.init.normal_(self.meta_params, mean=0.0, std=0.5)
                self.meta_params.data = self.meta_params.data.to(self.config['device'])
                self.meta_params.data[0,:] = self.meta_params.data[0,:] + inverse_softplus(torch.tensor(self.config['inner_user_lr']*500))
                self.meta_params.data[1,:] = self.meta_params.data[1,:] + inverse_softplus(torch.tensor(self.config['inner_user_lr']*500))

                self.meta_params.data[2,0] = torch.log(torch.exp(torch.tensor(self.config['lambda']*3))-1)
                self.meta_params.data[2,1] = 15
                self.meta_params.data[2,2] = 15

                self.meta_params.data[3,:] = self.meta_params.data[3,:]+40
                self.meta_params.data[4,:] = self.meta_params.data[4,:]+40
                self.meta_params.data[5,:] = self.meta_params.data[5,:]+15
                self.meta_params.data[6,:] = self.meta_params.data[6,:]+15

                self.weights = torch.tensor([1.0, 1.0], device=self.config['device'])

            case 'Approx_GAP':
                self.inner_step = self.Approx_GAP_inner_step
                self.meta_params = torch.nn.Parameter(torch.empty(1, metadata['num_dimension_id']))
                torch.nn.init.normal_(self.meta_params, mean=0.0, std=0.5)
                self.meta_params.data = self.meta_params.data.to(self.config['device'])
                self.meta_params.data[0,:] = self.meta_params.data[0,:] + inverse_softplus(torch.tensor(self.config['inner_user_lr']*500))
                self.weights = torch.tensor([1.0, 1.0], device=self.config['device'])

            case 'Approx_GAP_cw':
                self.inner_step = self.Approx_GAP_cw_inner_step
                self.meta_params = torch.nn.Parameter(torch.empty(1, metadata['num_dimension_id']))
                torch.nn.init.normal_(self.meta_params, mean=0.0, std=0.5)
                self.meta_params.data = self.meta_params.data.to(self.config['device'])
                self.meta_params.data[0,:] = self.meta_params.data[0,:] + inverse_softplus(torch.tensor(self.config['inner_user_lr']*500))

            case 'Approx_GAP_mult':
                self.inner_step = self.Approx_GAP_mult_inner_step
                self.meta_params = torch.nn.Parameter(torch.empty(2, metadata['num_dimension_id']))
                torch.nn.init.normal_(self.meta_params, mean=0.0, std=0.5)
                self.meta_params.data = self.meta_params.data.to(self.config['device'])
                self.meta_params.data[0,:] = self.meta_params.data[0,:] + inverse_softplus(torch.tensor(self.config['inner_user_lr']*500))
                self.meta_params.data[1,:] = self.meta_params.data[1,:] + inverse_softplus(torch.tensor(self.config['inner_user_lr']*500))
                self.weights = torch.tensor([1.0, 1.0], device=self.config['device'])            
            
            case 'Approx_GAP_mult_lambda_prior':
                self.inner_step = self.Approx_GAP_mult_lambda_prior_inner_step
                self.meta_params = torch.nn.Parameter(torch.empty(2, metadata['num_dimension_id']))
                torch.nn.init.normal_(self.meta_params, mean=0.0, std=0.5)
                self.meta_params.data = self.meta_params.data.to(self.config['device'])
                self.meta_params.data[0,:] = self.meta_params.data[0,:] + inverse_softplus(torch.tensor(self.config['inner_user_lr']*500))
                self.meta_params.data[1,:] = self.meta_params.data[1,:] + inverse_softplus(torch.tensor(self.config['inner_user_lr']*500))
                self.weights = torch.tensor([1.0, 1.0], device=self.config['device'])

            case 'Approx_GAP_mult_cw':
                self.inner_step = self.Approx_GAP_mult_cw_inner_step
                self.meta_params = torch.nn.Parameter(torch.empty(2, metadata['num_dimension_id']))
                torch.nn.init.normal_(self.meta_params, mean=0.0, std=0.5)
                self.meta_params.data = self.meta_params.data.to(self.config['device'])
                self.meta_params.data[0,:] = self.meta_params.data[0,:] + inverse_softplus(torch.tensor(self.config['inner_user_lr']*500))
                self.meta_params.data[1,:] = self.meta_params.data[1,:] + inverse_softplus(torch.tensor(self.config['inner_user_lr']*500))

            case 'Approx_GAP_mult_mean_prior':
                self.inner_step = self.Approx_GAP_mult_mean_prior_inner_step
                self.meta_params = torch.nn.Parameter(torch.empty(2, metadata['num_dimension_id']))
                torch.nn.init.normal_(self.meta_params, mean=0.0, std=0.5)
                self.meta_params.data = self.meta_params.data.to(self.config['device'])
                self.meta_params.data[0,:] = self.meta_params.data[0,:] + inverse_softplus(torch.tensor(self.config['inner_user_lr']*500))
                self.meta_params.data[1,:] = self.meta_params.data[1,:] + inverse_softplus(torch.tensor(self.config['inner_user_lr']*500))
                self.weights = torch.tensor([1.0, 1.0], device=self.config['device'])

            case 'Approx_GAP_mult_mean_lambda_prior':
                self.inner_step = self.Approx_GAP_mult_lambda_prior_inner_step
                self.meta_params = torch.nn.Parameter(torch.empty(2, metadata['num_dimension_id']))
                torch.nn.init.normal_(self.meta_params, mean=0.0, std=0.5)
                self.meta_params.data = self.meta_params.data.to(self.config['device'])
                self.meta_params.data[0,:] = self.meta_params.data[0,:] + inverse_softplus(torch.tensor(self.config['inner_user_lr']*500))
                self.meta_params.data[1,:] = self.meta_params.data[1,:] + inverse_softplus(torch.tensor(self.config['inner_user_lr']*500))
                self.weights = torch.tensor([1.0, 1.0], device=self.config['device'])
        

            case 'Approx_GAP_mult_std_prior':
                self.inner_step = self.Approx_GAP_mult_inner_step
                self.meta_params = torch.nn.Parameter(torch.empty(2, metadata['num_dimension_id']))
                torch.nn.init.normal_(self.meta_params, mean=0.0, std=0.5)
                self.meta_params.data = self.meta_params.data.to(self.config['device'])
                self.meta_params.data[0,:] = self.meta_params.data[0,:] + inverse_softplus(torch.tensor(self.config['inner_user_lr']*500))
                self.meta_params.data[1,:] = self.meta_params.data[1,:] + inverse_softplus(torch.tensor(self.config['inner_user_lr']*500))
                self.weights = torch.tensor([1.0, 1.0], device=self.config['device'])
                

            case 'Approx_GAP_mult_full_prior':
                self.inner_step = self.Approx_GAP_mult_inner_step
                self.meta_params = torch.nn.Parameter(torch.empty(2, metadata['num_dimension_id']))
                torch.nn.init.normal_(self.meta_params, mean=0.0, std=0.5)
                self.meta_params.data = self.meta_params.data.to(self.config['device'])
                self.meta_params.data[0,:] = self.meta_params.data[0,:] + inverse_softplus(torch.tensor(self.config['inner_user_lr']*500))
                self.meta_params.data[1,:] = self.meta_params.data[1,:] + inverse_softplus(torch.tensor(self.config['inner_user_lr']*500))
                self.weights = torch.tensor([1.0, 1.0], device=self.config['device'])
                self.weights = torch.tensor([1.0, 1.0], device=self.config['device'])

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

            if not self.config['debug'] : 
                if hasattr(self, 'meta_params') and self.meta_params is not None and self.meta_params.grad is not None:
                    logging.info(f"meta_params gradient norm: {[self.meta_params[i,:].grad.norm() for i in range(self.meta_params.shape[0])]}")
                if hasattr(self, 'meta_mean') and self.meta_mean is not None and self.meta_mean.grad is not None:
                    logging.info(f"meta_mean gradient norm: {self.meta_mean.grad.norm()}")
                if hasattr(self, 'meta_lambda') and self.meta_lambda is not None and self.meta_lambda.grad is not None:
                    logging.info(f"meta_lambda gradient norm: {self.meta_lambda.grad.norm()}")
            
            # Add gradient clipping for numerical stability
            if hasattr(self, 'meta_params') and self.meta_params is not None:
                torch.nn.utils.clip_grad_norm_(self.meta_params, max_norm=10.0)
            if hasattr(self, 'meta_mean') and self.meta_mean is not None:
                torch.nn.utils.clip_grad_norm_(self.meta_mean, max_norm=10.0)
            if hasattr(self, 'meta_lambda') and self.meta_lambda is not None:
                torch.nn.utils.clip_grad_norm_(self.meta_lambda, max_norm=1.0)
                
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

        prec_L1 = torch.nn.Softplus()(self.meta_params[0,:]).repeat(self.metadata["num_user_id"], 1)#+self.meta_params[3,:]*torch.nn.Sigmoid()(grads_L3[0]+self.meta_params[5,:])
        prec_L3 = torch.nn.Softplus()(self.meta_params[1,:]).repeat(self.metadata["num_user_id"], 1)#+self.meta_params[4,:]*torch.nn.Sigmoid()(grads_L1[0]+self.meta_params[6,:])

        updated_users_emb = learning_users_emb - prec_L1 * grads_L1[0] - prec_L3 * grads_L3[0] - self.config['lambda'] * grads_R[0] 

        return updated_users_emb,  prec_L1 * L1 + prec_L3 * grads_L3[0] + self.config['lambda'] * R

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

        updated_users_emb = learning_users_emb - prec_L1 * (self.weights[0]*grads_L1[0] + self.weights[1]* grads_L3[0]) - self.config['lambda'] * grads_R[0] 

        return updated_users_emb,  prec_L1 * (self.weights[0]*L1 + self.weights[1]* L3) + self.config['lambda'] * R
    
    def Approx_GAP_mult_inner_step(self, users_id, questions_id, labels, categories_id, learning_users_emb=None):
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
        prec_L3 = torch.nn.Softplus()(self.meta_params[1,:]).repeat(self.metadata["num_user_id"], 1)

        updated_users_emb = learning_users_emb - prec_L1 * self.weights[0]*grads_L1[0] - prec_L3 * self.weights[1]* grads_L3[0] - self.config['lambda'] * grads_R[0] 

        return updated_users_emb,  prec_L1 * (self.weights[0]*L1 + self.weights[1]* L3) + self.config['lambda'] * R
    
    def Approx_GAP_mult_mean_prior_inner_step(self, users_id, questions_id, labels, categories_id, learning_users_emb=None):
        """
        Meta-learning style inner step: returns updated user embeddings tensor (does not update model in-place).
        """
        # Forward pass: compute loss using the copied embeddings
        L1, L3, _ = self.CDM._compute_loss(users_id=users_id, items_id=questions_id, concepts_id=categories_id, labels=labels, learning_users_emb=learning_users_emb)

        unique_users = torch.unique(users_id)
        
        # Compute custom regularizer: (E-M)Σ^(-1)(E-M)^T
        A = learning_users_emb[unique_users] - torch.nn.Softplus()(self.meta_mean) * self.CDM.model.prior_mean  # Shape: [n_users, d_features]
        S = self.CDM.model.prior_cov_inv.to(dtype=learning_users_emb.dtype, device=learning_users_emb.device)  # Ensure same dtype
        
        # Compute gradients w.r.t. learning_users_emb
        grads_L1 = torch.autograd.grad(L1, learning_users_emb, create_graph=False, retain_graph=True)
        grads_L3 = torch.autograd.grad(L3, learning_users_emb, create_graph=False, retain_graph=True)
        
        # Compute analytical gradient of regularizer w.r.t. learning_users_emb
        grads_R_tensor = torch.zeros_like(learning_users_emb)
        # Analytical gradient: ∂R/∂E = 2(E-M)Σ^(-1)
        grads_R_tensor[unique_users] = 2 * torch.matmul(A, S).to(dtype=learning_users_emb.dtype)  # 2(E-M)Σ^(-1)
        
        # Preconditioners
        prec_L1 = torch.nn.Softplus()(self.meta_params[0,:]).repeat(self.metadata["num_user_id"], 1)
        prec_L3 = torch.nn.Softplus()(self.meta_params[1,:]).repeat(self.metadata["num_user_id"], 1)
        
        # Update user embeddings
        updated_users_emb = (learning_users_emb 
                             - prec_L1 * self.weights[0] * grads_L1[0] 
                             - prec_L3 * self.weights[1] * grads_L3[0] 
                             - self.config['lambda'] * grads_R_tensor) 

        return updated_users_emb, prec_L1 * (self.weights[0]*L1 + self.weights[1]* L3) 
    
    def Approx_GAP_mult_lambda_prior_inner_step(self, users_id, questions_id, labels, categories_id, learning_users_emb=None):
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
        prec_L3 = torch.nn.Softplus()(self.meta_params[1,:]).repeat(self.metadata["num_user_id"], 1)

        updated_users_emb = learning_users_emb - prec_L1 * self.weights[0]*grads_L1[0] - prec_L3 * self.weights[1]* grads_L3[0] - torch.nn.Softplus()(self.meta_lambda) * grads_R[0]

        return updated_users_emb,  prec_L1 * (self.weights[0]*L1 + self.weights[1]* L3) + torch.nn.Softplus()(self.meta_lambda) * R

    
    def Approx_GAP_mult_cw_inner_step(self, users_id, questions_id, labels, categories_id, learning_users_emb=None):
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
        
        losses = torch.stack([L1, L3])  # Shape: (4,)

        # Update statistics and compute weights
        self.weights = self.L_W.compute_weights(losses)

        # 3. Compute gradients w.r.t. the copied embeddings
        grads_L1 = torch.autograd.grad(L1, learning_users_emb, create_graph=False)
        grads_L3 = torch.autograd.grad(L3, learning_users_emb, create_graph=False)
        grads_R = torch.autograd.grad(R, learning_users_emb, create_graph=False)

        prec_L1 = torch.nn.Softplus()(self.meta_params[0,:]).repeat(self.metadata["num_user_id"], 1)
        prec_L3 = torch.nn.Softplus()(self.meta_params[1,:]).repeat(self.metadata["num_user_id"], 1)

        updated_users_emb = learning_users_emb - prec_L1 * self.weights[0]*grads_L1[0] - prec_L3 * self.weights[1]* grads_L3[0] - self.config['lambda'] * grads_R[0] 

        return updated_users_emb,  prec_L1 * (self.weights[0]*L1 + self.weights[1]* L3) + self.config['lambda'] * R
    
    def Adam_inner_step(self,users_id, questions_id, labels, categories_id, users_emb):

        self.user_params_optimizer.zero_grad()
        unique_users =  torch.unique(users_id)
        L1, L3, _ = self.CDM._compute_loss(users_id=users_id, items_id=questions_id, concepts_id=categories_id, labels=labels, learning_users_emb=users_emb)
        loss = L1 + L3 + self.config['lambda'] * users_emb[unique_users].norm().pow(2)
        loss.backward()
        self.user_params_optimizer.step()
        return users_emb, loss

    def inner_loop(self,query_data, meta_data, users_emb):
        """
            User parameters update (theta) on the query set
        """
        logging.debug("- Update users ")

        with torch.enable_grad():

            sub_data = dataset.SubmittedDataset(query_data)
            sub_dataloader = DataLoader(sub_data, batch_size=2048, shuffle=True, pin_memory=self.config['pin_memory'], num_workers=self.config['num_workers'])
            n_batches = len(sub_dataloader)

            if self.config['debug']:
                u_emb_copy = users_emb.clone().requires_grad_(True)

            for k in range(self.config['num_inner_users_epochs']) :
    
                if self.config['debug']:
                    sum_loss_0 = sum_acc_0 = sum_acc_1 = sum_meta_acc = sum_meta_loss = 0
                   
    
                for  batch in sub_dataloader:
                    users_id, items_id, labels, concepts_id, nb_modalities = batch["user_ids"], batch["question_ids"], batch["labels"], batch["category_ids"], batch["nb_modalities"]
    
                    if self.config['debug']:
                        self.CDM.model.eval()
                        with torch.no_grad() , torch.amp.autocast('cuda'):                    
                            preds = self.CDM.forward(users_id, items_id, concepts_id,users_emb=users_emb)
                            sum_acc_0 += utils.micro_ave_accuracy(labels, preds,nb_modalities)
                            meta_preds = self.CDM.forward(users_id=meta_data['users_id'], items_id=meta_data['questions_id'], concepts_id=meta_data['categories_id'],users_emb=users_emb)                            

                    self.CDM.model.train()
                    users_emb, loss = self.inner_step(users_id, items_id, labels, concepts_id, users_emb)
                    self.CDM.model.eval()

                    if self.config['debug']:
                        with torch.no_grad(), torch.amp.autocast('cuda'):
                            sum_loss_0 += loss.item()
                            preds = self.CDM.forward(users_id, items_id, concepts_id,users_emb=users_emb)
                            sum_acc_1 += utils.micro_ave_accuracy(labels,preds, nb_modalities)
                            sum_meta_acc += utils.micro_ave_accuracy(meta_data['labels'], meta_preds,meta_data['nb_modalities'])
                            L1,L3,R = self.CDM._compute_loss(users_id=meta_data['users_id'], items_id=meta_data['questions_id'], concepts_id=meta_data['categories_id'], labels=meta_data['labels'],learning_users_emb=users_emb)
                            sum_meta_loss += (L1.item() + L3.item() + R.item())

                if self.config['debug']:
                    logging.debug(
                        f'- inner step : {k} '
                        f'- emb dist : {torch.norm(users_emb - u_emb_copy)} '
                        f'- query loss 0 : {sum_loss_0/n_batches:.5f} '
                        f'- query acc 0 : {sum_acc_0/n_batches:.5f} '
                        f'- query acc 1 : {sum_acc_1/n_batches:.10f} '
                        f'- meta acc 0 : {sum_meta_acc/n_batches:.5f}'
                        f'- meta loss 1 : {sum_meta_loss:.5f}'
                    )
                    for handler in logging.getLogger().handlers:
                        handler.flush()

            return users_emb

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
        loss_list = []

        logging.debug("-- evaluate valid --")

        orig_emb = learning_users_emb.detach().clone()

        for batch in valid_loader:

            valid_query_env.load_batch(batch)

            # Prepare the meta set
            meta_data = valid_query_env.generate_IMPACT_meta()

            for t in range(self.config['n_query']):

                # Select the action (question to submit)
                actions = self.select_action(valid_query_env.get_query_options(t))

                valid_query_env.update(actions, t)

                learning_users_emb = self.inner_loop(valid_query_env.feed_IMPACT_sub(),meta_data, learning_users_emb)

            L1, L3, _ = self.CDM._compute_loss(users_id=meta_data['users_id'],items_id=meta_data['questions_id'],concepts_id=meta_data['categories_id'], labels=meta_data['labels'].int(),learning_users_emb=learning_users_emb)

            losses = torch.stack([L1, L3])  # Shape: (4,)
            
            total_loss = torch.dot(self.weights, losses)
            
            # Check for NaN and skip if found

            loss_list.append(total_loss/len(batch))
        
        mean_loss = torch.mean(torch.stack(loss_list))
        learning_users_emb.copy_(orig_emb)

        return mean_loss
    
    @evaluation_state
    def evaluate_test(self, test_dataset: dataset.EvalDataset,train_dataset: dataset.EvalDataset,valid_dataset: dataset.EvalDataset):
        """CATDataset
        Evaluate the model on the given data using the given metrics.
        """
        logging.debug("-- Evaluate test --")

        # Device cleanup
        torch.cuda.empty_cache()
        logging.info('train on {}'.format(self.config['device']))

        learning_users_emb = nn.Parameter(self.CDM.model.users_emb.weight.detach().clone().requires_grad_(True))

        # Initialize testing config
        match self.config['meta_trainer']:
            case 'GAP':
                pass
            case 'Approx_GAP':
                pass
            case 'Approx_GAP_cw':
                pass
            case 'Approx_GAP_mult':
                pass
            case 'Approx_GAP_mult_lambda_prior':
                pass
            case 'Approx_GAP_mult_cw':
                pass
            case 'Approx_GAP_mult_mean_prior':
                pass
            case 'Approx_GAP_mult_mean_lambda_prior':
                pass
            case 'Approx_GAP_mult_std_prior':
                pass
            case 'Approx_GAP_mult_full_prior':
                pass
            case 'Adam':
                self.user_params_optimizer = torch.optim.Adam(
                    [learning_users_emb],
                    lr=self.config['inner_user_lr']
                )
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

            for t in tqdm(range(self.config['n_query']), total=self.config['n_query'], disable=self.config['disable_tqdm']):

                # Select the action (question to submit)
                options = test_query_env.get_query_options(t)
                actions = self.select_action(options)
                test_query_env.update(actions, t)

                learning_users_emb = self.inner_loop(test_query_env.feed_IMPACT_sub(),meta_data,learning_users_emb)

                with torch.no_grad() :
                    preds = self.CDM.forward(users_id=meta_data['users_id'], items_id=meta_data['questions_id'], concepts_id=meta_data['categories_id'], users_emb=learning_users_emb)

                pred_list[t].append(preds)
                label_list[t].append(meta_data['labels'])
                nb_modalities_list[t].append(nb_modalities)

                emb_tensor[t, :, :] = learning_users_emb

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
        learning_users_emb = self.CDM.reset_train_valid_users(train_dataset, valid_dataset)

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
                self.meta_optimizer = torch.optim.Adam([self.meta_params], lr=0.01) #todo : wrap in a correct module
                self.meta_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.meta_optimizer, patience=2, factor=0.5)
                self.meta_scaler = torch.amp.GradScaler(self.config['device'])

            case 'Approx_GAP':
                self.meta_optimizer = torch.optim.Adam([self.meta_params], lr=0.5) #todo : wrap in a correct module
                self.meta_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.meta_optimizer, patience=2, factor=0.5)
                self.meta_scaler = torch.amp.GradScaler(self.config['device'])

            case 'Approx_GAP_cw':
                self.meta_optimizer = torch.optim.Adam([self.meta_params], lr=0.5) #todo : wrap in a correct module
                self.meta_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.meta_optimizer, patience=2, factor=0.5)
                self.meta_scaler = torch.amp.GradScaler(self.config['device'])

            case 'Approx_GAP_mult_cw':
                self.meta_optimizer = torch.optim.Adam([self.meta_params], lr=0.5) #todo : wrap in a correct module
                self.meta_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.meta_optimizer, patience=2, factor=0.5)
                self.meta_scaler = torch.amp.GradScaler(self.config['device'])

            case 'Approx_GAP_mult':
                self.meta_optimizer = torch.optim.Adam([self.meta_params], lr=0.5) #todo : wrap in a correct module
                self.meta_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.meta_optimizer, patience=2, factor=0.5)
                self.meta_scaler = torch.amp.GradScaler(self.config['device'])

            case 'Approx_GAP_mult_lambda_prior':
                self.meta_lambda = torch.nn.Parameter(
                    inverse_softplus(torch.tensor(self.config['lambda'], device=self.device, dtype=torch.float32).clone()),
                    requires_grad=True
                )
                # You can pass parameter groups with individual learning rates:
                self.meta_optimizer = torch.optim.Adam(
                    [
                        {'params': self.meta_params,  'lr': self.config.get('meta_params_lr', 0.5)},
                        {'params': self.meta_lambda,  'lr': self.config.get('meta_lambda_lr', 0.5)},
                    ]
                )
                self.meta_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.meta_optimizer, patience=2, factor=0.5)
                self.meta_scaler = torch.amp.GradScaler(self.config['device'])

            case 'Approx_GAP_mult_mean_prior':
                self.meta_mean = torch.nn.Parameter(
                    inverse_softplus(torch.tensor([1.0],
                         device=self.config['device'],
                         dtype=torch.float32)),
                    requires_grad=True
                )
                
                self.meta_optimizer = torch.optim.Adam(
                    [
                        {'params': self.meta_params,  'lr': self.config.get('meta_params_lr', 0.5)},
                        {'params': self.meta_mean,  'lr': self.config.get('meta_mean_lr', 0.5)},
                    ]
                )
                self.meta_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.meta_optimizer, patience=2, factor=0.5)
                self.meta_scaler = torch.amp.GradScaler(self.config['device'])

            case 'Approx_GAP_mult_mean_lambda_prior':
                self.meta_mean = torch.nn.Parameter(
                    self.CDM.model.prior_mean.clone(),
                    requires_grad=True
                )
                self.meta_lambda = torch.nn.Parameter(
                    inverse_softplus(torch.tensor(self.config['lambda'], device=self.config['device'], dtype=torch.float32)),
                    requires_grad=True
                )

                def get_regularizer_with_learnable_mean_lambda_prior(self,unique_users, unique_items,users_emb):   
                    A = (users_emb[unique_users] - self.meta_mean) 
                    S = self.CDM.model.prior_cov_inv
                    SA_T = torch.matmul(A, S)  
                    
                    return torch.bmm(SA_T.unsqueeze(1), A.unsqueeze(2)).sum()   
    
                self.CDM.get_regularizer = functools.partial(get_regularizer_with_learnable_mean_lambda_prior,self)

                
                # You can pass parameter groups with individual learning rates:
                self.meta_optimizer = torch.optim.Adam(
                    [
                        {'params': self.meta_params,  'lr': self.config.get('meta_params_lr', 0.5)},
                        {'params': self.meta_mean,  'lr': self.config.get('meta_mean_lr', 0.001)},
                        {'params': self.meta_lambda,  'lr': self.config.get('meta_lambda_lr', 0.001)},
                    ]
                )
                
                # Use single scheduler for the combined optimizer
                self.meta_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.meta_optimizer, patience=2, factor=0.5)
                self.meta_scaler = torch.amp.GradScaler(self.config['device'])

            case 'Approx_GAP_mult_std_prior':
                
                self.prior_std = torch.nn.Parameter(
                    self.CDM.model.prior_cov_inv.clone(),
                    requires_grad=True
                )

                def get_regularizer_with_learnable_std_prior(self,unique_users, unique_items, users_emb):   

                    A = (users_emb[unique_users] - self.CDM.model.prior_mean) 
                    SA_T = torch.matmul(A, torch.nn.Softplus()(self.prior_std))  
                    
                    return torch.bmm(SA_T.unsqueeze(1), A.unsqueeze(2)).sum()
    
                self.CDM.get_regularizer = functools.partial(get_regularizer_with_learnable_std_prior,self)

                self.meta_optimizer = torch.optim.Adam([self.prior_std,self.meta_params], lr=0.5) #todo : wrap in a correct module
                self.meta_optimizer = torch.optim.Adam(
                    [
                        {'params': self.meta_params,  'lr': self.config.get('meta_params_lr', 0.5)},
                        {'params': self.prior_std,  'lr': self.config.get('meta_std_lr', 0.05)},
                    ]
                )
                self.meta_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.meta_optimizer, patience=2, factor=0.5)
                self.meta_scaler = torch.amp.GradScaler(self.config['device'])

            case 'Approx_GAP_mult_full_prior':

                self.meta_mean = torch.nn.Parameter(
                    self.CDM.model.prior_mean.clone(),
                    requires_grad=True
                )

                self.prior_std = torch.nn.Parameter(
                    self.CDM.model.prior_cov_inv.clone(),
                    requires_grad=True
                )

                def get_regularizer_with_learnable_prior(self,unique_users, unique_items, users_emb):    
                    A = (users_emb[unique_users] - self.meta_mean) 
                    SA_T = torch.matmul(A, torch.nn.Softplus()(self.prior_std))  
                    
                    return torch.bmm(SA_T.unsqueeze(1), A.unsqueeze(2)).sum()

                self.CDM.get_regularizer = functools.partial(get_regularizer_with_learnable_prior,self)

                self.meta_optimizer = torch.optim.Adam(
                    [
                        {'params': self.meta_params,  'lr': self.config.get('meta_params_lr', 0.5)},
                        {'params': self.prior_std,  'lr': self.config.get('meta_std_lr', 0.001)},
                        {'params': self.meta_mean,  'lr': self.config.get('meta_mean_lr', 0.001)},
                    ]
                )
                self.meta_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.meta_optimizer, patience=2, factor=0.5)
                self.meta_scaler = torch.amp.GradScaler(self.config['device'])
            
            case 'Adam':
                self.user_params_optimizer = torch.optim.Adam(
                    [learning_users_emb],
                    lr=self.config['inner_user_lr']
                )
                self.user_params_scaler = torch.amp.GradScaler(self.CDM.config['device'])
            case 'MAML':
                logging.warning("The prior needs to be canged !")
                self.user_params_optimizer = torch.optim.Adam(
                    [learning_users_emb],
                    lr=self.config['inner_user_lr']
                )
            case 'none':
                pass
            case _:
                raise ValueError(f"Unknown meta trainer: {self.config['meta_trainer']}")
            
        # Initialize early stopping parameters
        self.best_epoch = 0
        self.best_valid_loss = float('inf')

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
        
        self.train_meta_doa = lambda u,q : utils.train_meta_doa(self.CDM.model.users_emb.weight.detach(), learning_users_emb.detach(), train_dataset.log_tensor+ valid_dataset.log_tensor, train_dataset.log_tensor + valid_dataset.log_tensor, u, q, train_dataset.metadata, train_dataset.concept_map)
        
        orig_emb = learning_users_emb.detach().clone()

        for _, ep in tqdm(enumerate(range(epochs + 1)), total=epochs, disable=self.config['disable_tqdm']):

            logging.debug(f'------- Epoch : {ep}')

            train_dataset.set_query_seed(ep) # changes the meta and query set for each epoch
            
            learning_users_emb = orig_emb.clone().requires_grad_(True)

            mean_meta_loss = 0.0

            for u_batch, batch in enumerate(train_loader): # UserCollate directly load the data into the query environment
                
                logging.debug(f'----- User batch : {u_batch}')

                train_query_env.load_batch(batch)

                # Prepare the meta set
                meta_data = train_query_env.generate_IMPACT_meta()

                for i_query in range(self.config['n_query']):

                    

                    logging.debug(f'--- Query nb : {i_query}')

                    actions = self.select_action(train_query_env.get_query_options(i_query))
                    train_query_env.update(actions, i_query)

                    i_query_data = train_query_env.feed_IMPACT_sub()
                
                    learning_users_emb = self.inner_loop(i_query_data,meta_data,learning_users_emb)
                    
                L1, L3, R = self.CDM._compute_loss(users_id=meta_data["users_id"], items_id=meta_data["questions_id"], concepts_id=meta_data["categories_id"], labels=meta_data["labels"], learning_users_emb=learning_users_emb)
                losses = torch.stack([L1, L3])  # Shape: (4,)
                
                meta_loss = torch.dot(self.weights, losses) 
                mean_meta_loss += meta_loss / len(batch)
                    
            if self.config['meta_trainer'] != 'Adam':
                self.update_meta_params(mean_meta_loss)
            #self.update_S_params(meta_loss)
            #self.update_CDM_params(meta_loss)

            with torch.no_grad():
                learning_users_emb.copy_(orig_emb)
            learning_users_emb.requires_grad_(True)

            # Early stopping
            if (ep + 1) % eval_freq == 0:
                with torch.no_grad(), torch.amp.autocast('cuda'):
                    valid_loss = self.evaluate_valid(valid_loader, valid_query_env,learning_users_emb)

                    logging.info(f'valid_loss : {valid_loss}')

                    with warnings.catch_warnings(record=True) as w:
                        warnings.simplefilter("always")

                    self.S_scheduler.step(valid_loss)
                    if hasattr(self, 'meta_scheduler') and self.meta_scheduler is not None:
                        self.meta_scheduler.step(valid_loss)


                    #TODO : add CDM scheduler

                    # Checking loss improvement
                    if  self.best_valid_loss > valid_loss:  # (self.best_valid_metric - valid_rmse) / abs(self.best_valid_metric) > 0.001:
                        self.best_epoch = ep
                        self.best_valid_loss = valid_loss
                        self.best_model_params = {'state_dict': self.model.state_dict()}
                        if self.meta_params is not None:
                            self.best_model_params['meta_params'] = self.meta_params.detach().clone()
                        if hasattr(self, 'meta_mean') :
                            self.best_model_params['meta_mean'] = self.meta_mean.detach().clone()
                        if hasattr(self, 'meta_lambda') :
                            self.best_model_params['meta_lambda'] = self.meta_lambda.detach().clone()

                    if ep - self.best_epoch >= patience:
                        break

        self.model.load_state_dict(self.best_model_params['state_dict'])
        if self.meta_params is not None:
            self.meta_params = self.best_model_params['meta_params'].requires_grad_()
        if hasattr(self, 'meta_mean') :
            self.meta_mean = self.best_model_params['meta_mean'].requires_grad_()
        if hasattr(self, 'meta_lambda') :
            self.meta_lambda = self.best_model_params['meta_lambda'].requires_grad_()


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


