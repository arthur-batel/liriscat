import json
import numpy as np
import torch
import random
import pandas as pd
import numpy as np
from math import pow
import warnings

import logging
import sys
from datetime import datetime



def setuplogger(verbose: bool = True, log_path: str = "../../experiments/logs/", log_name: str = None):
    root = logging.getLogger()
    if verbose:
        root.setLevel(logging.INFO)
    else:
        root.setLevel(logging.ERROR)

    # Stream handler for console output
    stream_handler = logging.StreamHandler(sys.stdout)
    if verbose:
        stream_handler.setLevel(logging.INFO)
    else:
        stream_handler.setLevel(logging.ERROR)
    formatter = logging.Formatter("[%(levelname)s %(asctime)s] %(message)s")
    formatter.default_time_format = "%M:%S"
    formatter.default_msec_format = ""
    stream_handler.setFormatter(formatter)

    # Remove existing handlers
    for handler in root.handlers[:]:
        root.removeHandler(handler)

    if log_name is not None:
        now = datetime.now()
        time_str = now.strftime("_%d:%m:%y_%S:%M")
        file_handler = logging.FileHandler(log_path + log_name + time_str + ".log")

        if verbose:
            file_handler.setLevel(logging.INFO)
        else:
            file_handler.setLevel(logging.ERROR)
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    # Add new handlers
    root.addHandler(stream_handler)


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        print("CUDA is not available. Skipping CUDA seed setting.")

def _generate_config(dataset_name: str = None, seed: int = 0, load_params: bool = False,
                     save_params: bool = False, embs_path: str = '../embs/',
                     params_path: str = '../ckpt/', early_stopping: bool = True, esc: str = 'error',
                     verbose_early_stopping: str = False, disable_tqdm: bool = True,
                     valid_metric: str = 'rmse', learning_rate: float = 0.001, batch_size: int = 2048,
                     num_epochs: int = 200, eval_freq: int = 1, patience: int = 30,
                     device: str = None, lambda_: float = 7.7e-6, tensorboard: bool = False,
                     flush_freq: bool = True, pred_metrics: list = ['rmse'], profile_metrics: list = ['doa'],
                     num_responses: int = 12, low_mem: bool = False, n_query: int = 10, CDM:str = 'impact') -> dict:
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print("CUDA is available. Using GPU.")
        else:
            device = torch.device("cpu")
            print("CUDA is not available. Using CPU.")
    return {
        'seed': seed,
        'dataset_name': dataset_name,
        'load_params': load_params,
        'save_params': save_params,
        'embs_path': embs_path,
        'params_path': params_path,
        'early_stopping': early_stopping,
        'esc': esc,
        'verbose_early_stopping': verbose_early_stopping,
        'disable_tqdm': disable_tqdm,
        'valid_metric': valid_metric,
        'learning_rate': learning_rate,
        'batch_size': batch_size,
        'num_epochs': num_epochs,
        'eval_freq': eval_freq,
        'patience': patience,
        'device': device,
        'lambda': lambda_,
        'tensorboard': tensorboard,
        'flush_freq': flush_freq,
        'pred_metrics': pred_metrics,
        'profile_metrics': profile_metrics,
        'num_responses': num_responses,
        'low_mem': low_mem,
        'n_query': n_query,
        'CDM': CDM
    }

def generate_hs_config(dataset_name: str = None, seed: int = 0, load_params: bool = False,
                       save_params: bool = False, embs_path: str = '../embs/',
                       params_path: str = '../ckpt/', early_stopping: bool = True, esc: str = 'error',
                       verbose_early_stopping: str = False, disable_tqdm: bool = True,
                       valid_metric: str = 'rmse', learning_rate: float = 0.001, batch_size: int = 2048,
                       num_epochs: int = 200, eval_freq: int = 1, patience: int = 30,
                       device: str = None, lambda_: float = 7.7e-6, tensorboard: bool = False,
                       flush_freq: bool = True, pred_metrics: list = ['rmse'], profile_metrics: list = [],
                       num_responses: int = 12, low_mem: bool = False, n_query: int = 10, CDM:str='impact') -> dict:
    """
        Generate a configuration dictionary for the model hyperparameter search process.

        Args:
            dataset_name (str): Name of the dataset. Default is None.
            seed (int): Random seed for reproducibility. Default is 0.
            load_params (bool): Whether to load model parameters from a file. Default is False.
            save_params (bool): Whether to save model parameters to a file. Default is False.
            embs_path (str): Path to the directory where embeddings will be saved. Default is '../embs/'.
            params_path (str): Path to the directory where model parameters will be saved. Default is '../ckpt/'.
            early_stopping (bool): Whether to use early stopping during training. Default is True.
            esc (str): Early stopping criterion. Possible values: 'error', 'loss', 'delta_error', 'objectives'. Default is 'error'.
            verbose_early_stopping (str): Whether to print model learning statistics during training (frequency = eval_freq). Default is False.
            disable_tqdm (bool): Whether to disable tqdm progress bars. Default is True.
            valid_metric (str): Metric to be used for hyperparameters selection on the valid dataset (including early stopping). Possible values: 'rmse', 'mae', 'mi_acc'. Default is 'rmse'.
            learning_rate (float): Learning rate for the optimizer. Default is 0.001.
            batch_size (int): Batch size for training. Default is 2048.
            num_epochs (int): Number of epochs for training. (Maximum number if early stopping) Default is 200.
            eval_freq (int): Frequency of evaluation during training. Default is 1.
            patience (int): Patience for early stopping. Default is 30.
            device (str): Device to be used for training (e.g., 'cpu' or 'cuda'). Default is None.
            lambda_ (float): Regularization parameter. Default is 7.7e-6.
            tensorboard (bool): Whether to use TensorBoard for logging. Default is False.
            flush_freq (bool): Whether to flush the TensorBoard logs frequently. Default is True.
            pred_metrics (list): List of prediction metrics to be used for evaluation. Possible list elements: 'rmse', 'mae', 'r2', 'mi_acc', 'mi_prec', 'mi_rec', 'mi_f1', 'mi_auc' (mi = micro-averaged). Default is ['rmse', 'mae'].
            profile_metrics (list): List of profile metrics to be used for evaluation. Possible list elements: 'doa', 'pc-er', 'rm'. Default is [].
            num_responses (int): Number of responses IMPACT will use for each question in the case of dataset with continuous values. For discrete datasets, num_responses is the MAXIMUM number of responses IMPACT will use for each question. Default is 12.
            low_mem (bool): Whether to enable low memory mode for IMPACT with vector subspaces for question-response embeddings. Default is False.
            n_query (int) : Number of question to submit to users. Default is 10.
            CDM (str): Name of the CDM to be used. Default is 'impact'.
        Returns:
            dict: Configuration dictionary with the specified parameters.
        """
    return _generate_config(dataset_name, seed, load_params, save_params, embs_path, params_path,
                            early_stopping, esc, verbose_early_stopping, disable_tqdm,
                            valid_metric, learning_rate, batch_size, num_epochs, eval_freq, patience, device,
                            lambda_, tensorboard, flush_freq, pred_metrics, profile_metrics,
                            num_responses, low_mem, n_query)

def generate_eval_config(dataset_name: str = None, seed: int = 0, load_params: bool = False,
                         save_params: bool = True, embs_path: str = '../embs/',
                         params_path: str = '../ckpt/', early_stopping: bool = True, esc: str = 'error',
                         verbose_early_stopping: str = False, disable_tqdm: bool = False,
                         valid_metric: str = 'rmse', learning_rate: float = 0.001, batch_size: int = 2048,
                         num_epochs: int = 200, eval_freq: int = 1, patience: int = 30,
                         device: str = None, lambda_: float = 7.7e-6, tensorboard: bool = False,
                         flush_freq: bool = True, pred_metrics: list = ['rmse', 'mae', 'r2'],
                         profile_metrics: list = ['doa', 'pc-er'],
                         num_responses: int = 12, low_mem: bool = False, n_query: int = 10, CDM:str='impact') -> dict:
    """
        Generate a configuration dictionary for the model evaluation.

        Args:
            dataset_name (str): Name of the dataset. Default is None.
            seed (int): Random seed for reproducibility. Default is 0.
            load_params (bool): Whether to load model parameters from a file. Default is False.
            save_params (bool): Whether to save model parameters to a file. Default is True.
            embs_path (str): Path to the directory where embeddings will be saved. Default is '../embs/'.
            params_path (str): Path to the directory where model parameters will be saved. Default is '../ckpt/'.
            early_stopping (bool): Whether to use early stopping during training. Default is True.
            esc (str): Early stopping criterion. Possible values: 'error', 'loss', 'delta_error', 'objectives'. Default is 'error'.
            verbose_early_stopping (str): Whether to print model learning statistics during training (frequency = eval_freq). Default is False.
            disable_tqdm (bool): Whether to disable tqdm progress bars. Default is False.
            valid_metric (str): Metric to be used for hyperparameters selection on the valid dataset (including early stopping). Possible values: 'rmse', 'mae', 'mi_acc'. Default is 'rmse'.
            learning_rate (float): Learning rate for the optimizer. Default is 0.001.
            batch_size (int): Batch size for training. Default is 2048.
            num_epochs (int): Number of epochs for training. (Maximum number if early stopping) Default is 200.
            eval_freq (int): Frequency of evaluation during training. Default is 1.
            patience (int): Patience for early stopping. Default is 30.
            device (str): Device to be used for training (e.g., 'cpu' or 'cuda'). Default is None.
            lambda_ (float): Regularization parameter. Default is 7.7e-6.
            tensorboard (bool): Whether to use TensorBoard for logging. Default is False.
            flush_freq (bool): Whether to flush the TensorBoard logs frequently. Default is True.
            pred_metrics (list): List of prediction metrics to be used for evaluation. Possible list elements: 'rmse', 'mae', 'r2', 'mi_acc', 'mi_prec', 'mi_rec', 'mi_f1', 'mi_auc' (mi = micro-averaged). Default is ['rmse', 'mae'].
            profile_metrics (list): List of profile metrics to be used for evaluation. Possible list elements: 'doa', 'pc-er', 'rm'. Default is ['doa', 'pc-er'].
            num_responses (int): Number of responses IMPACT will use for each question in the case of dataset with continuous values. For discrete datasets, num_responses is the MAXIMUM number of responses IMPACT will use for each question. Default is 12.
            low_mem (bool): Whether to enable low memory mode for IMPACT with vector subspaces for question-response embeddings. Default is False.
            n_query (int) : Number of question to submit to users. Default is 10.
            CDM (str): Name of the CDM to be used. Default is 'impact'.
        Returns:
            dict: Configuration dictionary with the specified parameters.
        """
    return _generate_config(dataset_name, seed, load_params, save_params, embs_path, params_path,
                            early_stopping, esc, verbose_early_stopping, disable_tqdm,
                            valid_metric, learning_rate, batch_size, num_epochs, eval_freq, patience, device,
                            lambda_, tensorboard, flush_freq, pred_metrics, profile_metrics,
                            num_responses, low_mem,n_query)