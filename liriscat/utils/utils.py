import json
from turtledemo.forest import start

import numba
import numpy as np
import torch
import random
import pandas as pd
import numpy as np
from math import pow
import warnings
from torch.utils import data

import logging
import sys
from datetime import datetime

from sklearn.metrics import roc_auc_score


def setuplogger(verbose: bool = True, debug: bool=False, log_path: str = "../../experiments/logs/", log_name: str = None, os: str = 'Linux'):

    if os == 'Windows':
        root = logging.getLogger()
        
        if debug : 
            root.setLevel(logging.DEBUG)
        elif verbose:
            root.setLevel(logging.INFO)
        else:
            root.setLevel(logging.ERROR)

        # Stream handler for console output
        stream_handler = logging.StreamHandler(sys.stdout)
        
        if debug : 
            stream_handler.setLevel(logging.DEBUG)
        elif verbose:
            stream_handler.setLevel(logging.INFO)
        else:
            stream_handler.setLevel(logging.ERROR)
        formatter = logging.Formatter("[%(levelname)s %(asctime)s] %(message)s")
        formatter.default_time_format = "%M-%S"
        formatter.default_msec_format = ""
        stream_handler.setFormatter(formatter)

        # Remove existing handlers
        for handler in root.handlers[:]:
            root.removeHandler(handler)

        if log_name is not None:
            now = datetime.now()
            time_str = now.strftime("_%d-%m-%y_%S-%M")
            file_handler = logging.FileHandler(log_path + log_name + time_str + ".log")

            if debug : 
                file_handler.setLevel(logging.DEBUG)
            elif verbose:
                file_handler.setLevel(logging.INFO)
            else:
                file_handler.setLevel(logging.ERROR)
            file_handler.setFormatter(formatter)
            root.addHandler(file_handler)

        # Add new handlers
        root.addHandler(stream_handler)
    elif os == 'Linux':
        root = logging.getLogger()
        if debug : 
            root.setLevel(logging.DEBUG)
        elif verbose:
            root.setLevel(logging.INFO)
        else:
            root.setLevel(logging.ERROR)

        # Stream handler for console output
        stream_handler = logging.StreamHandler(sys.stdout)
        if debug : 
            stream_handler.setLevel(logging.DEBUG)
        elif verbose:
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

            if debug : 
                file_handler.setLevel(logging.DEBUG)
            elif verbose:
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
        torch.use_deterministic_algorithms(True)
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
                     valid_batch_size: int = 10000,
                     num_epochs: int = 200, eval_freq: int = 1, patience: int = 30,
                     device: str = None, lambda_: float = 7.7e-6, tensorboard: bool = False,
                     flush_freq: bool = True, pred_metrics: list = ['rmse'], profile_metrics: list = ['doa'],
                     num_responses: int = 12, low_mem: bool = False, n_query: int = 10, CDM: str = 'impact',
                     i_fold: int = 0, num_inner_users_epochs: int = 10, num_inner_epochs: int = 10,
                     inner_lr: float = 0.0001, inner_user_lr: float = 0.0001) -> dict:
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
        'valid_batch_size': valid_batch_size,
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
        'CDM': CDM,
        'i_fold': i_fold,
        'num_inner_users_epochs': num_inner_users_epochs,
        'num_inner_epochs': num_inner_epochs,
        "inner_lr": inner_lr,
        "inner_user_lr": inner_user_lr,
    }


def generate_hs_config(dataset_name: str = None, seed: int = 0, load_params: bool = False,
                       save_params: bool = False, embs_path: str = '../embs/',
                       params_path: str = '../ckpt/', early_stopping: bool = True, esc: str = 'error',
                       verbose_early_stopping: str = False, disable_tqdm: bool = True,
                       valid_metric: str = 'rmse', learning_rate: float = 0.001, batch_size: int = 2048,
                       valid_batch_size: int = 10000,
                       num_epochs: int = 200, eval_freq: int = 1, patience: int = 30,
                       device: str = None, lambda_: float = 7.7e-6, tensorboard: bool = False,
                       flush_freq: bool = True, pred_metrics: list = ['rmse'], profile_metrics: list = [],
                       num_responses: int = 12, low_mem: bool = False, n_query: int = 10, CDM: str = 'impact',
                       i_fold: int = 0, num_inner_users_epochs: int = 10, num_inner_epochs: int = 10,
                       inner_lr: float = 0.0001, inner_user_lr: float = 0.0001) -> dict:
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
            valid_batch_size (int): Batch size for validation. Default is 10000.
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
            i_fold (int): Fold number for cross-validation. Default is 0.
        Returns:
            dict: Configuration dictionary with the specified parameters.
        """
    return _generate_config(dataset_name=dataset_name, seed=seed, load_params=load_params, save_params=save_params,
                            embs_path=embs_path, params_path=params_path,
                            early_stopping=early_stopping, esc=esc, verbose_early_stopping=verbose_early_stopping,
                            disable_tqdm=disable_tqdm,
                            valid_metric=valid_metric, learning_rate=learning_rate, batch_size=batch_size,
                            valid_batch_size=valid_batch_size, num_epochs=num_epochs, eval_freq=eval_freq,
                            patience=patience, device=device,
                            lambda_=lambda_, tensorboard=tensorboard, flush_freq=flush_freq, pred_metrics=pred_metrics,
                            profile_metrics=profile_metrics,
                            num_responses=num_responses, low_mem=low_mem, n_query=n_query, CDM=CDM, i_fold=i_fold,
                            num_inner_users_epochs=num_inner_users_epochs, num_inner_epochs=num_inner_epochs,
                            inner_lr=inner_lr, inner_user_lr=inner_user_lr)


def generate_eval_config(dataset_name: str = None, seed: int = 0, load_params: bool = False,
                         save_params: bool = True, embs_path: str = '../embs/',
                         params_path: str = '../ckpt/', early_stopping: bool = True, esc: str = 'error',
                         verbose_early_stopping: str = False, disable_tqdm: bool = False,
                         valid_metric: str = 'rmse', learning_rate: float = 0.001, batch_size: int = 2048,
                         valid_batch_size: int = 10000,
                         num_epochs: int = 200, eval_freq: int = 1, patience: int = 30,
                         device: str = None, lambda_: float = 7.7e-6, tensorboard: bool = False,
                         flush_freq: bool = True, pred_metrics: list = ['rmse', 'mae', 'r2'],
                         profile_metrics: list = ['doa', 'pc-er'],
                         num_responses: int = 12, low_mem: bool = False, n_query: int = 10, CDM: str = 'impact',
                         i_fold: int = 0, num_inner_users_epochs: int = 10, num_inner_epochs: int = 10,
                         inner_lr: float = 0.0001, inner_user_lr: float = 0.0001) -> dict:
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
            valid_batch_size (int): Batch size for validation. Default is 10000.
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
            i_fold (int): Fold number for cross-validation. Default is 0.
        Returns:
            dict: Configuration dictionary with the specified parameters.
        """
    return _generate_config(dataset_name=dataset_name, seed=seed, load_params=load_params, save_params=save_params,
                            embs_path=embs_path, params_path=params_path,
                            early_stopping=early_stopping, esc=esc, verbose_early_stopping=verbose_early_stopping,
                            disable_tqdm=disable_tqdm,
                            valid_metric=valid_metric, learning_rate=learning_rate, batch_size=batch_size,
                            valid_batch_size=valid_batch_size, num_epochs=num_epochs, eval_freq=eval_freq,
                            patience=patience, device=device,
                            lambda_=lambda_, tensorboard=tensorboard, flush_freq=flush_freq, pred_metrics=pred_metrics,
                            profile_metrics=profile_metrics,
                            num_responses=num_responses, low_mem=low_mem, n_query=n_query, CDM=CDM, i_fold=i_fold,
                            num_inner_users_epochs=num_inner_users_epochs, num_inner_epochs=num_inner_epochs,
                            inner_lr=inner_lr, inner_user_lr=inner_user_lr)


def convert_config_to_EduCAT(config, metadata, strategy_name: str, threshold: float = None, betas: float = None,
                             start=None, end=None, policy_path=None, prednet_len1=None, prednet_len2=None,
                             meta_param=None, available_mask=None, train_mask=None, mode=None, epoch=None):
    config['num_dim'] = metadata['num_dimension_id']
    config['policy'] = strategy_name
    # For NCAT
    config['THRESHOLD'] = threshold
    config['start'] = start
    config['end'] = end

    # For BOBCAT
    config['betas'] = betas
    config['policy_path'] = policy_path
    config['meta_param'] = meta_param
    config['available_mask'] = available_mask
    config["train_mask"] = train_mask
    config["mode"] = mode
    config["epoch"] = epoch

    # For NCD
    config['prednet_len1'] = prednet_len1
    config['prednet_len2'] = prednet_len2


def evaluate_doa(E, R, metadata, concept_map):
    q = {}
    for r in range(metadata['num_item_id']):
        q[r] = []

    for u, i in torch.tensor(R).nonzero():
        q[i.item()].append(u.item())

    max_concepts_per_item = 0
    list_concept_map = []
    for d in concept_map:
        list_concept_map.append(concept_map[d])
        l = len(concept_map[d])
        if l > max_concepts_per_item:
            max_concepts_per_item = l

    list_q = []
    list_q_len = []
    for key in q.keys():
        list_q.append(q[key])
        list_q_len.append(len(q[key]))

    max_q_len = max(len(q_i) for q_i in list_q)
    q_array = _preprocess_list_q(list_q, max_q_len)
    concept_map_array = _preprocess_concept_map(list_concept_map, max_concepts_per_item)

    # Convert q_len to a NumPy array
    q_len = np.array(list_q_len, dtype=np.int32)

    num_dim = metadata['num_dimension_id']

    # Optionally ensure concept indices are in range inside _compute_doa:
    # You can either filter concept_indices there or ensure _preprocess_concept_map
    # doesn't produce out-of-range indices.

    return _compute_doa(q_array, q_len, num_dim, E, concept_map_array, R)


@numba.jit(nopython=True, cache=True)
def _compute_doa(q, q_len, num_dim, E, concept_map_array, R):
    s = np.zeros(shape=(1, num_dim))
    beta = np.zeros(shape=(1, num_dim))

    for i in range(len(q)):  # Adjusted to loop over indices
        concept_indices = concept_map_array[i]
        concept_indices = concept_indices[(concept_indices >= 0) & (concept_indices < num_dim)]

        E_i = E[:, concept_indices]  # Index E using NumPy array
        q_i_len = q_len[i]

        for u_i in range(q_i_len - 1):
            u = q[i, u_i]
            for v in q[i, u_i + 1:q_i_len]:
                if R[u, i] > R[v, i]:
                    for idx in range(len(concept_indices)):
                        s[0, concept_indices[idx]] += E_i[u, idx] > E_i[v, idx]
                        beta[0, concept_indices[idx]] += E_i[u, idx] != E_i[v, idx]
                elif R[u, i] < R[v, i]:
                    for idx in range(len(concept_indices)):
                        s[0, concept_indices[idx]] += E_i[u, idx] < E_i[v, idx]
                        beta[0, concept_indices[idx]] += E_i[u, idx] != E_i[v, idx]

    # Avoid division by zero
    for idx in range(num_dim):
        if beta[0, idx] == 0:
            beta[0, idx] = 1

    return s / beta


def _preprocess_list_q(list_q, max_len):
    q_array = -np.ones((len(list_q), max_len), dtype=np.int64)  # Initialize with -1 for padding
    for i, q_i in enumerate(list_q):
        q_array[i, :len(q_i)] = q_i  # Copy each list q_i into the array, pad with -1 if shorter
    return q_array


# Helper function to convert list of lists into padded NumPy array
def _preprocess_concept_map(list_concept_map, max_len):
    concept_map_array = -np.ones((len(list_concept_map), max_len), dtype=np.int64)  # Initialize with -1
    for i, concepts in enumerate(list_concept_map):
        concept_map_array[i, :len(concepts)] = concepts  # Copy valid values into array
    return concept_map_array


def compute_doa(emb: torch.Tensor, test_data):
    return np.mean(evaluate_doa(emb.cpu().numpy(), test_data.meta_tensor.cpu().numpy(), test_data.metadata,
                                test_data.concept_map))


def compute_pc_er(emb, test_data):
    U_resp_sum = torch.zeros(size=(test_data.n_users, test_data.n_categories)).to(test_data.raw_data_array.device,
                                                                                  non_blocking=True)
    U_resp_nb = torch.zeros(size=(test_data.n_users, test_data.n_categories)).to(test_data.raw_data_array.device,
                                                                                 non_blocking=True)

    data_loader = data.DataLoader(test_data, batch_size=1, shuffle=False, num_workers=0)
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
    logging.warning("Computing RM on meta and QUERY set")
    concept_array, concept_lens = preprocess_concept_map(test_data.concept_map)
    r = compute_rm_fold(emb.cpu().numpy(), test_data.df.to_records(index=False,
                                                                   column_dtypes={'student_id': int, 'item_id': int,
                                                                                  "correct": float,
                                                                                  "dimension_id": int}), concept_array,
                        concept_lens)
    logging.info(f"RM: {r}")
    return r


def preprocess_concept_map(concept_map):
    # concept_map: dict[item_id -> list of concept_ids]
    items = sorted(concept_map.keys())
    max_len = max(len(v) for v in concept_map.values())

    concept_array = np.full((max(items) + 1, max_len), -1, dtype=np.int32)
    concept_lens = np.zeros(max(items) + 1, dtype=np.int32)

    for k, v in concept_map.items():
        v = np.array(v, dtype=np.int32)
        concept_array[k, :len(v)] = v
        concept_lens[k] = len(v)

    return concept_array, concept_lens


@numba.njit
def compute_rm_fold(emb, d, concept_array, concept_lens):
    # emb: (n_users, n_dims)
    # d: structured array with fields (student_id, item_id, correct, concept_id)
    # concept_array, concept_lens: from preprocess

    n_users, n_dims = emb.shape
    U_resp_sum = np.zeros((n_users, n_dims), dtype=np.float64)
    U_resp_nb = np.zeros((n_users, n_dims), dtype=np.float64)

    # Fill U_resp_sum and U_resp_nb
    for rec in d:
        student_id = rec[0]
        item_id = rec[1]
        correct_val = rec[2]
        length = concept_lens[item_id]
        for i in range(length):
            cid = concept_array[item_id, i]
            U_resp_sum[student_id, cid] += correct_val
            U_resp_nb[student_id, cid] += 1.0

    # Compute U_ave = U_resp_sum / U_resp_nb where nb>0 else 0
    U_ave = np.zeros((n_users, n_dims), dtype=np.float64)
    for i in range(n_users):
        for j in range(n_dims):
            if U_resp_nb[i, j] > 0:
                U_ave[i, j] = U_resp_sum[i, j] / U_resp_nb[i, j]
            else:
                U_ave[i, j] = 0.0

    # Build u_array and e_array
    # Condition: user answered any question => at least one dim in U_ave[i_user] != 0.0
    # Also handle NaNs: If any appear, handle them manually
    u_list = []
    e_list = []
    for i_user in range(n_users):
        # Check if user answered any question
        answered_any = False
        for j in range(n_dims):
            val = U_ave[i_user, j]
            # NaN check
            if val != val:  # val != val means val is NaN
                # Replace NaN with 0
                U_ave[i_user, j] = 0.0
            if U_resp_nb[i_user, j] > 0:
                answered_any = True

        if answered_any:
            # Create copies to avoid modifying emb/U_ave arrays directly
            u_vec = U_ave[i_user].copy()
            e_vec = emb[i_user].copy()

            # If a dimension not answered => it's already 0.0 in U_ave
            # Set corresponding dimension in e_vec to 0 if not answered:
            # Actually we already know unanswered are 0 in U_ave. We'll do same for e_vec:
            for j in range(n_dims):
                if U_resp_nb[i_user, j] == 0:
                    e_vec[j] = 0.0

            u_list.append(u_vec)
            e_list.append(e_vec)

    if len(u_list) == 0:
        # No users answered anything
        return np.nan

    # Convert lists to arrays
    # Numba cannot directly convert list of arrays if their shape is known.
    # But here each element should have the same shape: (n_dims,)
    # We'll allocate arrays directly:
    n_users_filtered = len(u_list)
    u_array = np.zeros((n_users_filtered, n_dims), dtype=np.float64)
    e_array = np.zeros((n_users_filtered, n_dims), dtype=np.float64)
    for i in range(n_users_filtered):
        for j in range(n_dims):
            u_array[i, j] = u_list[i][j]
            e_array[i, j] = e_list[i][j]

    c = 0.0
    s = 0

    # For each dimension:
    # We must extract the users who answered this dimension (u_array[i,dim] !=0)
    # Then compute covariance and do sorting logic
    for dim in range(n_dims):
        # Count how many users answered this dim
        count_true = 0
        for i_user in range(n_users_filtered):
            if u_array[i_user, dim] != 0.0:
                count_true += 1

        if count_true == 0:
            continue

        # Extract those users' responses
        X_u = np.empty(count_true, dtype=np.float64)
        X_e = np.empty(count_true, dtype=np.float64)
        idx_pos = 0
        for i_user in range(n_users_filtered):
            if u_array[i_user, dim] != 0.0:
                X_u[idx_pos] = u_array[i_user, dim]
                X_e[idx_pos] = e_array[i_user, dim]
                idx_pos += 1

        # Compute covariance
        cov_val = compute_cov(X_u, X_e)
        if np.isnan(cov_val):
            continue

        if cov_val > 0:
            # Sort both arrays
            X_u_sorted = np.sort(X_u)
            X_e_sorted = np.sort(X_e)
            cov_star = compute_cov(X_u_sorted, X_e_sorted)
            if np.isnan(cov_star) or cov_star == 0:
                # If cov_star is zero or NaN, handle gracefully
                continue
            rm = cov_val / cov_star
        elif cov_val == 0:
            rm = 0.0
        else:
            # cov_val < 0
            X_u_sorted = np.sort(X_u)
            X_e_sorted = np.sort(X_e)
            X_e_reversed = reverse_array(X_e_sorted)
            cov_prime = compute_cov(X_u_sorted, X_e_reversed)
            if np.isnan(cov_prime) or cov_prime == 0:
                continue
            rm = -cov_val / cov_prime

        c += rm
        s += 1

    if s == 0:
        return np.nan
    return c / s


@numba.njit
def compute_cov(x, y):
    # Compute sample covariance (same as np.cov(x,y)[0,1]) with denominator (N-1)
    n = x.size
    if n < 2:
        return np.nan
    mx = 0.0
    my = 0.0
    for i in range(n):
        mx += x[i]
        my += y[i]
    mx /= n
    my /= n
    s = 0.0
    for i in range(n):
        s += (x[i] - mx) * (y[i] - my)
    return s / (n - 1)


@numba.njit
def reverse_array(arr):
    n = arr.size
    res = np.empty(n, dtype=arr.dtype)
    for i in range(n):
        res[i] = arr[n - 1 - i]
    return res


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
