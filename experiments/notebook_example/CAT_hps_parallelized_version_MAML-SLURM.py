import sys
import os
import optuna
import gc
import torch
import logging
from functools import partial
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["PYTHONHASHSEED"] = "0"
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"

import liriscat
liriscat.utils.set_seed(0)

import logging
import gc
import json
import torch
liriscat.utils.set_seed(0)
import pandas as pd
from liriscat.dataset.preprocessing_utilities import *
from optuna.exceptions import DuplicatedStudyError

import os, torch
print("PID", os.getpid(), "sees CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))
if os.environ.get("CUDA_VISIBLE_DEVICES") is not None :
    print("→ torch.device is", torch.cuda.current_device(), torch.cuda.get_device_name(0))

def launch_test(trial, train_data, valid_data, config):
    gc.collect()
    torch.cuda.empty_cache()

    train_data.reset_rng()
    valid_data.reset_rng()

    S = liriscat.selectionStrategy.Random(train_data.metadata,**config)
    S.init_models(train_data, valid_data)
    S.train(train_data, valid_data)
    liriscat.utils.set_seed(0)
    S.reset_rng()
    d = (S.evaluate_test(valid_data, train_data, valid_data))
    pi = liriscat.utils.pareto_index(d)
    logging.info(f"Trial {trial.number}; Pareto_index: {pi}; Hyperparams: {trial.params}; Results: {d}")

    del S
    gc.collect()
    torch.cuda.empty_cache()

    return pi

def objective_hps(trial, config, train_data, valid_data):
    meta_lr = trial.suggest_float('meta_lr', 0.001, 0.1, log=True)
    inner_user_lr = trial.suggest_float('inner_user_lr', 0.0001, 0.05, log=True)
    
    config['meta_lr'] = meta_lr
    config['inner_user_lr'] = inner_user_lr
    
    return launch_test(trial, train_data, valid_data, config)

def main(dataset_name, nb_trials):
    # SLURM auto-assigns a GPU (index 0 for each task as visible)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    config = liriscat.utils.generate_eval_config(dataset_name=dataset_name,load_params=True, esc = 'error', valid_metric= 'mi_acc', pred_metrics = ["mi_acc"], profile_metrics = ['meta_doa'], save_params=False, n_query=6, num_epochs=100, patience = 20, num_inner_users_epochs=3, meta_trainer='MAML', batch_size=512, lambda_=9.972254466547545e-06, inner_user_lr=0.016848380924625605, num_responses=12)

    config['device'] = device

    concept_map = json.load(open(f'../datasets/2-preprocessed_data/{config["dataset_name"]}_concept_map.json', 'r'))
    concept_map = {int(k): [int(x) for x in v] for k, v in concept_map.items()}
    
    ## Metadata map format : {"num_user_id": ..., "num_item_id": ..., "num_dimension_id": ...}
    metadata = json.load(open(f'../datasets/2-preprocessed_data/{config["dataset_name"]}_metadata.json', 'r'))
    
    ## Tensor containing the nb of modalities per question
    nb_modalities = torch.load(f'../datasets/2-preprocessed_data/{config["dataset_name"]}_nb_modalities.pkl',weights_only=True)

    i_fold =0
    
    train_df = pd.read_csv(
        f'../datasets/2-preprocessed_data/{config["dataset_name"]}_train_{i_fold}.csv',
        encoding='utf-8', dtype={'student_id': int, 'item_id': int, "correct": float,
                                                                "dimension_id": int})
    valid_df = pd.read_csv(
        f'../datasets/2-preprocessed_data/{config["dataset_name"]}_valid_{i_fold}.csv',
        encoding='utf-8', dtype={'student_id': int, 'item_id': int, "correct": float,
                                                                "dimension_id": int})

    train_data = liriscat.dataset.CATDataset(train_df, concept_map, metadata, config,nb_modalities)
    valid_data = liriscat.dataset.EvalDataset(valid_df, concept_map, metadata, config,nb_modalities)

    # Shared SQLite storage accessible by all parallel tasks
    storage_name = f"sqlite:///{dataset_name}_IMPACT_MAML.db"
    study_name = f"hps_parallel_{dataset_name}_IMPACT_MAML"


    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        directions=["maximize"],
        load_if_exists=True,
    )

    objective = partial(
        objective_hps, config=config,
        train_data=train_data,
        valid_data=valid_data,
    )

    study.optimize(objective, n_trials=nb_trials, n_jobs=1, gc_after_trial=True)

    # After optimization, log best results (finalized in any task)
    if study.best_trials:
        print("Best trial(s):")
        for trial in study.best_trials:
            print(f"Trial #{trial.number}: {trial.values}, Params: {trial.params}")

if __name__ == '__main__':
    dataset_name = "math2"
    liriscat.utils.setuplogger(verbose = True, log_name=f"CAT_hps_{dataset_name}", debug=False)
    nb_trials = 25
    main(dataset_name, nb_trials)
