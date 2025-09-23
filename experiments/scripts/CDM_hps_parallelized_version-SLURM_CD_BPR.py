import sys
import os
import optuna
import gc
import torch
import logging
from functools import partial
from IMPACT import utils as utils_IMPACT
from IMPACT.model import IMPACT
from micat.dataset.preprocessing_utilities import *
from optuna.exceptions import DuplicatedStudyError

import os, torch
print("PID", os.getpid(), "sees CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("→ torch.device is", torch.cuda.current_device(), torch.cuda.get_device_name(0))

def launch_test(trial, train_data, valid_data, config):
    gc.collect()
    torch.cuda.empty_cache()

    algo = IMPACT(**config)
    algo.init_model(train_data, valid_data)
    algo.train(train_data, valid_data)

    best_valid_metric = algo.best_valid_metric
    logging.info(f"Trial {trial.number}: Best metric: {best_valid_metric}")

    del algo
    gc.collect()
    torch.cuda.empty_cache()

    return best_valid_metric

def objective_hps(trial, config, train_data, valid_data):
    lr = trial.suggest_float('learning_rate', 1e-5, 5e-2, log=True)
    lambda_param = trial.suggest_float('lambda', 1e-8, 1e-4, log=True)

    config['learning_rate'] = lr
    config['lambda'] = lambda_param
    config['num_responses'] = 12

    return launch_test(trial, train_data, valid_data, config)

def main(dataset_name, nb_trials):
    # SLURM auto-assigns a GPU (index 0 for each task as visible)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    config = utils_IMPACT.generate_hs_config(
        dataset_name=dataset_name,
        esc='error',
        valid_metric='rmse',
        pred_metrics=['rmse']
    )
    config['device'] = device

    utils_IMPACT.set_seed(config["seed"])
    train_data, valid_data, concept_map, metadata = load_dataset(config)

    # Shared SQLite storage accessible by all parallel tasks
    storage_name = f"sqlite:///{dataset_name}_IMPACT_rmse.db"
    study_name = f"hps_parallel_{dataset_name}_IMPACT"


    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        directions=["minimize"],
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
    dataset_name = "assist0910"
    nb_trials = 50
    main(dataset_name, nb_trials)
