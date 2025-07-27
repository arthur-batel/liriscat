import sys
import os
import optuna
import gc
import torch
import logging
from functools import partial
from IMPACT import utils as utils_IMPACT
from IMPACT.model import IMPACT
from IMPACT.dataset import LoaderDataset as IMPACT_dataset
from liriscat.dataset.preprocessing_utilities import *
from optuna.exceptions import DuplicatedStudyError
from liriscat.utils import convert_config_to_EduCAT
from liriscat.dataset import preprocessing_utilities as pu
from liriscat.CDM.NCDM import NCDM

import os, torch
print("PID", os.getpid(), "sees CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("→ torch.device is", torch.cuda.current_device(), torch.cuda.get_device_name(0))

def launch_test(trial, train_data, valid_data, config,  metadata):
    gc.collect()
    torch.cuda.empty_cache()

    cdm = NCDM(metadata['num_dimension_id'], metadata['num_item_id'], metadata['num_user_id'], config)
    cdm.train(train_data, valid_data, epoch=config['num_epochs'], device="cuda")

    best_valid_metric = cdm.best_valid_rmse
    logging.info(f"Trial {trial.number}: Best metric: {best_valid_metric}")

    del cdm
    gc.collect()
    torch.cuda.empty_cache()

    return best_valid_metric

def objective_hps(trial, config, train_data, valid_data, metadata):
    lr = trial.suggest_float('learning_rate', 1e-5, 5e-2, log=True)

    config['learning_rate'] = lr

    return launch_test(trial, train_data, valid_data, config, metadata)

def main(dataset_name, nb_trials):
    # SLURM auto-assigns a GPU (index 0 for each task as visible)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    config = utils_IMPACT.generate_hs_config(
        dataset_name=dataset_name,
        esc='error',
        valid_metric='mi_acc',
        pred_metrics=['mi_acc']
    )
    config['device'] = device

    utils_IMPACT.set_seed(config["seed"])

    concept_map, metadata, nb_modalities = pu.load_dataset_resources(config)

    config = convert_config_to_EduCAT(config, metadata)

    vertical_train, vertical_valid = pu.vertical_data(config, config['i_fold'])

    impact_train_data = IMPACT_dataset(vertical_train, concept_map, metadata, nb_modalities)
    impact_valid_data = IMPACT_dataset(vertical_valid, concept_map, metadata, nb_modalities)

    train_data, valid_data = [
        pu.transform(data.raw_data_array[:,0].long(), data.raw_data_array[:,1].long(), concept_map, data.raw_data_array[:,2], config['batch_size'], impact_train_data.n_categories)
        for data in [impact_train_data, impact_valid_data]
    ]

    # Shared SQLite storage accessible by all parallel tasks
    storage_name = f"sqlite:///{dataset_name}_NCDM.db"
    study_name = f"hps_parallel_{dataset_name}_NCDM"


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
        metadata=metadata
    )

    study.optimize(objective, n_trials=nb_trials, n_jobs=1, gc_after_trial=True)

    # After optimization, log best results (finalized in any task)
    if study.best_trials:
        print("Best trial(s):")
        for trial in study.best_trials:
            print(f"Trial #{trial.number}: {trial.values}, Params: {trial.params}")

if __name__ == '__main__':
    dataset_name = "algebra"
    nb_trials = 50
    main(dataset_name, nb_trials)
