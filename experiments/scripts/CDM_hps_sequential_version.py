from numpy.lib.function_base import interp
import sys
sys.path.append("../../")

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from IMPACT import utils as utils_IMPACT
utils_IMPACT.set_seed(0)
from IMPACT import dataset
from IMPACT import model
import optuna
import logging
import gc
import json
import torch
import pandas as pd
from importlib import reload
from micat.dataset.preprocessing_utilities import *
from micat import utils as utils_micat
from functools import partial

def main(dataset_name, nb_trials, nb_jobs):
    utils_IMPACT.setuplogger(verbose=True, log_name=f"micat_{dataset_name}")
    config = utils_IMPACT.generate_hs_config(dataset_name=dataset_name, esc='error', valid_metric='rmse',
                                             pred_metrics=['rmse'])

    utils_IMPACT.set_seed(config["seed"])
    logging.info(config['dataset_name'])
    train_data, valid_data, concept_map, metadata = load_dataset(config)

    study = optuna.create_study(
        directions=["minimize"],  # Warning : specify directions for each objective (depends on the validation metric)
    )
    gc.collect()
    torch.cuda.empty_cache()
    objective_with_args = partial(utils_micat.IMPACT_pre_train, config=config, train_data=train_data,
                                  valid_data=valid_data)
    study.optimize(objective_with_args, n_trials=1, n_jobs=1, gc_after_trial=True)

    # Analyze the results
    ## requirements : plotly, nbformat
    pareto_trials = study.best_trials

    logging.info(f"Best trial for {config['dataset_name']} : {study.best_trials}")
    for trial in pareto_trials:
        logging.info(f"Trial #{trial.number}")
        logging.info(f"  Metric value: {trial.values}")
        # logging.info(f"  DOA: {trial.values[1]}")
        logging.info(f"  Params: {trial.params}")

if __name__ == '__main__':
    dataset_name = sys.argv[1]
    nb_trials = int(sys.argv[2])
    nb_jobs = int(sys.argv[3])
    main(dataset_name, nb_trials, nb_jobs)

