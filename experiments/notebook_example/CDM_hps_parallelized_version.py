import importlib
import IMPACT.utils as utils_IMPACT
import IMPACT.model as model_IMPACT
import IMPACT.dataset as dataset_IMPACT

from liriscat import utils as utils_liriscat, dataset as dataset_liriscat
from functools import partial

from ipyparallel import Client
import dill
import json
import logging
import gc
import torch
from liriscat.dataset.preprocessing_utilities import *
cat_absolute_path = os.path.abspath('../../')

rc = Client()
rc[:].use_dill()
lview = rc.load_balanced_view()


rc[:].execute("import sys; sys.path.append('"+cat_absolute_path+"')")
logging.info("sys.path.append("+cat_absolute_path+")")
with rc[:].sync_imports():
    import json
    from IMPACT import utils as utils_IMPACT, model as model_IMPACT, dataset as dataset_IMPACT
    import logging
    import gc
    import torch

def main(dataset_name, nb_trials, nb_jobs):
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
    objective_with_args = partial(utils_liriscat.objective_hps, config=config, train_data=train_data,
                                  valid_data=valid_data)
    study.optimize(objective_with_args, n_trials=nb_trials, n_jobs=nb_jobs, gc_after_trial=True)

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