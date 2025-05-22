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

import warnings
import os
import pickle
os.chdir("./experiments/notebook_example/")
print(os.getcwd())

liriscat.utils.setuplogger(verbose = True, log_name="liriscat")

import warnings
import numpy as np

gc.collect()
torch.cuda.empty_cache()


config = liriscat.utils.generate_eval_config(load_params=True, esc = 'error', valid_metric= 'mi_acc', pred_metrics = ["mi_acc"], profile_metrics = ['doa'], save_params=False, n_query=4, num_epochs=1, batch_size=512)
liriscat.utils.set_seed(config["seed"])

config["dataset_name"] = "math2"
logging.info(config["dataset_name"])
config['learning_rate'] = 0.02026
config['lambda'] = 1.2e-5
config['d_in'] = 4
config['num_responses'] = 12
#pred_metrics,df_interp = test(config)

logging.info(f'#### {config["dataset_name"]} ####')
logging.info(f'#### config : {config} ####')
config['embs_path']='../embs/'+str(config["dataset_name"])
config['params_path']='../ckpt/'+str(config["dataset_name"])

pred_metrics = {m:[] for m in config['pred_metrics']}
profile_metrics = {m:[] for m in config['profile_metrics']}

gc.collect()
torch.cuda.empty_cache()

# Dataset downloading for doa and rm
warnings.filterwarnings("ignore", message="invalid value encountered in divide")
warnings.filterwarnings("ignore", category=RuntimeWarning)

## Metadata map format : {"num_user_id": ..., "num_item_id": ..., "num_dimension_id": ...}
metadata = json.load(open(f'../datasets/2-preprocessed_data/{config["dataset_name"]}_metadata.json', 'r'))

## Tensor containing the nb of modalities per question
nb_modalities = torch.load(f'../datasets/2-preprocessed_data/{config["dataset_name"]}_nb_modalities.pkl',weights_only=True)

config['inner_user_lr'] = 0.01
config['num_inner_users_epochs'] = 8
for i_fold in range(1) : 
    ## Dataframe columns : (user_id, question_id, response, category_id)
    
    # Load the pickled datasets back into memory
    train_path = f'../datasets/2-preprocessed_data/{config["dataset_name"]}_dataset_train_{i_fold}.pkl'
    valid_path = f'../datasets/2-preprocessed_data/{config["dataset_name"]}_dataset_valid_{i_fold}.pkl'
    test_path  = f'../datasets/2-preprocessed_data/{config["dataset_name"]}_dataset_test_{i_fold}.pkl'

    with open(train_path, "rb") as f:
        train_data = pickle.load(f)
    with open(valid_path, "rb") as f:
        valid_data = pickle.load(f)
    with open(test_path, "rb") as f:
        test_data = pickle.load(f)

    train_data.load_config(config)
    valid_data.load_config(config)
    test_data.load_config(config)

    S = liriscat.selectionStrategy.Random(metadata,**config)
    #S.train(train_data, valid_data)
    S.init_models(train_data, valid_data)
    print(S.evaluate_test(test_data))