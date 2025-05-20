import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["PYTHONHASHSEED"] = "0"
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"

import liriscat
liriscat.utils.set_seed(0)

import logging
import json
import torch
liriscat.utils.set_seed(0)
import pandas as pd

import warnings
import os
import pickle
os.chdir("./experiments/notebook_example/")
print(os.getcwd())

liriscat.utils.setuplogger(verbose = True, log_name="liriscat")

import warnings

config = liriscat.utils.generate_eval_config(load_params=True, esc = 'error', valid_metric= 'mi_acc', pred_metrics = ["mi_acc"], profile_metrics = ['doa'], save_params=False, n_query=4, num_epochs=1, batch_size=512)
liriscat.utils.set_seed(config["seed"])

config["dataset_name"] = "math2"
logging.info(config["dataset_name"])

#pred_metrics,df_interp = test(config)

logging.info(f'#### {config["dataset_name"]} ####')
logging.info(f'#### config : {config} ####')
config['embs_path']='../embs/'+str(config["dataset_name"])
config['params_path']='../ckpt/'+str(config["dataset_name"])

# Dataset downloading for doa and rm
warnings.filterwarnings("ignore", message="invalid value encountered in divide")
warnings.filterwarnings("ignore", category=RuntimeWarning)

## Concept map format : {question_id : [category_id1, category_id2, ...]}
concept_map = json.load(open(f'../datasets/2-preprocessed_data/{config["dataset_name"]}_concept_map.json', 'r'))
concept_map = {int(k): [int(x) for x in v] for k, v in concept_map.items()}

## Metadata map format : {"num_user_id": ..., "num_item_id": ..., "num_dimension_id": ...}
metadata = json.load(open(f'../datasets/2-preprocessed_data/{config["dataset_name"]}_metadata.json', 'r'))


## Tensor containing the nb of modalities per question
nb_modalities = torch.load(f'../datasets/2-preprocessed_data/{config["dataset_name"]}_nb_modalities.pkl',weights_only=True)

for i_fold in range(5) : 
    logger.info(f'i_fold : {i_fold} ')
    ## Dataframe columns : (user_id, question_id, response, category_id)
    train_df = pd.read_csv(
        f'../datasets/2-preprocessed_data/{config["dataset_name"]}_train_{i_fold}.csv',
        encoding='utf-8', dtype={'student_id': int, 'item_id': int, "correct": float,
                                                                 "dimension_id": int})
    valid_df = pd.read_csv(
        f'../datasets/2-preprocessed_data/{config["dataset_name"]}_valid_{i_fold}.csv',
        encoding='utf-8', dtype={'student_id': int, 'item_id': int, "correct": float,
                                                                 "dimension_id": int})
    test_df = pd.read_csv(
        f'../datasets/2-preprocessed_data/{config["dataset_name"]}_test_{i_fold}.csv',
        encoding='utf-8', dtype={'student_id': int, 'item_id': int, "correct": float,
                                                                 "dimension_id": int})

    train_data = liriscat.dataset.CATDataset(train_df, concept_map, metadata, config, nb_modalities)
    valid_data = liriscat.dataset.EvalDataset(valid_df, concept_map, metadata, config, nb_modalities)
    test_data  = liriscat.dataset.EvalDataset(test_df,  concept_map, metadata, config, nb_modalities)

    # pickle each split
    for name, data in (("train", train_data), ("valid", valid_data), ("test", test_data)):
        path = f'../datasets/2-preprocessed_data/{config["dataset_name"]}_dataset_{name}_{i_fold}.pkl'
        with open(path, "wb") as f:
            pickle.dump(data, f)