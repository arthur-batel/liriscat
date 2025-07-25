#!/usr/bin/env python3
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["PYTHONHASHSEED"] = "0"
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"

import liriscat
liriscat.utils.set_seed(0)
from liriscat.dataset import preprocessing_utilities as pu

import logging
import gc
import json
import torch
import pandas as pd
import argparse
import warnings
import time  # ← Ajouté pour mesurer le temps

print("PID", os.getpid(), "sees CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))
if os.environ.get("CUDA_VISIBLE_DEVICES") is not None:
    print("→ torch.device is", torch.cuda.current_device(), torch.cuda.get_device_name(0))

def pareto_index(d):
    d_acc = d[0]
    d_meta = d[1]
    r = []
    for i in range(len(d_acc)):
        r.append(d_acc[i]['mi_acc'] * d_meta[i]['meta_doa'])
    return sum(r)

def main(dataset_name, i_fold=None):

    gc.collect()
    torch.cuda.empty_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    config = liriscat.utils.generate_eval_config(
        dataset_name=dataset_name,
        i_fold = i_fold,
        load_params=True,
        save_params=False,
        esc = 'error',
        learning_rate = 0.001,
        batch_size = 512,
        valid_batch_size = 10000,
        num_epochs=100,
        eval_freq = 1,
        patience = 20,
        device = device,
        pred_metrics = ["mi_acc","rmse","mae","mi_prec","mi_rec","mi_f_b","mi_auc","ma_prec","ma_rec","ma_f_b"],
        profile_metrics = ['meta_doa','pc-er','rm'],
        CDM = 'impact',
        meta_trainer='Approx_GAP',
        valid_metric= 'mi_acc',
        n_query=16,
        num_inner_users_epochs=3,
        meta_lr=0.01,
        lambda_=2.2656270501845414e-06,
        inner_user_lr=0.0016969685554352153
    )
    logging.info(f'#### config : {config} ####')

    concept_map = json.load(open(
        f'../datasets/2-preprocessed_data/{config["dataset_name"]}_concept_map.json', 'r'
    ))
    concept_map = {int(k): [int(x) for x in v] for k, v in concept_map.items()}

    metadata = json.load(open(
        f'../datasets/2-preprocessed_data/{config["dataset_name"]}_metadata.json', 'r'
    ))

    nb_modalities = torch.load(
        f'../datasets/2-preprocessed_data/{config["dataset_name"]}_nb_modalities.pkl',
        weights_only=True
    )

    train_df = pd.read_csv(
        f'../datasets/2-preprocessed_data/{config["dataset_name"]}_train_{i_fold}.csv',
        encoding='utf-8',
        dtype={'student_id': int, 'item_id': int, "correct": float, "dimension_id": int}
    )
    valid_df = pd.read_csv(
        f'../datasets/2-preprocessed_data/{config["dataset_name"]}_valid_{i_fold}.csv',
        encoding='utf-8',
        dtype={'student_id': int, 'item_id': int, "correct": float, "dimension_id": int}
    )
    test_df = pd.read_csv(
        f'../datasets/2-preprocessed_data/{config["dataset_name"]}_test_{i_fold}.csv',
        encoding='utf-8',
        dtype={'student_id': int, 'item_id': int, "correct": float, "dimension_id": int}
    )

    train_data = liriscat.dataset.CATDataset(train_df, concept_map, metadata, config, nb_modalities)
    valid_data = liriscat.dataset.EvalDataset(valid_df, concept_map, metadata, config, nb_modalities)
    test_data  = liriscat.dataset.EvalDataset(test_df,  concept_map, metadata, config, nb_modalities)

    S = liriscat.selectionStrategy.Random(train_data.metadata, **config)
    S.init_models(train_data, valid_data)

    # — Mesure du temps d'entraînement
    logging.info("⏳ Début de l'entraînement...")
    t_start = time.time()
    S.train(train_data, valid_data)
    t_end = time.time()
    logging.info(f"✅ Entraînement terminé en {t_end - t_start:.2f} secondes.")

    S.reset_rng()
    d = S.evaluate_test(test_data, train_data, valid_data)
    pi = pareto_index(d)
    logging.info(f"Fold {config['i_fold']}; pareto_index: {pi}; results: {d}")

    del S
    gc.collect()
    torch.cuda.empty_cache()

def parse_args():
    parser = argparse.ArgumentParser(
        description="A program that runs the CAT testing session"
    )
    parser.add_argument('dataset_name', help="the dataset name")
    parser.add_argument(
        '--i_fold',
        type=int,
        default=None,
        help="0-indexed fold number (if omitted runs all folds)"
    )
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    liriscat.utils.setuplogger(verbose=True, log_name="CAT_test", debug=False)
    main(args.dataset_name, args.i_fold)
