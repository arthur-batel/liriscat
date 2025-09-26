#!/usr/bin/env python3
import os
import re
import ast
import csv
import glob
import json
import math
import collections
from typing import Dict, Tuple, List, Any
import numpy as np

ACCURACY_METRICS = [
    "mi_acc", "rmse", "mae", "mi_prec", "mi_rec", "mi_f_b",
    "mi_auc", "ma_prec", "ma_rec", "ma_f_b"
]
INTERP_METRICS = ["meta_doa", "pc-er", "rm"]

def try_int(x):
    try:
        return int(x)
    except Exception:
        return x

def is_finite_number(x) -> bool:
    try:
        v = float(x)
        return math.isfinite(v)
    except Exception:
        return False

def normalize_numpy_scalars(s: str) -> str:
    s = re.sub(r'np\.float64\(\s*([^)]+)\s*\)', r'\1', s)
    s = re.sub(r'np\.int64\(\s*([^)]+)\s*\)', r'\1', s)
    return s

def literal_eval_dict(s: str) -> dict:
    s = normalize_numpy_scalars(s)
    s = re.sub(r'\bnan\b', 'None', s)
    return ast.literal_eval(s)

def split_experiment_key(path: str):
    name = os.path.basename(path)
    m = re.match(r'^CAT_launch_(.+?)_(.+?)_(.+?)_(\d+)_([0-9]+)\.out$', name)
    if not m:
        return None
    dataset, cdm, meta_trainer, experiment_id, fold = m.groups()
    return (dataset, cdm, meta_trainer, experiment_id, int(fold))

def group_by_experiment(log_files: List[str]) -> Dict[Tuple[str, str, str, str], List[str]]:
    groups = collections.defaultdict(list)
    for fp in log_files:
        key = split_experiment_key(fp)
        if key is None:
            continue
        dataset, cdm, meta_trainer, experiment_id, _ = key
        groups[(dataset, cdm, meta_trainer, experiment_id)].append(fp)
    return groups

def parse_log_file(file_path: str):
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        key = split_experiment_key(file_path)
        fold_number = key[-1] if key is not None else None
        pat = r'[Rr]esults?:\s*\(\s*({.*?})\s*,\s*({.*?})\s*\)'
        m = re.search(pat, content, flags=re.DOTALL)
        if not m:
            return None, None, None
        acc = literal_eval_dict(m.group(1))
        itp = literal_eval_dict(m.group(2))
        return fold_number, acc, itp
    except Exception:
        return None, None, None

def aggregate_results(log_files: List[str]):
    all_accuracy_results = {}
    all_interpretability_results = {}
    for file_path in log_files:
        fold_number, accuracy_results, interpretability_results = parse_log_file(file_path)
        if fold_number is None:
            continue
        all_accuracy_results.setdefault(fold_number, {})
        all_interpretability_results.setdefault(fold_number, {})
        if isinstance(accuracy_results, dict):
            for step, metrics in accuracy_results.items():
                all_accuracy_results[fold_number].setdefault(step, {})
                if isinstance(metrics, dict):
                    for metric, value in metrics.items():
                        if metric in ACCURACY_METRICS:
                            all_accuracy_results[fold_number][step][metric] = value
        if isinstance(interpretability_results, dict):
            for step, metrics in interpretability_results.items():
                all_interpretability_results[fold_number].setdefault(step, {})
                if isinstance(metrics, dict):
                    for metric, value in metrics.items():
                        if metric in INTERP_METRICS:
                            all_interpretability_results[fold_number][step][metric] = value
    return all_accuracy_results, all_interpretability_results

def compute_mean_and_std(acc_results, itp_results):
    combined = {}
    for fold, data in acc_results.items():
        combined[fold] = {k: dict(v) for k, v in data.items()}
    for fold, data in itp_results.items():
        combined.setdefault(fold, {})
        for step, metrics in data.items():
            combined[fold].setdefault(step, {})
            combined[fold][step].update(metrics)
    all_steps = set()
    all_metric_names = set()
    for fold_data in combined.values():
        for step in fold_data.keys():
            all_steps.add(step)
        for step_data in fold_data.values():
            for metric in step_data.keys():
                all_metric_names.add(metric)
    values_map = {}
    for step in all_steps:
        for metric in all_metric_names:
            values_map[(step, metric)] = []
    for fold_data in combined.values():
        for step, step_data in fold_data.items():
            for metric, value in step_data.items():
                values_map[(step, metric)].append(value)
    summary = []
    for step in sorted(all_steps, key=try_int):
        for metric in sorted(all_metric_names):
            vals = values_map.get((step, metric), [])
            clean = [float(v) for v in vals if is_finite_number(v)]
            if not clean:
                continue
            mean = float(np.mean(clean))
            std = float(np.std(clean))
            n = len(clean)
            mtype = 'predictive' if metric in ACCURACY_METRICS else 'interpretability'
            summary.append({
                'step': step,
                'metric': metric,
                'mean': mean,
                'std': std,
                'n': n,
                'type': mtype
            })
    return summary

def generate_outputs_for_group(dataset: str, cdm: str, meta_trainer: str,
                               experiment_id: str, files: List[str]):
    acc, itp = aggregate_results(files)
    summary_data = compute_mean_and_std(acc, itp)
    base = f"CAT_launch_{dataset}_{cdm}_{meta_trainer}_{experiment_id}"
    csv_filename = f"logs/{base}_metrics_summary.csv"
    with open(csv_filename, 'w', newline='') as csvfile:
        fieldnames = ['step', 'metric', 'mean', 'std', 'n', 'type']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_data:
            writer.writerow(row)
    print(f"Successfully wrote CSV file: {csv_filename}")
    json_summary_filename = f"logs/{base}_metrics_summary.json"
    with open(json_summary_filename, 'w') as jf:
        summary_dict = {idx: row for idx, row in enumerate(summary_data)}
        json.dump(summary_dict, jf, indent=4)
    print(f"Successfully wrote JSON file: {json_summary_filename}")
    all_results = {}
    fold_ids = sorted(set(acc.keys()) | set(itp.keys()))
    for fold_num in fold_ids:
        all_results[fold_num] = {
            0: acc.get(fold_num, {}),
            1: itp.get(fold_num, {})
        }
    all_results_filename = f"logs/{base}_all_results.json"
    with open(all_results_filename, 'w') as jf:
        json.dump(all_results, jf, indent=4)
    print(f"Successfully wrote JSON file: {all_results_filename}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pattern", default="./logs/CAT_launch_*_*_*_*_*.out")
    args = parser.parse_args()
    log_files = glob.glob(args.pattern)
    if not log_files:
        raise SystemExit(0)
    groups = group_by_experiment(log_files)
    if not groups:
        raise SystemExit(0)
    for (dataset, cdm, meta_trainer, experiment_id), files in sorted(groups.items()):
        generate_outputs_for_group(dataset, cdm, meta_trainer, experiment_id,
                                   sorted(files, key=split_experiment_key))
