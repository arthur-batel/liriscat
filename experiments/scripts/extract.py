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
ALL_METRICS_SET = set(ACCURACY_METRICS) | set(INTERP_METRICS)

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
    s = re.sub(r'\bnan\b', 'None', s)
    return s

def literal_eval_dict(s: str) -> dict:
    s = normalize_numpy_scalars(s)
    try:
        obj = ast.literal_eval(s)
    except Exception as e:
        raise ValueError(f"Failed to literal_eval results dict: {e}\nSnippet: {s[:200]}...")
    if not isinstance(obj, dict):
        raise TypeError(f"Parsed object is not a dict: type={type(obj)}")
    return obj

def split_experiment_key(path: str):
    name = os.path.basename(path)
    m = re.match(r'^CAT_launch_(.+?)_(.+?)_(.+?)_(\d+)_([0-9]+)\.out$', name)
    if not m:
        return None
    dataset, cdm, meta_trainer, experiment_id, fold = m.groups()
    try:
        fold_i = int(fold)
    except Exception:
        raise ValueError(f"Fold is not an integer in filename: {name}")
    return (dataset, cdm, meta_trainer, experiment_id, fold_i)

def group_by_experiment(log_files: List[str]) -> Dict[Tuple[str, str, str, str], List[str]]:
    if not log_files:
        raise RuntimeError("No log files provided to group_by_experiment.")
    groups = collections.defaultdict(list)
    for fp in log_files:
        key = split_experiment_key(fp)
        if key is None:
            raise ValueError(f"Filename does not match expected pattern: {fp}")
        dataset, cdm, meta_trainer, experiment_id, _ = key
        groups[(dataset, cdm, meta_trainer, experiment_id)].append(fp)
    return groups

def parse_results_tuple_from_text(content: str):
    pat = r'[Rr]esults?:\s*\(\s*({.*?})\s*,\s*({.*?})\s*\)'
    m = re.search(pat, content, flags=re.DOTALL)
    if not m:
        preview = re.search(r'[Rr]esults?.{0,200}', content, re.DOTALL)
        prev_txt = preview.group(0) if preview else "NO 'results' PREVIEW FOUND"
        raise ValueError(f"'results: ({{...}}, {{...}})' tuple not found. Preview near 'results':\n{prev_txt}")
    acc = literal_eval_dict(m.group(1))
    itp = literal_eval_dict(m.group(2))
    return acc, itp

def parse_log_file(file_path: str):
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Log file not found: {file_path}")
    with open(file_path, 'r') as f:
        content = f.read()
    key = split_experiment_key(file_path)
    if key is None:
        raise ValueError(f"Filename does not match expected pattern: {file_path}")
    fold_number = key[-1]
    acc, itp = parse_results_tuple_from_text(content)
    return fold_number, acc, itp

def normalize_step_metrics_dict(d: dict, source_label: str) -> Dict[Any, Dict[str, float]]:
    if not isinstance(d, dict) or not d:
        raise TypeError(f"{source_label}: empty or non-dict structure.")
    # Case A: nested dict (step -> metrics)
    if all(isinstance(v, dict) for v in d.values()):
        out = {}
        for step, metrics in d.items():
            if not isinstance(metrics, dict):
                raise TypeError(f"{source_label}[{step}] should be dict, got {type(metrics)}")
            filt = {m: float(v) for m, v in metrics.items() if (m in ALL_METRICS_SET and is_finite_number(v))}
            out[step] = filt
        return out
    # Case B: flat dict (metric -> value)
    if all((k in ALL_METRICS_SET) and is_finite_number(v) for k, v in d.items()):
        return {0: {k: float(v) for k, v in d.items()}}
    # Optional wrapper: {"accuracy": {...}, "interpretability": {...}}
    if set(d.keys()) <= {"accuracy", "interpretability"} and all(isinstance(v, dict) for v in d.values()):
        merged = {}
        for sub in d.values():
            for m, v in sub.items():
                if m in ALL_METRICS_SET and is_finite_number(v):
                    merged[m] = float(v)
        if merged:
            return {0: merged}
    raise TypeError(f"{source_label}: unrecognized structure (neither 'step->metrics' nor flat 'metric->value').")

def aggregate_results(log_files: List[str]):
    if not log_files:
        raise RuntimeError("aggregate_results received empty file list.")
    all_accuracy_results: Dict[int, Dict[Any, Dict[str, float]]] = {}
    all_interpretability_results: Dict[int, Dict[Any, Dict[str, float]]] = {}
    for file_path in log_files:
        fold_number, acc_raw, itp_raw = parse_log_file(file_path)
        acc_norm = normalize_step_metrics_dict(acc_raw, f"accuracy_results in {file_path}")
        itp_norm = normalize_step_metrics_dict(itp_raw, f"interpretability_results in {file_path}")

        all_accuracy_results.setdefault(fold_number, {})
        all_interpretability_results.setdefault(fold_number, {})

        for step, metrics in acc_norm.items():
            all_accuracy_results[fold_number].setdefault(step, {})
            for metric, value in metrics.items():
                if metric in ACCURACY_METRICS:
                    all_accuracy_results[fold_number][step][metric] = value

        for step, metrics in itp_norm.items():
            all_interpretability_results[fold_number].setdefault(step, {})
            for metric, value in metrics.items():
                if metric in INTERP_METRICS:
                    all_interpretability_results[fold_number][step][metric] = value

    return all_accuracy_results, all_interpretability_results

def compute_mean_and_std(acc_results, itp_results):
    combined: Dict[int, Dict[Any, Dict[str, float]]] = {}
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
            all_metric_names |= set(step_data.keys())

    values_map: Dict[Tuple[Any, str], List[float]] = {}
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
            mtype = 'predictive' if metric in ACCURACY_METRICS else ('interpretability' if metric in INTERP_METRICS else 'unknown')
            summary.append({'step': step, 'metric': metric, 'mean': mean, 'std': std, 'n': n, 'type': mtype})
    return summary

def find_cdm_basis_files(search_dir: str, dataset: str, cdm: str) -> Dict[int, str]:
    if not os.path.isdir(search_dir):
        raise FileNotFoundError(f"Search directory not found: {search_dir}")
    paths: Dict[int, str] = {}
    missing = []
    ambiguous = []
    for fold in range(5):
        pattern = os.path.join(search_dir, f"CAT_launch_{dataset}_{cdm}_CDM_basis_*_{fold}.out")
        matches = sorted(glob.glob(pattern))
        if not matches:
            missing.append(fold)
        elif len(matches) > 1:
            ambiguous.append((fold, matches))
        else:
            paths[fold] = matches[0]
    if missing:
        raise RuntimeError(f"Missing CDM_basis files for dataset={dataset}, cdm={cdm}, folds={missing}. "
                           f"Expected 5 files named like 'CAT_launch_{dataset}_{cdm}_CDM_basis_*_{{fold}}.out'.")
    if ambiguous:
        details = "; ".join([f"fold={f}: {ms}" for f, ms in ambiguous])
        raise RuntimeError(f"Ambiguous CDM_basis matches for dataset={dataset}, cdm={cdm}: {details}")
    return paths

def extract_last_step_metrics_flexible(d: dict, source_label: str) -> Dict[str, float]:
    if not isinstance(d, dict) or not d:
        raise ValueError(f"{source_label}: empty or non-dict metrics structure.")
    if all(isinstance(v, dict) for v in d.values()):
        steps = sorted(d.keys(), key=try_int)
        last = steps[-1]
        block = d[last]
        if not isinstance(block, dict) or not block:
            raise ValueError(f"{source_label}: last-step block is empty or non-dict.")
        out = {}
        for m, v in block.items():
            if m in ALL_METRICS_SET and is_finite_number(v):
                out[m] = float(v)
        if not out:
            raise ValueError(f"{source_label}: no valid metrics in last-step block.")
        return out
    if all((k in ALL_METRICS_SET) and is_finite_number(v) for k, v in d.items()):
        return {k: float(v) for k, v in d.items()}
    raise TypeError(f"{source_label}: expected either 'step->metrics' dict or flat 'metric->value' dict; got mixed/invalid structure.")

def load_cdm_basis_metrics(search_dir: str, dataset: str, cdm: str) -> Tuple[Dict[int, Dict[str, float]], Dict[str, float]]:
    paths = find_cdm_basis_files(search_dir, dataset, cdm)
    per_fold: Dict[int, Dict[str, float]] = {}
    per_metric_values: Dict[str, List[float]] = collections.defaultdict(list)
    for fold, path in paths.items():
        fold_num, acc_dict, itp_dict = parse_log_file(path)
        if fold_num != fold:
            raise RuntimeError(f"Basis filename fold={fold} but parsed fold={fold_num} in file {path}")
        acc_last = extract_last_step_metrics_flexible(acc_dict, f"basis-acc[{path}]")
        itp_last = extract_last_step_metrics_flexible(itp_dict, f"basis-itp[{path}]")
        merged = {}
        merged.update(acc_last)
        merged.update(itp_last)
        if not merged:
            raise ValueError(f"No metrics extracted from basis file: {path}")
        per_fold[fold] = merged
        for m, v in merged.items():
            per_metric_values[m].append(v)
    basis_mean: Dict[str, float] = {}
    for metric, vals in per_metric_values.items():
        if len(vals) != 5:
            raise RuntimeError(f"CDM_basis metric '{metric}' not available for all 5 folds (got {len(vals)}).")
        basis_mean[metric] = float(np.mean(vals))
    return per_fold, basis_mean

def adjust_with_cdm_basis(acc, itp, basis_per_fold: Dict[int, Dict[str, float]], basis_mean: Dict[str, float]):
    exp_folds = set(acc.keys()) | set(itp.keys())
    missing_folds = [f for f in exp_folds if f not in basis_per_fold]
    if missing_folds:
        raise RuntimeError(f"Experiment contains folds without CDM_basis: {sorted(missing_folds)}")
    for fold, steps in acc.items():
        for step, metrics in steps.items():
            for metric, value in list(metrics.items()):
                if metric in ACCURACY_METRICS:
                    if metric not in basis_mean or metric not in basis_per_fold[fold]:
                        raise RuntimeError(f"Metric '{metric}' missing in CDM_basis for fold={fold}.")
                    metrics[metric] = float(value) + basis_mean[metric] - basis_per_fold[fold][metric]
    for fold, steps in itp.items():
        for step, metrics in steps.items():
            for metric, value in list(metrics.items()):
                if metric in INTERP_METRICS:
                    if metric not in basis_mean or metric not in basis_per_fold[fold]:
                        raise RuntimeError(f"Metric '{metric}' missing in CDM_basis for fold={fold}.")
                    metrics[metric] = float(value) + basis_mean[metric] - basis_per_fold[fold][metric]

def generate_outputs_for_group(dataset: str, cdm: str, meta_trainer: str,
                               experiment_id: str, files: List[str]):
    if not files:
        raise RuntimeError(f"No files for group {(dataset, cdm, meta_trainer, experiment_id)}")
    acc, itp = aggregate_results(files)
    if not acc and not itp:
        raise RuntimeError(f"No metrics parsed for group {(dataset, cdm, meta_trainer, experiment_id)}")
    search_dir = os.path.dirname(files[0]) if files else "."
    basis_per_fold, basis_mean = load_cdm_basis_metrics(search_dir, dataset, cdm)
    adjust_with_cdm_basis(acc, itp, basis_per_fold, basis_mean)
    base = f"CAT_launch_{dataset}_{cdm}_{meta_trainer}_{experiment_id}"
    os.makedirs("logs", exist_ok=True)
    summary_data = compute_mean_and_std(acc, itp)
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
        all_results[fold_num] = {0: acc.get(fold_num, {}), 1: itp.get(fold_num, {})}
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
        files_sorted = sorted(files, key=split_experiment_key)
        generate_outputs_for_group(dataset, cdm, meta_trainer, experiment_id, files_sorted)
