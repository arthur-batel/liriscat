#!/usr/bin/env python3
import os
import re
import ast
import csv
import glob
import json
import math
import traceback
import collections
from typing import Dict, Tuple, List, Any
import numpy as np

ACCURACY_METRICS = [
    "mi_acc", "rmse", "mae", "mi_prec", "mi_rec", "mi_f_b",
    "mi_auc", "ma_prec", "ma_rec", "ma_f_b"
]
INTERP_METRICS = ["meta_doa", "pc-er", "rm"]
ALL_METRICS_SET = set(ACCURACY_METRICS) | set(INTERP_METRICS)

# ---------------- Utils

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
    # Convert np.float64(x) -> x; np.int64(x) -> x ; and nan -> None
    s = re.sub(r'np\.float64\(\s*([^)]+)\s*\)', r'\1', s)
    s = re.sub(r'np\.int64\(\s*([^)]+)\s*\)', r'\1', s)
    s = re.sub(r'\bnan\b', 'None', s)
    return s

def literal_eval_dict(s: str) -> dict:
    s = normalize_numpy_scalars(s)
    try:
        obj = ast.literal_eval(s)
    except Exception as e:
        snippet = s[:400].replace("\n", " ")
        raise ValueError(f"literal_eval failed. Cause={e}; snippet='{snippet} ...'") from e
    if not isinstance(obj, dict):
        raise TypeError(f"Expected dict after literal_eval, got {type(obj)}")
    return obj

def split_experiment_key(path: str):
    name = os.path.basename(path)
    m = re.match(r'^CAT_launch_(.+?)_(.+?)_(.+?)_(\d+)_([0-9]+)\.out$', name)
    if not m:
        return None
    dataset, cdm, meta_trainer, experiment_id, fold = m.groups()
    try:
        fold_i = int(fold)
    except Exception as e:
        raise ValueError(f"Fold is not an integer in filename: {name}") from e
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

def parse_results_tuple_from_text(content: str, file_path: str):
    # tolerate any prefix like "Fold 4; results: (...)"
    pat = r'[Rr]esults?:\s*\(\s*({.*?})\s*,\s*({.*?})\s*\)'
    m = re.search(pat, content, flags=re.DOTALL)
    if not m:
        preview = re.search(r'[Rr]esults?.{0,200}', content, re.DOTALL)
        prev_txt = preview.group(0) if preview else "NO 'results' PREVIEW FOUND"
        raise ValueError(f"Missing results tuple in file: {file_path}. Preview: {prev_txt}")
    try:
        acc = literal_eval_dict(m.group(1))
    except Exception as e:
        raise ValueError(f"Failed parsing accuracy dict in {file_path}: {e}") from e
    try:
        itp = literal_eval_dict(m.group(2))
    except Exception as e:
        raise ValueError(f"Failed parsing interpretability dict in {file_path}: {e}") from e
    return acc, itp

def parse_log_file(file_path: str):
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Log file not found: {file_path}")
    try:
        with open(file_path, 'r') as f:
            content = f.read()
    except Exception as e:
        raise IOError(f"Failed to read file: {file_path}. Cause={e}") from e
    key = split_experiment_key(file_path)
    if key is None:
        raise ValueError(f"Filename does not match expected pattern: {file_path}")
    fold_number = key[-1]
    acc, itp = parse_results_tuple_from_text(content, file_path)
    return fold_number, acc, itp

# ---------------- Normalization of shapes (accept non-finite values gracefully)

def _filter_metric_block(block: dict) -> dict:
    """Keep only known metrics; convert finite values to float; drop non-finite values."""
    out = {}
    for m, v in block.items():
        if m in ALL_METRICS_SET and is_finite_number(v):
            out[m] = float(v)
    return out

def normalize_step_metrics_dict(d: dict, source_label: str) -> Dict[Any, Dict[str, float]]:
    """
    Accepts:
      - nested dict: {step: {metric: value, ...}, ...}
      - flat dict:   {metric: value, ...}
      - wrapper:     {"accuracy": {...}, "interpretability": {...}}
    Returns: {step: {metric: float_value, ...}}, with non-finite values dropped.
    """
    if not isinstance(d, dict) or not d:
        raise TypeError(f"{source_label}: empty or non-dict structure.")

    # Case A: nested dict (step -> metrics)
    if all(isinstance(v, dict) for v in d.values()):
        out = {}
        for step, metrics in d.items():
            if not isinstance(metrics, dict):
                raise TypeError(f"{source_label}[{step}] should be dict, got {type(metrics)}")
            filt = _filter_metric_block(metrics)
            if filt:  # only store if something remains
                out[step] = filt
        if out:
            return out
        # otherwise fall through to try wrappers/flat
    # Case B: flat dict (metric -> value); allow some non-finite values present
    if any(k in ALL_METRICS_SET for k in d.keys()):
        filt = _filter_metric_block(d)
        if filt:
            return {0: filt}
        else:
            # no finite metrics — return empty to be ignored upstream
            return {}

    # Case C: optional wrapper: {"accuracy": {...}, "interpretability": {...}}
    if set(d.keys()) <= {"accuracy", "interpretability"} and all(isinstance(v, dict) for v in d.values()):
        merged = {}
        for sub in d.values():
            merged.update(_filter_metric_block(sub))
        if merged:
            return {0: merged}
        else:
            return {}

    raise TypeError(f"{source_label}: unrecognized structure (neither 'step->metrics' nor flat 'metric->value').")

# ---------------- Aggregate experiments

def aggregate_results(log_files: List[str], debug: bool = False):
    if not log_files:
        raise RuntimeError("aggregate_results received empty file list.")
    all_accuracy_results: Dict[int, Dict[Any, Dict[str, float]]] = {}
    all_interpretability_results: Dict[int, Dict[Any, Dict[str, float]]] = {}

    for file_path in log_files:
        try:
            fold_number, acc_raw, itp_raw = parse_log_file(file_path)
        except Exception as e:
            if debug: traceback.print_exc()
            print(f"[ERROR] parse_log_file failed for '{file_path}': {e}")
            continue

        try:
            acc_norm = normalize_step_metrics_dict(acc_raw, f"accuracy_results in {file_path}")
        except Exception as e:
            if debug: traceback.print_exc()
            print(f"[ERROR] normalize accuracy failed for '{file_path}': {e}")
            acc_norm = {}
        try:
            itp_norm = normalize_step_metrics_dict(itp_raw, f"interpretability_results in {file_path}")
        except Exception as e:
            if debug: traceback.print_exc()
            print(f"[ERROR] normalize interpretability failed for '{file_path}': {e}")
            itp_norm = {}

        if not acc_norm and not itp_norm:
            print(f"[WARN] No usable metrics after filtering in '{file_path}'")
            continue

        all_accuracy_results.setdefault(fold_number, {})
        all_interpretability_results.setdefault(fold_number, {})

        try:
            for step, metrics in acc_norm.items():
                if not metrics:  # skip empty
                    continue
                all_accuracy_results[fold_number].setdefault(step, {})
                for metric, value in metrics.items():
                    if metric in ACCURACY_METRICS:
                        all_accuracy_results[fold_number][step][metric] = value
        except Exception as e:
            if debug: traceback.print_exc()
            print(f"[ERROR] Storing accuracy metrics failed for '{file_path}', fold={fold_number}: {e}")

        try:
            for step, metrics in itp_norm.items():
                if not metrics:
                    continue
                all_interpretability_results[fold_number].setdefault(step, {})
                for metric, value in metrics.items():
                    if metric in INTERP_METRICS:
                        all_interpretability_results[fold_number][step][metric] = value
        except Exception as e:
            if debug: traceback.print_exc()
            print(f"[ERROR] Storing interpretability metrics failed for '{file_path}', fold={fold_number}: {e}")

    return all_accuracy_results, all_interpretability_results

# ---------------- Summary

def compute_mean_and_std(acc_results, itp_results, debug: bool = False):
    try:
        combined: Dict[int, Dict[Any, Dict[str, float]]] = {}
        for fold, data in acc_results.items():
            combined[fold] = {k: dict(v) for k, v in data.items()}
        for fold, data in itp_results.items():
            combined.setdefault(fold, {})
            for step, metrics in data.items():
                combined[fold].setdefault(step, {})
                combined[fold][step].update(metrics)
    except Exception as e:
        if debug: traceback.print_exc()
        raise RuntimeError(f"Failed to combine results for mean/std: {e}") from e

    all_steps = set()
    all_metric_names = set()
    for fold, fold_data in combined.items():
        for step in fold_data.keys():
            all_steps.add(step)
        for step_data in fold_data.values():
            all_metric_names |= set(step_data.keys())

    values_map: Dict[Tuple[Any, str], List[float]] = {}
    for step in all_steps:
        for metric in all_metric_names:
            values_map[(step, metric)] = []

    for f, fold_data in combined.items():
        for step, step_data in fold_data.items():
            for metric, value in step_data.items():
                values_map[(step, metric)].append(value)

    summary = []
    try:
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
    except Exception as e:
        if debug: traceback.print_exc()
        raise RuntimeError(f"Failed computing mean/std for step/metric ('{step}', '{metric}'): {e}") from e

    return summary

# ---------------- CDM-basis loading (more tolerant)

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
                           f"Expected 5 files like 'CAT_launch_{dataset}_{cdm}_CDM_basis_*_{{fold}}.out'.")
    if ambiguous:
        details = "; ".join([f"fold={f}: {ms}" for f, ms in ambiguous])
        raise RuntimeError(f"Ambiguous CDM_basis matches for dataset={dataset}, cdm={cdm}: {details}")
    return paths

def extract_last_step_metrics_flexible(d: dict, source_label: str) -> Dict[str, float]:
    """
    Returns a flat {metric: float} map.
    Accepts nested (step->metrics) or flat dict. Drops non-finite values.
    """
    if not isinstance(d, dict) or not d:
        raise ValueError(f"{source_label}: empty or non-dict metrics structure.")

    # Nested: step -> metrics dict
    if all(isinstance(v, dict) for v in d.values()):
        steps = sorted(d.keys(), key=try_int)
        last = steps[-1]
        block = d[last]
        if not isinstance(block, dict) or not block:
            raise ValueError(f"{source_label}: last-step block is empty or non-dict.")
        out = _filter_metric_block(block)  # drops non-finite
        if not out:
            # No finite metrics in last step: return empty (caller can skip)
            return {}
        return out

    # Flat dict: metric -> value (allow some non-finite)
    if any(k in ALL_METRICS_SET for k in d.keys()):
        out = _filter_metric_block(d)  # drops non-finite
        # May be empty if all were non-finite; return empty to let caller decide
        return out

    # Wrapper
    if set(d.keys()) <= {"accuracy", "interpretability"} and all(isinstance(v, dict) for v in d.values()):
        merged = {}
        for sub in d.values():
            merged.update(_filter_metric_block(sub))
        return merged

    raise TypeError(f"{source_label}: expected 'step->metrics' or flat 'metric->value'; got mixed/invalid.")

def load_cdm_basis_metrics(search_dir: str, dataset: str, cdm: str, debug: bool = False) -> Tuple[Dict[int, Dict[str, float]], Dict[str, float]]:
    paths = find_cdm_basis_files(search_dir, dataset, cdm)
    per_fold: Dict[int, Dict[str, float]] = {}
    per_metric_values: Dict[str, List[float]] = collections.defaultdict(list)

    for fold, path in paths.items():
        try:
            fold_num, acc_dict, itp_dict = parse_log_file(path)
        except Exception as e:
            if debug: traceback.print_exc()
            raise RuntimeError(f"Parsing CDM_basis file failed for fold={fold}, path={path}: {e}") from e
        if fold_num != fold:
            raise RuntimeError(f"CDM_basis filename fold={fold} but parsed fold={fold_num} in file {path}")

        try:
            acc_last = extract_last_step_metrics_flexible(acc_dict, f"basis-acc[{path}]")
            itp_last = extract_last_step_metrics_flexible(itp_dict, f"basis-itp[{path}]")
        except Exception as e:
            if debug: traceback.print_exc()
            raise RuntimeError(f"Extracting last-step metrics failed for basis path={path}: {e}") from e

        merged = {}
        if acc_last: merged.update(acc_last)
        if itp_last: merged.update(itp_last)

        if not merged:
            # If a fold has no finite metrics at all, keep it empty and continue.
            print(f"[WARN] No finite basis metrics in {path}; fold {fold} will not normalize any metric from this file.")
            per_fold[fold] = {}
            continue

        per_fold[fold] = merged
        for m, v in merged.items():
            per_metric_values[m].append(v)

    # Compute per-metric mean only from folds where metric was present & finite
    basis_mean: Dict[str, float] = {}
    for metric, vals in per_metric_values.items():
        if not vals:
            continue
        basis_mean[metric] = float(np.mean(vals))

    return per_fold, basis_mean

# ---------------- Adjust with CDM-basis (skip missing metrics gracefully)

def adjust_with_cdm_basis(acc, itp, basis_per_fold: Dict[int, Dict[str, float]], basis_mean: Dict[str, float], debug: bool = False):
    exp_folds = set(acc.keys()) | set(itp.keys())
    missing_folds = [f for f in exp_folds if f not in basis_per_fold]
    if missing_folds:
        raise RuntimeError(f"Experiment contains folds without CDM_basis: {sorted(missing_folds)}")

    # Helper to adjust one dict group
    def _adjust_block(block: Dict[int, Dict[Any, Dict[str, float]]], which: str):
        for fold, steps in block.items():
            for step, metrics in steps.items():
                for metric, value in list(metrics.items()):
                    if metric not in basis_mean or metric not in basis_per_fold.get(fold, {}):
                        # No usable basis for this metric/fold => skip normalization
                        print(f"[WARN] Skipping normalization for metric '{metric}' (fold={fold}, step={step}, group={which}): no finite CDM_basis reference.")
                        continue
                    try:
                        metrics[metric] = float(value) + basis_mean[metric] - basis_per_fold[fold][metric]
                    except Exception as e:
                        if debug: traceback.print_exc()
                        print(f"[ERROR] Adjust error ({which}): fold={fold}, step={step}, metric={metric}, value={value}: {e}")

    _adjust_block(acc, "accuracy")
    _adjust_block(itp, "interpretability")

# ---------------- Output

def compute_mean_and_std(acc_results, itp_results, debug: bool = False):
    try:
        combined: Dict[int, Dict[Any, Dict[str, float]]] = {}
        for fold, data in acc_results.items():
            combined[fold] = {k: dict(v) for k, v in data.items()}
        for fold, data in itp_results.items():
            combined.setdefault(fold, {})
            for step, metrics in data.items():
                combined[fold].setdefault(step, {})
                combined[fold][step].update(metrics)
    except Exception as e:
        if debug: traceback.print_exc()
        raise RuntimeError(f"Failed to combine results for mean/std: {e}") from e

    all_steps = set()
    all_metric_names = set()
    for fold, fold_data in combined.items():
        for step in fold_data.keys():
            all_steps.add(step)
        for step_data in fold_data.values():
            all_metric_names |= set(step_data.keys())

    values_map: Dict[Tuple[Any, str], List[float]] = {}
    for step in all_steps:
        for metric in all_metric_names:
            values_map[(step, metric)] = []

    for f, fold_data in combined.items():
        for step, step_data in fold_data.items():
            for metric, value in step_data.items():
                values_map[(step, metric)].append(value)

    summary = []
    try:
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
    except Exception as e:
        if debug: traceback.print_exc()
        raise RuntimeError(f"Failed computing mean/std for step/metric ('{step}', '{metric}'): {e}") from e

    return summary

def generate_outputs_for_group(dataset: str, cdm: str, meta_trainer: str,
                               experiment_id: str, files: List[str], debug: bool = False):
    if not files:
        raise RuntimeError(f"No files for group {(dataset, cdm, meta_trainer, experiment_id)}")
    try:
        acc, itp = aggregate_results(files, debug=debug)
    except Exception as e:
        raise RuntimeError(f"Aggregating results failed for group {(dataset, cdm, meta_trainer, experiment_id)}: {e}") from e

    if not acc and not itp:
        raise RuntimeError(f"No metrics parsed for group {(dataset, cdm, meta_trainer, experiment_id)}")

    search_dir = os.path.dirname(files[0]) if files else "."
    try:
        basis_per_fold, basis_mean = load_cdm_basis_metrics(search_dir, dataset, cdm, debug=debug)
    except Exception as e:
        raise RuntimeError(f"Loading CDM_basis failed for dataset={dataset}, cdm={cdm} in dir={search_dir}: {e}") from e

    try:
        adjust_with_cdm_basis(acc, itp, basis_per_fold, basis_mean, debug=debug)
    except Exception as e:
        raise RuntimeError(f"Adjusting with CDM_basis failed for group {(dataset, cdm, meta_trainer, experiment_id)}: {e}") from e

    base = f"CAT_launch_{dataset}_{cdm}_{meta_trainer}_{experiment_id}"
    try:
        os.makedirs("logs", exist_ok=True)
    except Exception as e:
        raise IOError(f"Failed to create output directory 'logs': {e}") from e

    try:
        summary_data = compute_mean_and_std(acc, itp, debug=debug)
    except Exception as e:
        raise RuntimeError(f"Computing mean/std failed for group {(dataset, cdm, meta_trainer, experiment_id)}: {e}") from e

    csv_filename = f"logs/{base}_metrics_summary.csv"
    try:
        with open(csv_filename, 'w', newline='') as csvfile:
            fieldnames = ['step', 'metric', 'mean', 'std', 'n', 'type']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in summary_data:
                writer.writerow(row)
        print(f"Successfully wrote CSV file: {csv_filename}")
    except Exception as e:
        raise IOError(f"Failed writing CSV '{csv_filename}': {e}") from e

    json_summary_filename = f"logs/{base}_metrics_summary.json"
    try:
        with open(json_summary_filename, 'w') as jf:
            summary_dict = {idx: row for idx, row in enumerate(summary_data)}
            json.dump(summary_dict, jf, indent=4)
        print(f"Successfully wrote JSON file: {json_summary_filename}")
    except Exception as e:
        raise IOError(f"Failed writing JSON summary '{json_summary_filename}': {e}") from e

    all_results = {}
    try:
        fold_ids = sorted(set(acc.keys()) | set(itp.keys()))
        for fold_num in fold_ids:
            all_results[fold_num] = {0: acc.get(fold_num, {}), 1: itp.get(fold_num, {})}
    except Exception as e:
        raise RuntimeError(f"Building all_results dict failed for group {(dataset, cdm, meta_trainer, experiment_id)}: {e}") from e

    all_results_filename = f"logs/{base}_all_results.json"
    try:
        with open(all_results_filename, 'w') as jf:
            json.dump(all_results, jf, indent=4)
        print(f"Successfully wrote JSON file: {all_results_filename}")
    except Exception as e:
        raise IOError(f"Failed writing JSON '{all_results_filename}': {e}") from e

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pattern", default="./logs/CAT_launch_*_*_*_*_*.out")
    parser.add_argument("--debug", action="store_true", help="Print full tracebacks for debugging.")
    args = parser.parse_args()

    log_files = glob.glob(args.pattern)
    if not log_files:
        raise SystemExit(0)

    try:
        groups = group_by_experiment(log_files)
    except Exception as e:
        if args.debug:
            traceback.print_exc()
        raise SystemExit(f"[FATAL] group_by_experiment failed: {e}")

    if not groups:
        raise SystemExit(0)

    for (dataset, cdm, meta_trainer, experiment_id), files in sorted(groups.items()):
        files_sorted = sorted(files, key=split_experiment_key)
        try:
            generate_outputs_for_group(dataset, cdm, meta_trainer, experiment_id, files_sorted, debug=args.debug)
        except Exception as e:
            if args.debug:
                traceback.print_exc()
            print(f"[ERROR] Failed group (dataset={dataset}, cdm={cdm}, meta_trainer={meta_trainer}, exp_id={experiment_id}): {e}")
            # continue to next group
