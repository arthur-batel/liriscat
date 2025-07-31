#!/usr/bin/env python3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scikit_posthocs as sp
import json
import glob
import re
import os

# Configuration
algorithms = ["Adam", "Approx_GAP", "Beta_cd", "MICAT", "MAML"]
datasets = ["algebra", "assist0910", "math2"]
subalgos = ["IMPACT", "NCDM"]
metrics = ["mi_acc", "mi_auc", "meta_doa"]
pattern = re.compile(r"CAT_launch_(?P<dataset>[^_]+)_(?P<subalgo>IMPACT|NCDM)_(?P<algorithm>[^_]+)_\d+_all_results\.json")
output_dir = "cd_plots"
os.makedirs(output_dir, exist_ok=True)

# Collect metric values per algorithm
metric_values = {m: {alg: [] for alg in algorithms} for m in metrics}
files = files = glob.glob("CAT_launch_*_IMPACT_*_all_results.json") + \
        glob.glob("CAT_launch_*_NCDM_*_all_results.json")

print(f"[INFO] Found {len(files)} result files.")

for file in files:
    m = pattern.match(os.path.basename(file))
    if not m:
        continue
    ds, subalgo, algo = m["dataset"], m["subalgo"], m["algorithm"]
    if ds not in datasets or algo not in algorithms:
        continue

    try:
        with open(file, "r") as f:
            folds = json.load(f)
        for fold_idx, (pred, meta) in enumerate(folds):
            for step in pred:
                for metric in metrics:
                    val = pred[step].get(metric) or meta[step].get(metric)
                    if val is not None and not np.isnan(val):
                        metric_values[metric][algo].append(val)
    except Exception as e:
        print(f"[ERROR] Failed to process {file}: {e}")

# Plotting
for metric in metrics:
    algo_means = {}
    for algo in algorithms:
        vals = metric_values[metric][algo]
        if vals:
            mean_val = np.mean(vals)
            algo_means[algo] = mean_val
            print(f"[INFO] {algo} average {metric} over {len(vals)} values: {mean_val:.4f}")
        else:
            print(f"[WARN] No data for {algo} on metric '{metric}'")

    if len(algo_means) < 2:
        print(f"[WARN] Not enough algorithms with data to plot '{metric}' — skipping.")
        continue

    # Sort by descending performance
    sorted_algos = sorted(algo_means, key=algo_means.get, reverse=True)
    scores = [algo_means[alg] for alg in sorted_algos]

    # Create fake "evaluation points" (replicated once) to use posthoc ranking
    data = np.array([scores])
    ranks = np.argsort(np.argsort(-data, axis=1), axis=1) + 1
    avg_ranks = ranks.mean(axis=0)

    # CD Plot
    plt.figure(figsize=(10, 4))
    sp.sign_plot(avg_ranks, sorted_algos, cd=None, reverse=True)
    plt.title(f"Critical Difference Diagram — {metric}")
    plt.tight_layout()
    path = os.path.join(output_dir, f"cd_diagram_{metric}.png")
    plt.savefig(path)
    plt.close()
    print(f"[✔] Saved CD diagram: {path}")
