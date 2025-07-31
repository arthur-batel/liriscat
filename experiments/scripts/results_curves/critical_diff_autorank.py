#!/usr/bin/env python3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from autorank import autorank, create_report, plot_stats 
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

# Collect metric values across all valid datasets/subalgos/folds/steps
results = {m: {a: [] for a in algorithms} for m in metrics}
files = glob.glob("CAT_launch_*_IMPACT_*_all_results.json") + \
        glob.glob("CAT_launch_*_NCDM_*_all_results.json")
print(f"[INFO] Found {len(files)} candidate result files.")

for file in files:
    print(f"[DEBUG] Processing file: {file}")
    m = pattern.match(os.path.basename(file))
    if not m:
        print(f"[WARN] Filename does not match expected pattern: {file}")
        continue
    ds, subalgo, algo = m["dataset"], m["subalgo"], m["algorithm"]
    print(f"[DEBUG] Parsed: dataset={ds}, subalgo={subalgo}, algo={algo}")
    if ds not in datasets or algo not in algorithms:
        print(f"[DEBUG] Skipping file due to unmatched dataset/algo: {ds}, {algo}")
        continue

    try:
        with open(file, "r") as f:
            folds = json.load(f)
        print(f"[DEBUG] Loaded {len(folds)} folds")
        for fold_idx, (pred, meta) in enumerate(folds):
            for step in pred:
                for metric in metrics:
                    value = pred[step].get(metric)
                    if value is None:
                        value = meta[step].get(metric)
                    if value is not None and not np.isnan(value):
                        results[metric][algo].append(value)
    except Exception as e:
        print(f"[ERROR] Failed to process {file}: {e}")

# Plotting using autorank
for metric in metrics:
    data = results[metric]
    filtered_data = {k: v for k, v in data.items() if len(v) > 0}

    print(f"[INFO] Evaluating metric '{metric}' with {len(filtered_data)} algorithms having data.")
    for algo, vals in filtered_data.items():
        print(f"  [DEBUG] {algo} → {len(vals)} values")

    if len(filtered_data) < 2:
        print(f"[WARN] Not enough algorithms with data to plot '{metric}' — skipping.")
        continue

    # Equalize lengths by trimming to shortest list
    min_len = min(len(v) for v in filtered_data.values())
    print(f"[INFO] Trimming all data to {min_len} samples")
    trimmed_data = {k: v[:min_len] for k, v in filtered_data.items()}
    df = pd.DataFrame(trimmed_data)

    # Statistical test + CD plot
    try:
        result = autorank(df, alpha=0.05, verbose=True)
        print(f"[DEBUG] autorank result object for metric '{metric}':")
        from pprint import pprint
        print("[DEBUG] autorank result object (pretty print):")
        pprint(result)


        report = create_report(result)
        if report is None:
            print(f"[WARN] create_report() failed for metric '{metric}', writing summary manually.")
            summary_path = os.path.join(output_dir, f"report_{metric}.txt")
            with open(summary_path, "w") as f:
                f.write("Manual report fallback:\n")
                f.write(str(result.df_res))
                f.write("\n\nRanks:\n")
                f.write(str(result.ranks))
                f.write("\n\nRanked groups:\n")
                f.write(str(result.rankdf))
            continue

        
        # Sauvegarde du rapport
        try:
            report_path = os.path.join(output_dir, f"report_{metric}.txt")
            with open(report_path, "w") as f:
                try:
                    report = create_report(result)
                    if not isinstance(report, str):
                        raise TypeError("create_report() did not return a string")
                    f.write(report)
                except Exception as e:
                    print(f"[WARN] create_report() failed for metric '{metric}', writing summary manually.")
                    from pprint import pprint
                    f.write("Fallback summary:\n")
                    pprint(result, stream=f)
            print(f"[✔] Saved statistical report: {report_path}")
        except Exception as e:
            print(f"[ERROR] Failed to save report for {metric}: {e}")


        try:
            plt.figure(figsize=(10, 4))
            # Correction manuelle pour contourner le bug de df_res manquant
            plot_stats(result.rankdf, cd=result.cd, reverse=True)
            plt.title(f"Critical Difference Diagram — {metric}")
            plt.tight_layout()
            fig_path = os.path.join(output_dir, f"cd_diagram_{metric}.png")
            plt.savefig(fig_path)
            plt.close()
            print(f"[✔] Saved CD diagram: {fig_path}")
        except Exception as e:
            print(f"[ERROR] Failed to generate CD plot for {metric}: {e}")

    except Exception as e:
        print(f"[ERROR] Failed to generate CD plot for {metric}: {e}")
