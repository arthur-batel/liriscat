#!/usr/bin/env python3
import glob
import re
import json
import os
import numpy as np
import matplotlib.pyplot as plt

def main():
    # Create output directory
    output_dir = "plots"
    os.makedirs(output_dir, exist_ok=True)
    print(f"[INFO] Output dir: {output_dir}")

    # only these datasets & variants
    datasets = ['algebra', 'assist0910', 'math2']
    variants = ['IMPACT', 'NCDM']
    alg_order = ["Adam", "Approx_GAP", "Beta_cd", "MICAT", "MAML"]

    # compile once
    pattern = re.compile(
        r"CAT_launch_(?P<dataset>[^_]+)_(?P<variant>IMPACT|NCDM)_(?P<algorithm>.*?)_\d+_all_results\.json$"
    )

    # list all matching files
    files = glob.glob("CAT_launch_*_IMPACT_*_all_results.json") + \
        glob.glob("CAT_launch_*_NCDM_*_all_results.json")

    print(f"[INFO] Found {len(files)} JSON files:")
    for f in files:
        print("   ", f)

    # organize: data[dataset][variant][algorithm] = list of folds
    data = {}
    for path in files:
        fname = os.path.basename(path)
        m = pattern.match(fname)
        if not m:
            print(f"[WARN] Filename did not match pattern, skipping: {fname}")
            continue
        ds = m.group("dataset")
        var = m.group("variant")
        alg = m.group("algorithm")
        if ds not in datasets:
            print(f"[WARN] Dataset '{ds}' not in {datasets}, skipping")
            continue

        # load JSON
        try:
            folds = json.load(open(path, "r"))
            print(f"[OK] Loaded {len(folds)} folds for {ds}/{var}/{alg}")
        except Exception as e:
            print(f"[ERROR] Could not load {path}: {e}")
            continue

        data.setdefault(ds, {}).setdefault(var, {})[alg] = folds

    # prepare color map
    base_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    color_map = {alg: base_colors[i % len(base_colors)]
                 for i, alg in enumerate(alg_order)}

    # now loop and plot
    for ds in datasets:
        for var in variants:
            algs = data.get(ds, {}).get(var, {})
            if not algs:
                print(f"[WARN] No data for {ds}/{var}, skipping plots")
                continue

            # determine number of folds from any one algorithm
            nfolds = len(next(iter(algs.values())))
            print(f"[INFO] Plotting {nfolds} folds for {ds}/{var}")

            for fold_idx in range(nfolds):
                plt.figure()
                any_plotted = False

                for alg in alg_order:
                    if alg not in algs:
                        print(f"[DEBUG]   {alg} not present in {ds}/{var}, skipping")
                        continue
                    folds = algs[alg]
                    if fold_idx >= len(folds):
                        print(f"[DEBUG]   {alg} has no fold {fold_idx}, skipping")
                        continue

                    entry = folds[fold_idx]
                    if not (isinstance(entry, (list, tuple)) and len(entry) >= 2):
                        print(f"[DEBUG]   bad entry for {alg} fold {fold_idx}, skipping")
                        continue

                    meta = entry[1]
                    steps = sorted(int(s) for s in meta.keys())
                    xs, ys = [], []
                    for s in steps:
                        v = meta.get(str(s), {}).get("meta_doa", None)
                        if v is None or (isinstance(v, float) and np.isnan(v)):
                            # skip missing / nan
                            continue
                        xs.append(s)
                        ys.append(v)

                    if not xs:
                        print(f"[DEBUG]   no valid meta_doa for {alg} fold {fold_idx}")
                        continue

                    plt.plot(xs, ys, color=color_map[alg], label=alg)
                    any_plotted = True
                    print(f"[PLOT]   {ds}/{var}/{alg} fold {fold_idx}: {len(xs)} points")

                if not any_plotted:
                    print(f"[WARN]   nothing plotted for {ds}/{var} fold {fold_idx}, skipping save")
                    plt.close()
                    continue

                plt.xlabel("Number of submitted questions")
                plt.ylabel("Meta_doa")
                plt.title(f"{ds} – {var} – fold {fold_idx}")
                plt.legend(title="Algorithm")
                plt.tight_layout()
                fname = f"{ds}_{var}_fold{fold_idx}_meta_doa.png"
                out_path = os.path.join(output_dir, fname)
                plt.savefig(out_path)
                plt.close()
                print(f"[SAVED] {out_path}")

    print("→ Done.")

if __name__ == "__main__":
    main()
