#!/usr/bin/env python3
import os, re, csv, sys, argparse, math, glob
from collections import OrderedDict, defaultdict
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# ---------------- Config / Metadata ----------------
# Target ablation methods (normalized keys below)
ABLATION_METHODS = [
    "MICAT",
    "MICAT_CROSS",
    "MICAT_CROSS_PREC",
    "MICAT_INIT",
    "MICAT_INIT_CROSS",
    "MICAT_INIT_PREC",
    "MICAT_PREC",
]
# Pretty labels (feel free to adjust)
METHOD_LABELS = {
    "MICAT": "MICAT",
    "MICAT_CROSS": "Cross-weighting",
    "MICAT_CROSS_PREC": "Cross-weighting + Preconditioners",
    "MICAT_INIT": "Init",
    "MICAT_INIT_CROSS": "Init + Cross-weighting",
    "MICAT_INIT_PREC": "Init + Preconditioners",
    "MICAT_PREC": "Preconditioners",
}
# --- Matplotlib default colors (Tableau palette) ---
default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

# MICAT = red (#d62728), others cycle through remaining default colors
METHOD_COLORS = OrderedDict({
    "MICAT": "#d62728",  # Matplotlib red
    "MICAT_CROSS": default_colors[0],
    "MICAT_CROSS_PREC": default_colors[1],
    "MICAT_INIT": default_colors[2],
    "MICAT_INIT_CROSS": default_colors[6],
    "MICAT_INIT_PREC": default_colors[4],
    "MICAT_PREC": default_colors[5],
})

# Which CDM(s) we accept; default is CD-BPR (IMPACT)
CDM_LABELS = {"ncdm": "NCDM", "impact": "IMPACT", "cd-bpr": "CD-BPR", "cdbpr": "CD-BPR"}
DATASET_LABELS = {"algebra": "Algebra", "math2": "Math2", "assist0910": "Assist0910", "assist": "Assist0910"}

# --- Filename parsing: CAT_launch_<dataset>_<cdm>_<meta>_<exp>_metrics_summary.csv
FNAME_RE = re.compile(
    r"^CAT_launch_(?P<dataset>[^_]+)_(?P<cdm>[^_]+)_(?P<meta>.+)_(?P<exp>\d+)_metrics_summary\.csv$",
    re.IGNORECASE,
)

def parse_filename(path):
    name = os.path.basename(path)
    m = FNAME_RE.match(name)
    if m:
        d = m.groupdict()
        return d["dataset"], d["cdm"], d["meta"], d["exp"]
    parts = name.split("_")
    if len(parts) >= 6 and name.endswith("_metrics_summary.csv") and parts[0] == "CAT" and parts[1] == "launch":
        dataset = parts[2]; cdm = parts[3]; meta = "_".join(parts[4:-2]); exp = parts[-2]
        if exp.isdigit():
            return dataset, cdm, meta, exp
    return None

def canon_cdm(cdm_raw):
    return CDM_LABELS.get(cdm_raw.strip().lower(), cdm_raw)

def canon_dataset(ds_raw):
    return DATASET_LABELS.get(ds_raw.strip().lower(), ds_raw)

def canon_method_name(meta_raw):
    """
    Normalize meta string to one of ABLATION_METHODS:
    - uppercase
    - replace hyphens with underscores
    - common aliases for CROSS vs CW, PREC vs PRECOND, etc. (if needed)
    """
    k = meta_raw.strip().upper().replace("-", "_")
    # Allow common alias "CW" -> "CROSS"
    k = k.replace("CW", "CROSS")
    # PREC already fine; if "PRECOND" used somewhere, normalize:
    k = k.replace("PRECOND", "PREC")
    return k

def is_finite_number(x):
    try:
        v = float(x)
        return math.isfinite(v)
    except Exception:
        return False

def load_csv(path):
    rows = []
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)
    return rows

# ---------- Load & aggregate ----------
def load_ablation(dirpath, metric_key, cdm_wanted="CD-BPR", dataset_filter=None, quiet=False):
    """
    Returns:
      steps_sorted: sorted list of steps (ints)
      curves: dict method_key -> dict step -> list of (mean,std) across matching files/datasets
              (we keep a list to average if multiple datasets match)
    Only includes methods in ABLATION_METHODS and CDM == cdm_wanted (if provided).
    If dataset_filter is provided (e.g., 'assist0910' or display label), restricts to that dataset only.
    """
    files = glob.glob(os.path.join(dirpath, "CAT_launch_*_*_*_*_metrics_summary.csv"))
    if not files:
        print(f"[ERROR] No metrics_summary.csv found in {dirpath}", file=sys.stderr)
        sys.exit(1)

    curves_raw = {m: defaultdict(list) for m in ABLATION_METHODS}
    steps_all = set()

    for fp in files:
        parsed = parse_filename(fp)
        if not parsed:
            if not quiet:
                print(f"[WARN] Skip unexpected file name: {fp}", file=sys.stderr)
            continue
        dataset_raw, cdm_raw, meta_raw, exp_id = parsed
        cdm_label = canon_cdm(cdm_raw)
        if cdm_wanted and cdm_label != cdm_wanted:
            continue

        ds_disp = canon_dataset(dataset_raw)
        # If dataset filter is provided, accept either raw key or display label
        if dataset_filter:
            if dataset_filter.lower() != dataset_raw.lower() and dataset_filter.lower() != ds_disp.lower():
                continue

        method_key = canon_method_name(meta_raw)
        if method_key not in ABLATION_METHODS:
            continue  # ignore other trainers/methods

        try:
            rows = load_csv(fp)
        except Exception as e:
            if not quiet:
                print(f"[WARN] Failed to read {fp}: {e}", file=sys.stderr)
            continue

        for row in rows:
            metric = row.get("metric", "").strip()
            if metric != metric_key:
                continue
            step_raw = row.get("step", "").strip()
            try:
                step = int(step_raw)
            except Exception:
                continue  # steps should be integers here

            mean = row.get("mean"); std = row.get("std")
            if not (is_finite_number(mean) and is_finite_number(std)):
                continue
            mean = float(mean); std = float(std)

            curves_raw[method_key][step].append((mean, std))
            steps_all.add(step)

    steps_sorted = sorted(list(steps_all))
    return steps_sorted, curves_raw

def average_curves(steps_sorted, curves_raw):
    """
    Average across files/datasets if multiple entries exist for a given (method, step).
    Returns: curves_mean, curves_std  (both dict method -> np.array aligned with steps_sorted)
    The 'std' here is the average of reported stds (not re-computed across runs),
    as the inputs are already "mean ± std" summaries.
    """
    curves_mean = {}
    curves_std = {}
    for m in ABLATION_METHODS:
        ys = []
        ys_std = []
        for st in steps_sorted:
            vals = curves_raw[m].get(st, [])
            if not vals:
                ys.append(np.nan)
                ys_std.append(np.nan)
            else:
                arr = np.array(vals, dtype=float)  # shape (k,2): mean,std
                ys.append(np.nanmean(arr[:,0]))
                ys_std.append(np.nanmean(arr[:,1]))
        curves_mean[m] = np.array(ys, dtype=float)
        curves_std[m]  = np.array(ys_std, dtype=float)
    return curves_mean, curves_std

# ---------- Plotting ----------
def plot_metric_over_steps(steps, curves_mean, curves_std, metric_key, outpath, show=False, quiet=False, last_box_width=0.8):
    """
    Plot mean(metric) vs step for all methods; draw a single std-rectangle at the last step.
    - steps: list[int]
    - curves_mean/std: dict method -> np.array aligned to steps
    - metric_key: 'mi_acc' or 'meta_doa'
    - last_box_width: width in X units (steps) for the std rectangle
    """
    if not steps:
        if not quiet:
            print("[SKIP] No steps to plot.", file=sys.stderr)
        return

    fig, ax = plt.subplots(figsize=(7.6, 5.2))

    # remove the plt.rc(...) line and use:
    ax.set_prop_cycle(linestyle=['-', '--', '-.', ':'])

    # Lines only (no dots)
    for m in ABLATION_METHODS:
        y = curves_mean[m]
        if np.all(~np.isfinite(y)):
            continue
        ax.plot(steps, y, label=METHOD_LABELS.get(m, m), color=METHOD_COLORS.get(m, "#333333"), marker=None)

    # Draw std "moustache" (error bar) at last submitted question for each method
    # last_x = steps[-1]
    # for m in ABLATION_METHODS:
    #     y = curves_mean[m]; s = curves_std[m]
    #     if not (np.isfinite(y[-1]) and np.isfinite(s[-1])):
    #         continue
    #     ax.errorbar(
    #         last_x, y[-1],
    #         yerr=s[-1],
    #         fmt='none',
    #         ecolor=METHOD_COLORS.get(m, "#333333"),
    #         elinewidth=1.8,
    #         capsize=5,
    #         capthick=1.4,
    #         alpha=0.9,
    #         zorder=5
    #     )


    # Labels, legend, grid
    ylab = "Accuracy" if metric_key == "mi_acc" else ("Meta-DOA" if metric_key == "meta_doa" else metric_key)
    ax.set_xlabel("Number of submitted questions", fontsize=18)
    ax.set_ylabel(ylab, fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    ax.grid(True, linestyle=":", linewidth=0.7, alpha=0.6)

    leg = ax.legend(frameon=True, fontsize=10, loc="best")
    leg.get_frame().set_alpha(0.9)

    plt.tight_layout()
    # Save
    base = f"{outpath}_{metric_key}"
    for ext in ("png", "pdf"):
        try:
            fig.savefig(f"{base}.{ext}", dpi=300, bbox_inches="tight")
            if not quiet:
                print(f"[SAVE] {base}.{ext}")
        except Exception as e:
            print(f"[WARN] Failed to save {base}.{ext}: {e}", file=sys.stderr)
    if show:
        plt.show()
    else:
        plt.close(fig)

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser(
        description="Ablation plots for MICAT variants: average metric (mi_acc, meta_doa) vs number of submitted questions."
    )
    ap.add_argument("--dir", default="./results_ablation3/", help="Directory containing *_metrics_summary.csv")
    ap.add_argument("--cdm", default="IMPACT", help="CDM to use (e.g., 'CD-BPR' or 'NCDM'). Default: CD-BPR")
    ap.add_argument("--dataset", default="assist0910", help="Dataset filter (raw key or display label). If omitted, averages across all datasets found.")
    ap.add_argument("--outdir", default="../data/", help="Output directory")
    ap.add_argument("--show", action="store_true", help="Show figures interactively")
    ap.add_argument("--quiet", action="store_true", help="Reduce logs")
    ap.add_argument("--boxwidth", type=float, default=0.8, help="Width (in step units) of the std rectangle at the last step")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    for metric_key in ("mi_acc", "meta_doa"):
        steps, curves_raw = load_ablation(args.dir, metric_key=metric_key, cdm_wanted=args.cdm, dataset_filter=args.dataset, quiet=args.quiet)
        if not steps:
            print(f"[WARN] No data for metric '{metric_key}' (cdm={args.cdm}, dataset={args.dataset}).", file=sys.stderr)
            continue
        curves_mean, curves_std = average_curves(steps, curves_raw)
        outbase = os.path.join(args.outdir, "ablation")
        plot_metric_over_steps(steps, curves_mean, curves_std, metric_key, outbase, show=args.show, quiet=args.quiet, last_box_width=args.boxwidth)

if __name__ == "__main__":
    main()
