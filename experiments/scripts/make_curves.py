#!/usr/bin/env python3
import os, re, csv, sys, argparse, math, glob
from collections import defaultdict, OrderedDict

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle  # <-- NEW

# ---------------- Existing metadata ----------------
METHOD_ORDER = ["naive", "maml", "approxgap", "betacd", "method"]
METHOD_LABELS = {
    "naive": r"\naive", "maml": r"\maml", "approxgap": r"\approxgap",
    "betacd": r"\betacd", "method": r"\method",
}
META_TRAINER_TO_METHOD = {
    "adam":"naive","sgd":"naive","naive":"naive",
    "maml":"maml",
    "approx_gap":"approxgap","approx-gap":"approxgap","approxgap":"approxgap",
    "beta_cd":"betacd","beta-cd":"betacd","betacd":"betacd",
    "micat":"method",
}
CDM_LABELS = {"ncdm":"NCDM","impact":"CD-BPR","cd-bpr":"CD-BPR","cdbpr":"CD-BPR"}
DATASET_LABELS = {"algebra":r"\algebra{}","math2":"Math2","assist0910":r"\assist{}","assist":r"\assist{}"}
DATASET_ORDER = ["algebra","math2","assist0910"]

# --- Filename parsing (robust) ---
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
    # Fallback
    parts = name.split("_")
    if len(parts) >= 6 and name.endswith("_metrics_summary.csv") and parts[0]=="CAT" and parts[1]=="launch":
        dataset = parts[2]; cdm = parts[3]; meta = "_".join(parts[4:-2]); exp  = parts[-2]
        if exp.isdigit():
            return dataset, cdm, meta, exp
    return None

def canon_method(meta_raw):
    k = meta_raw.strip().lower()
    return META_TRAINER_TO_METHOD.get(k)

def canon_cdm(cdm_raw):
    k = cdm_raw.strip().lower()
    return CDM_LABELS.get(k, cdm_raw)

def canon_dataset(ds_raw):
    k = ds_raw.strip().lower()
    return DATASET_LABELS.get(k, ds_raw)

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
def build_results(dirpath, target_metric=None, verbose=True):
    """
    results[dataset_key]['_label'] = display_name
    results[dataset_key]['steps'] = sorted_steps
    results[dataset_key]['cells'][(step, method_key, cdm_label)] = {metric: (mean,std), ...}
    """
    results = {}
    metrics_seen = set()
    files = glob.glob(os.path.join(dirpath, "CAT_launch_*_*_*_*_metrics_summary.csv"))
    if not files:
        print(f"[ERROR] No metrics_summary.csv found in {dirpath}", file=sys.stderr)
        sys.exit(1)

    for fp in files:
        parsed = parse_filename(fp)
        if not parsed:
            if verbose:
                print(f"[WARN] Skip file with unexpected name: {fp}", file=sys.stderr)
            continue
        dataset_raw, cdm_raw, meta_raw, exp_id = parsed

        if meta_raw.strip().lower() == "cdm_basis":
            if verbose:
                print(f"[INFO] Skipping CDM_basis summary (not a method): {fp}", file=sys.stderr)
            continue

        method_key = canon_method(meta_raw)
        if method_key is None:
            if verbose:
                print(f"[WARN] Unknown meta_trainer '{meta_raw}' in {fp}; skipping.", file=sys.stderr)
            continue

        cdm_label = canon_cdm(cdm_raw)
        dataset_label = canon_dataset(dataset_raw)
        dataset_key = dataset_raw.lower()

        try:
            rows = load_csv(fp)
        except Exception as e:
            if verbose:
                print(f"[WARN] Failed to read {fp}: {e}", file=sys.stderr)
            continue

        for row in rows:
            metric = row.get("metric", "").strip()
            if not metric:
                continue
            metrics_seen.add(metric)
            if target_metric and metric != target_metric:
                continue

            step_raw = row.get("step", "").strip()
            try:
                step = int(step_raw)
            except Exception:
                step = step_raw

            mean = row.get("mean"); std = row.get("std")
            if not (is_finite_number(mean) and is_finite_number(std)):
                continue
            mean = float(mean); std = float(std)

            ds = results.setdefault(dataset_key, {"_label": dataset_label, "steps": set(), "cells": {}})
            ds["steps"].add(step)
            key = (step, method_key, cdm_label)
            prev = ds["cells"].get(key, {})
            prev = prev if isinstance(prev, dict) else {}
            prev[metric] = (mean, std)
            ds["cells"][key] = prev

    for dskey, d in results.items():
        try:
            d["steps"] = sorted(d["steps"], key=lambda x: (isinstance(x, str), x))
        except Exception:
            d["steps"] = sorted(list(d["steps"]))
    return results, metrics_seen

# ---------- Plotting: mean(metric_y) vs mean(metric_x) ----------
def plot_metric_vs_metric(results,
                          metric_x: str,
                          metric_y: str,
                          outdir: str,
                          formats=("png",),
                          show=False,
                          quiet=False):
    """
    For each dataset in `results`, produce a figure with:
        x: mean(metric_x)
        y: mean(metric_y)
    One curve per (method, cdm), points ordered by step and connected.
    Additionally, draw rectangular error boxes (±std in both metrics) at the
    initial and final steps of each curve.
    """
    os.makedirs(outdir, exist_ok=True)

    # Styling: color per METHOD, marker/linestyle per CDM
    method_colors = OrderedDict({
        "naive":  "#1f77b4",
        "maml":   "#ff7f0e",
        "approxgap": "#2ca02c",
        "betacd": "#d62728",
        "method": "#9467bd",
    })
    default_colors = ["#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]

    cdm_markers = OrderedDict({"NCDM": "o", "CD-BPR": "o"})
    cdm_linestyles = OrderedDict({"NCDM": "-", "CD-BPR": "-"})

    available_ds = list(results.keys())
    ordered_ds = [ds for ds in DATASET_ORDER if ds in results] + \
                 [ds for ds in sorted(available_ds) if ds not in DATASET_ORDER]

    for dskey in ordered_ds:
        ds = results[dskey]
        steps_sorted = ds["steps"]
        cells = ds["cells"]  # (step, method, cdm) -> {metric: (mean,std), ...}

        # Collect all (method, cdm) present and DROP NCDM + BETACD
        pairs = set((k[1], k[2]) for k in cells.keys())
        pairs = {
            p for p in pairs
            if str(p[1]).strip().lower() != "ncdm"     # drop NCDM CDM
        }


        def _pair_sort_key(p):
            m, c = p
            mo = METHOD_ORDER.index(m) if m in METHOD_ORDER else len(METHOD_ORDER)
            return (mo, c)
        pairs_sorted = sorted(pairs, key=_pair_sort_key)

        fig, ax = plt.subplots(figsize=(7.2, 5.6), constrained_layout=False)

        drawn = 0
        color_cycle_extra = iter(default_colors)
        for (method, cdm) in pairs_sorted:
            xs, ys, xsd, ysd, labels_steps = [], [], [], [], []

            for st in steps_sorted:
                key = (st, method, cdm)
                if key not in cells:
                    continue
                metric_dict = cells[key]
                if (metric_x not in metric_dict) or (metric_y not in metric_dict):
                    continue
                mx, sx = metric_dict[metric_x]
                my, sy = metric_dict[metric_y]
                if not (math.isfinite(mx) and math.isfinite(my) and math.isfinite(sx) and math.isfinite(sy)):
                    continue
                xs.append(mx); ys.append(my); xsd.append(sx); ysd.append(sy); labels_steps.append(st)

            if len(xs) == 0:
                continue

            color = method_colors.get(method, next(color_cycle_extra, "#333333"))
            marker = cdm_markers.get(cdm, "x")
            ls = cdm_linestyles.get(cdm, "-.")

            # mean curve
            ax.plot(xs, ys, linestyle=ls, marker=marker,
                    label=f"Avg {METHOD_CONV.get(method,method.upper())}", color=color)

            # ---------- NEW: error boxes at initial and final steps ----------
            # helper to draw one rectangle given center (mx,my) and stds (sx,sy)
            def draw_error_box(mx, my, sx, sy, ec, fc_alpha=0.18, lw=1.0,method=""):
                if not (math.isfinite(mx) and math.isfinite(my) and math.isfinite(sx) and math.isfinite(sy)):
                    return
                rect = Rectangle((mx - sx, my - sy), 2*sx, 2*sy,
                                 facecolor=ec, edgecolor=ec, lw=lw, alpha=fc_alpha, label= f"Avg ± Std  {METHOD_CONV.get(method,method.upper())}")
                ax.add_patch(rect)

            # first point
            #draw_error_box(xs[0], ys[0], xsd[0], ysd[0], color)
            # last point (if distinct)
            if len(xs) > 1:
                draw_error_box(xs[-1], ys[-1], xsd[-1], ysd[-1], color,method=method)


            # (optional) annotate first and last step labels
            ax.annotate(f"{1}", (xs[0], ys[0]), xytext=(4, 4),
                        textcoords="offset points", fontsize=8)
            if len(xs) >= 2:
                ax.annotate(f"{labels_steps[-1]+1}", (xs[-1], ys[-1]), xytext=(4, -10),
                            textcoords="offset points", fontsize=8)

            drawn += 1

        ax.set_xlabel(f"{METRIC_CONV.get(metric_x,metric_x)}")
        ax.set_ylabel(f"{METRIC_CONV.get(metric_y,metric_y)}")
        ds_label_raw = ds.get("_label", dskey)
        title = str(ds_label_raw).replace("\\", "") if isinstance(ds_label_raw, str) else str(ds_label_raw)
        
        ax.grid(True, linestyle=":", linewidth=0.7, alpha=0.6)
        if drawn > 0:
            leg = ax.legend(frameon=True, fontsize=9, loc="best")
            leg.get_frame().set_alpha(0.9)

        plt.tight_layout()

        if drawn > 0:
            for fmt in formats:
                outpath = os.path.join(outdir, f"fig_{dskey}_{metric_x}_vs_{metric_y}.{fmt}")
                try:
                    fig.savefig(outpath, dpi=200, bbox_inches="tight")
                    if not quiet:
                        print(f"[SAVE] {outpath}")
                except Exception as e:
                    print(f"[WARN] Failed to save {outpath}: {e}", file=sys.stderr)
        else:
            if not quiet:
                print(f"[SKIP] {dskey}: no non-NCDM curves with data.")

        if show:
            plt.show()
        else:
            plt.close(fig)

METRIC_CONV = {'mi_acc':'Accuracy',"meta_doa":"Meta-DOA"}
METHOD_CONV = {'method':'MICAT', 'naive':'Naive-CAT', 'approxgap':'Approx-GAP','betacd':'BETA-CD'}

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser(description="Plot mean(metric_y) vs mean(metric_x) per dataset with error boxes at first/last steps.")
    ap.add_argument("--dir", default=".", help="Directory containing *_metrics_summary.csv")
    ap.add_argument("--metric-x", required=True, help="Metric name for x-axis (e.g., rmse)")
    ap.add_argument("--metric-y", required=True, help="Metric name for y-axis (e.g., mi_acc)")
    ap.add_argument("--outdir", default="./figures_metric_vs_metric", help="Output directory for figures")
    ap.add_argument("--formats", default="png", help="Comma-separated formats to save (e.g., png,pdf)")
    ap.add_argument("--show", action="store_true", help="Show figures interactively")
    ap.add_argument("--quiet", action="store_true", help="Reduce logs")
    args = ap.parse_args()

    results, metrics_seen = build_results(args.dir, target_metric=None, verbose=not args.quiet)

    if (args.metric_x not in metrics_seen) or (args.metric_y not in metrics_seen):
        missing = []
        if args.metric_x not in metrics_seen: missing.append(args.metric_x)
        if args.metric_y not in metrics_seen: missing.append(args.metric_y)
        print(f"[WARN] Requested metric(s) not found in any file: {', '.join(missing)}", file=sys.stderr)

    formats = tuple(s.strip().lower() for s in args.formats.split(",") if s.strip())
    if not formats:
        formats = ("png",)

    plot_metric_vs_metric(results, metric_x=args.metric_x, metric_y=args.metric_y,
                          outdir=args.outdir, formats=formats, show=args.show, quiet=args.quiet)

if __name__ == "__main__":
    main()
