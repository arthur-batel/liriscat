#!/usr/bin/env python3
import os, re, csv, sys, argparse, math, glob
from collections import OrderedDict

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# ---------------- Metadata ----------------
METHOD_ORDER = ["naive", "maml", "approxgap", "betacd", "method"]
METHOD_LABELS = {
    "naive": r"\naive", "maml": r"\maml", "approxgap": r"\approxgap",
    "betacd": r"\betacd", "method": r"\method",
}
META_TRAINER_TO_METHOD = {
    "adam": "naive", "sgd": "naive", "naive": "naive",
    "maml": "maml",
    "approx_gap": "approxgap", "approx-gap": "approxgap", "approxgap": "approxgap",
    "beta_cd": "betacd", "beta-cd": "betacd", "betacd": "betacd",
    "micat": "method",
}
CDM_LABELS = {"ncdm": "NCDM", "impact": "CD-BPR", "cd-bpr": "CD-BPR", "cdbpr": "CD-BPR"}
DATASET_LABELS = {"algebra": "Algebra", "math2": "Math2", "assist0910": "Assist0910", "assist": "Assist0910"}
DATASET_ORDER = ["algebra", "math2", "assist0910"]

# Metrics typically minimized (others are maximized by default)
DEFAULT_MINIMIZE = {"rmse", "mae", "pc-er"}

# --- Filename parsing ---
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

# ---------- Pareto front utilities ----------
def pareto_front_indices(points, minimize_x=False, minimize_y=False):
    if len(points) == 0:
        return set()
    xf = np.array([(-p[0] if minimize_x else p[0]) for p in points], dtype=float)
    yf = np.array([(-p[1] if minimize_y else p[1]) for p in points], dtype=float)
    n = len(points)
    on_front = np.ones(n, dtype=bool)
    for i in range(n):
        if not on_front[i]:
            continue
        for j in range(n):
            if i == j:
                continue
            if (xf[j] >= xf[i] and yf[j] >= yf[i]) and (xf[j] > xf[i] or yf[j] > yf[i]):
                on_front[i] = False
                break
    return {i for i, v in enumerate(on_front) if v}

def objective_direction(metric_name):
    return (metric_name or "").strip().lower() in DEFAULT_MINIMIZE

# ---------- Plotting (write two files per dataset) ----------
def _draw_one_figure(dskey, ds, metric_x, metric_y, pairs_used, outpath, formats, quiet):
    """
    Colors & markers per your TikZ spec:
      Simple-CAT -> blue, asterisk (*)
      MAML       -> orange, circle (o)
      Approx-GAP -> green, triangle (^)
      BETA-CD    -> brown, diamond (D)
      MICAT      -> red, square (s)
    """
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch, Rectangle

    steps_sorted = ds["steps"]
    cells = ds["cells"]

    # ----- COLORS (Matplotlib "Tableau" defaults) -----
    method_colors = OrderedDict({
        "naive":    "#1f77b4",  # mplBlue
        "maml":     "#ff7f0e",  # mplOrange
        "approxgap":"#2ca02c",  # mplGreen
        "betacd":   "#8c564b",  # mplBrown
        "method":   "#d62728",  # mplRed
    })

    # ----- MARKERS (to match TikZ marks) -----
    method_markers = {
        "naive":    "o",  # asterisk
        "maml":     "o",  # filled circle
        "approxgap":"o",  # filled triangle up
        "betacd":   "o",  # filled diamond
        "method":   "o",  # filled square
    }

    # (CDM styles kept but we won't use their markers; we use per-method markers)
    cdm_linestyles = {"NCDM": "-", "CD-BPR": "-"}

    min_x = objective_direction(metric_x)
    min_y = objective_direction(metric_y)

    fig, ax = plt.subplots(figsize=(7.2, 5.6), constrained_layout=False)
    drawn = 0
    series = []

    # Collect data per series
    for (method, cdm) in pairs_used:
        xs, ys, xsd, ysd, steps_vec = [], [], [], [], []
        for st in steps_sorted:
            key = (st, method, cdm)
            if key not in cells:
                continue
            metric_dict = cells[key]
            if (metric_x not in metric_dict) or (metric_y not in metric_dict):
                continue
            mx, sx = metric_dict[metric_x]
            my, sy = metric_dict[metric_y]
            if not all(map(math.isfinite, [mx, my, sx, sy])):
                continue
            xs.append(mx); ys.append(my); xsd.append(sx); ysd.append(sy); steps_vec.append(st)
        if len(xs) == 0:
            continue

        color = method_colors.get(method, "#333333")
        marker = method_markers.get(method, "o")       # <-- use per-method marker
        ls = cdm_linestyles.get(cdm, "-")
        series.append({
            "method": method, "cdm": cdm, "color": color, "ls": ls, "marker": marker,
            "xs": np.asarray(xs), "ys": np.asarray(ys),
            "xsd": np.asarray(xsd), "ysd": np.asarray(ysd),
            "steps": np.asarray(steps_vec, int),
        })
        drawn += 1

    # Draw colored lines (no markers on the line itself, like your script)
    for s in series:
        ax.plot(s["xs"], s["ys"], linestyle=s["ls"], color=s["color"],
                label=f"Avg {METHOD_CONV.get(s['method'], s['method'].upper())}", alpha=1.0)

    # Error rectangles (unchanged)
    for s in series:
        if s["xs"].size > 1:
            mx, my, sx, sy = s["xs"][-1], s["ys"][-1], s["xsd"][-1], s["ysd"][-1]
            if all(map(math.isfinite, [mx, my, sx, sy])):
                rect = Rectangle((mx - sx, my - sy), 2*sx, 2*sy,
                                 facecolor=s["color"], edgecolor=s["color"], lw=1.0, alpha=0.15)
                ax.add_patch(rect)

    # Annotations (unchanged)
    for s in series:
        if s["xs"].size >= 1:
            ax.annotate(f"{1}", (s["xs"][0], s["ys"][0]), xytext=(4, 4),
                        textcoords="offset points", fontsize=10)
        if s["xs"].size >= 2:
            ax.annotate(f"{s['steps'][-1] + 1}", (s["xs"][-1], s["ys"][-1]),
                        xytext=(5, -12), textcoords="offset points", fontsize=10)

    # Pareto front highlighting with the per-method markers/colors
    if series:
        all_steps = sorted(set(int(st) for s in series for st in s["steps"]))
        for st in all_steps:
            pts, idx_map = [], []
            for si, s in enumerate(series):
                idx = np.where(s["steps"] == st)[0]
                if idx.size == 0:
                    continue
                pi = int(idx[0])
                x, y = s["xs"][pi], s["ys"][pi]
                if math.isfinite(x) and math.isfinite(y):
                    pts.append((x, y))
                    idx_map.append((si, pi))
            if not pts:
                continue
            front = pareto_front_indices(pts, minimize_x=min_x, minimize_y=min_y)
            for k, (si, pi) in enumerate(idx_map):
                s = series[si]
                x, y = s["xs"][pi], s["ys"][pi]
                if k in front:
                    ax.scatter(x, y, s=42, color=s["color"], marker=s["marker"],
                               edgecolor="black", linewidths=1.6, zorder=4)
                else:
                    ax.scatter(x, y, s=28, color=s["color"], marker=s["marker"],
                               edgecolor="none", alpha=0.9, zorder=3)

    # --- Unified legend (unchanged) ---
    handles, labels = ax.get_legend_handles_labels()

    from matplotlib.legend_handler import HandlerTuple, HandlerBase
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    class HandlerDotWithText(HandlerBase):
        def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
            dot = Line2D([width * 0.25], [height / 2.0],
                         color='gray', marker='o', linestyle='None', markersize=6,
                         transform=trans)
            txt = plt.Text(width * 0.55, height / 2.0, "1",
                           color='black', fontsize=10,
                           verticalalignment='center', horizontalalignment='left',
                           transform=trans)
            return [dot, txt]

    pareto_handle = Line2D([], [], color='gray', marker='o', linestyle='None', markeredgewidth=1.5,
                           markeredgecolor='black', markersize=6)
    std_handle = Patch(facecolor='gray', edgecolor='gray', alpha=0.15)
    dot1_handle = object()

    handles += [std_handle, pareto_handle, dot1_handle]
    labels  += ['± Std deviation area', 'Pareto front point\nper sub question nb', 'Nb of sub questions']

    leg = ax.legend(handles=handles, labels=labels,
                    handler_map={dot1_handle: HandlerDotWithText()},
                    frameon=True, fontsize=10, loc="best", title=None)
    leg.get_frame().set_alpha(0.9)

    # Axes setup
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    ax.set_xlabel(f"{METRIC_CONV.get(metric_x, metric_x)}",fontsize=18)
    ax.set_ylabel(f"{METRIC_CONV.get(metric_y, metric_y)}",fontsize=18)
    ax.grid(True, linestyle=":", linewidth=0.7, alpha=0.6)

    plt.tight_layout()

    # Save
    if drawn > 0:
        for fmt in formats:
            path = f"{outpath}.{fmt}"
            try:
                fig.savefig(path, dpi=200, bbox_inches="tight")
                if not quiet:
                    print(f"[SAVE] {path}")
            except Exception as e:
                print(f"[WARN] Failed to save {path}: {e}", file=sys.stderr)
    else:
        if not quiet:
            print(f"[SKIP] {dskey}: no curves with data.")
    plt.close(fig)

def plot_metric_vs_metric(results, metric_x, metric_y, outdir, formats=("png",), show=False, quiet=False):
    os.makedirs(outdir, exist_ok=True)
    available_ds = list(results.keys())
    ordered_ds = [ds for ds in DATASET_ORDER if ds in results] + \
                 [ds for ds in sorted(available_ds) if ds not in DATASET_ORDER]

    for dskey in ordered_ds:
        ds = results[dskey]
        cells = ds["cells"]
        # Only CD-BPR
        pairs_all = {(k[1], k[2]) for k in cells.keys() if str(k[2]).strip().lower() == "cd-bpr"}
        pairs_full = sorted(pairs_all, key=lambda p: (METHOD_ORDER.index(p[0]) if p[0] in METHOD_ORDER else 99))
        pairs_no_betacd = [p for p in pairs_full if p[0].lower() != "betacd"]

        base = os.path.join(outdir, f"fig_{dskey}_{metric_x}_vs_{metric_y}")
        _draw_one_figure(dskey, ds, metric_x, metric_y, pairs_full, base, formats, quiet)

        base_nb = os.path.join(outdir, f"fig_{dskey}_{metric_x}_vs_{metric_y}_no_betacd")
        _draw_one_figure(dskey, ds, metric_x, metric_y, pairs_no_betacd, base_nb, formats, quiet)

        if show:
            pass

METRIC_CONV = {'mi_acc': 'Accuracy', "meta_doa": "Meta-DOA"}
METHOD_CONV = {'method': 'MICAT', 'naive': 'Simple-CAT', 'approxgap': 'Approx-GAP', 'betacd': 'BETA-CD'}

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser(description="Plots with per-step Pareto front; Pareto points circled in black.")
    ap.add_argument("--dir", default="./logs/", help="Directory containing *_metrics_summary.csv")
    ap.add_argument("--metric-x", default="mi_acc", help="Metric for x-axis (e.g., rmse)")
    ap.add_argument("--metric-y", default="meta_doa", help="Metric for y-axis (e.g., mi_acc)")
    ap.add_argument("--outdir", default="../data/", help="Output directory for figures")
    ap.add_argument("--formats", default="pdf", help="Comma-separated formats (e.g., png,pdf)")
    ap.add_argument("--show", action="store_true", help="(Optional) Show figures interactively")
    ap.add_argument("--quiet", action="store_true", help="Reduce logs")
    args = ap.parse_args()

    results, metrics_seen = build_results(args.dir, target_metric=None, verbose=not args.quiet)

    if (args.metric_x not in metrics_seen) or (args.metric_y not in metrics_seen):
        missing = []
        if args.metric_x not in metrics_seen: missing.append(args.metric_x)
        if args.metric_y not in metrics_seen: missing.append(args.metric_y)
        if missing:
            print(f"[WARN] Requested metric(s) not found: {', '.join(missing)}", file=sys.stderr)

    formats = tuple(s.strip().lower() for s in args.formats.split(",") if s.strip()) or ("png",)
    plot_metric_vs_metric(results, metric_x=args.metric_x, metric_y=args.metric_y,
                          outdir=args.outdir, formats=formats, show=args.show, quiet=args.quiet)

if __name__ == "__main__":
    main()
