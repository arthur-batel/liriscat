#!/usr/bin/env python3
import os, re, csv, sys, argparse, math, glob
from collections import defaultdict, OrderedDict

import numpy as np
import matplotlib.pyplot as plt

# ---------------- Metadata ----------------
METHOD_ORDER = ["naive", "maml", "approxgap", "betacd", "method"]
METHOD_LABELS = {
    "naive": r"Simple-CAT", "maml": r"MAML", "approxgap": r"Approx-GAP",
    "betacd": r"BETA-CD", "method": r"MICAT",
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

# Matplotlib-style colors & markers to match your LaTeX figure
METHOD_COLORS = OrderedDict({
    "naive":     "#1f77b4",  # mplBlue
    "maml":      "#ff7f0e",  # mplOrange
    "approxgap": "#2ca02c",  # mplGreen
    "betacd":    "#8c564b",  # mplBrown
    "method":    "#d62728",  # mplRed (MICAT)
})
METHOD_MARKERS = {
    "naive":     "*",  # asterisk
    "maml":      "o",  # filled circle
    "approxgap": "^",  # filled triangle
    "betacd":    "D",  # filled diamond
    "method":    "s",  # filled square
}

# Metrics typically minimized (others are maximized by default)
DEFAULT_MINIMIZE = {"rmse","mae","pc-er"}

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
    if len(parts) >= 6 and name.endswith("_metrics_summary.csv") and parts[0]=="CAT" and parts[1]=="launch":
        dataset = parts[2]; cdm = parts[3]; meta = "_".join(parts[4:-2]); exp  = parts[-2]
        if exp.isdigit(): return dataset, cdm, meta, exp
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

def fmt_mean_std(mean, std):
    return f"{mean:.3f} $\\pm$ {std:.3f}"

def load_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))

def parse_objective_overrides(s):
    if not s: return set()
    return {m.strip() for m in s.split(",") if m.strip()}

DEFAULT_MAXIMIZE = set()

def decide_objective(metric, min_set, max_set):
    if metric in min_set: return "min"
    if metric in max_set: return "max"
    return "min" if metric in DEFAULT_MINIMIZE else "max"

def best_indices_tier(means, objective, tier_rel=0.0, tier_abs=0.0, tol=1e-15):
    valid = [(i, m) for i, m in enumerate(means) if m is not None]
    if not valid:
        return set()
    if objective == "min":
        best = min(m for _, m in valid)
        thr = best * (1.0 + tier_rel) + tier_abs + tol
        return {i for i, m in valid if m <= thr}
    else:
        best = max(m for _, m in valid)
        thr = best * (1.0 - tier_rel) - tier_abs - tol
        return {i for i, m in valid if m >= thr}

# ---------- Load & aggregate (methods) ----------
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
            if verbose: print(f"[WARN] Skip file with unexpected name: {fp}", file=sys.stderr)
            continue
        dataset_raw, cdm_raw, meta_raw, exp_id = parsed

        # keep skipping CDM_basis here; we'll load them separately as baselines
        if meta_raw.strip().lower() == "cdm_basis":
            if verbose: print(f"[INFO] Skipping CDM_basis summary: {fp}", file=sys.stderr)
            continue

        method_key = canon_method(meta_raw)
        if method_key is None:
            if verbose: print(f"[WARN] Unknown meta_trainer '{meta_raw}' in {fp}; skipping.", file=sys.stderr)
            continue

        cdm_label = canon_cdm(cdm_raw)
        dataset_label = canon_dataset(dataset_raw)
        dataset_key = dataset_raw.lower()

        try:
            rows = load_csv(fp)
        except Exception as e:
            if verbose: print(f"[WARN] Failed to read {fp}: {e}", file=sys.stderr)
            continue

        for row in rows:
            metric = row.get("metric", "").strip()
            if not metric: continue
            metrics_seen.add(metric)
            if target_metric and metric != target_metric: continue

            step_raw = row.get("step", "").strip()
            try:
                step = int(step_raw)
            except Exception:
                step = step_raw

            mean = row.get("mean"); std = row.get("std")
            if not (is_finite_number(mean) and is_finite_number(std)): continue
            mean = float(mean); std = float(std)

            ds = results.setdefault(dataset_key, {"_label": dataset_label, "steps": set(), "cells": {}})
            ds["steps"].add(step)
            prev = ds["cells"].get((step, method_key, cdm_label), {})
            if not isinstance(prev, dict):
                prev = {}
            prev[metric] = (mean, std)
            ds["cells"][(step, method_key, cdm_label)] = prev

    # finalize step ordering per dataset
    for d in results.values():
        try:
            d["steps"] = sorted(d["steps"], key=lambda x: (isinstance(x, str), x))
        except Exception:
            d["steps"] = sorted(list(d["steps"]))
    return results, metrics_seen

# ---------- Load CDM_basis baselines ----------
def build_cdm_basis(dirpath, verbose=True):
    """
    Returns:
      baselines[dataset_key][cdm_label][metric] = (mean, std)
    If multiple CDM_basis files exist for the same (dataset, cdm, metric),
    averages the reported means and stds.
    """
    files = glob.glob(os.path.join(dirpath, "CAT_launch_*_*_*_*_metrics_summary.csv"))
    baselines_accum = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    baselines = defaultdict(lambda: defaultdict(dict))

    for fp in files:
        parsed = parse_filename(fp)
        if not parsed:
            continue
        dataset_raw, cdm_raw, meta_raw, exp_id = parsed
        if meta_raw.strip().lower() != "cdm_basis":
            continue

        cdm_label = canon_cdm(cdm_raw)
        dataset_key = dataset_raw.lower()

        try:
            rows = load_csv(fp)
        except Exception as e:
            if verbose: print(f"[WARN] Failed to read basis {fp}: {e}", file=sys.stderr)
            continue

        for row in rows:
            metric = (row.get("metric", "") or "").strip()
            if not metric:
                continue
            mean = row.get("mean"); std = row.get("std")
            if not (is_finite_number(mean) and is_finite_number(std)):
                continue
            baselines_accum[dataset_key][cdm_label][metric].append((float(mean), float(std)))

    # average per key
    for dskey, dcdm in baselines_accum.items():
        for cdm_label, dmet in dcdm.items():
            for metric, arr in dmet.items():
                A = np.array(arr, dtype=float)  # k x 2
                baselines[dskey][cdm_label][metric] = (float(np.nanmean(A[:,0])),
                                                       float(np.nanmean(A[:,1])))

    return baselines

# ---------------- Plotting: Accuracy vs #questions per (dataset, CDM) ----------------
def plot_accuracy_per_pair(results, baselines, outdir="./fig_acc_curves", formats=("png","pdf"), quiet=False):
    """
    For each dataset present and for each CDM in ('CD-BPR','NCDM'), if data exists,
    draw Accuracy (mi_acc) vs number of submitted questions, one curve per meta-trainer,
    with fill_between for ±std. Also draw a horizontal CDM-basis line (and ±std band) if available.
    """
    os.makedirs(outdir, exist_ok=True)

    ordered_ds = [ds for ds in DATASET_ORDER if ds in results] + \
                 [ds for ds in sorted(results.keys()) if ds not in DATASET_ORDER]
    cdms = ("CD-BPR", "NCDM")

    for dskey in ordered_ds:
        ds = results[dskey]
        steps = ds["steps"]
        cells = ds["cells"]

        for cdm in cdms:
            # Collect series
            xs = np.array([int(s)+1 if isinstance(s, int) else s for s in steps], dtype=float)  # t = step+1
            have_any = False

            fig, ax = plt.subplots(figsize=(7.6, 5.2))
            for method in METHOD_ORDER:
                means = []
                stds  = []

                for st in steps:
                    d = cells.get((st, method, cdm))
                    if d and ("mi_acc" in d):
                        m, s = d["mi_acc"]
                        means.append(m); stds.append(s)
                    else:
                        means.append(np.nan); stds.append(np.nan)

                means = np.asarray(means, dtype=float)
                stds  = np.asarray(stds, dtype=float)
                mask  = np.isfinite(means) & np.isfinite(stds)

                if np.any(mask):
                    have_any = True
                    xplt = xs[mask]
                    yplt = means[mask]
                    splt = stds[mask]
                    color = METHOD_COLORS.get(method, "#333333")
                    marker = METHOD_MARKERS.get(method, "o")

                    # mean curve + markers
                    ax.plot(xplt, yplt, color=color, marker=marker, markersize=4,
                            linewidth=1.8, label=METHOD_LABELS[method])

                    # ± std band
                    ax.fill_between(xplt, yplt - splt, yplt + splt,
                                    color=color, alpha=0.15, linewidth=0)

            # Add CDM-basis horizontal line (+/- std) if available
            b = baselines.get(dskey, {}).get(cdm, {}).get("mi_acc", None)
            if b is not None:
                b_mean, b_std = b
                # Horizontal dashed line at the basis mean
                ax.axhline(y=b_mean, color="#555555", linestyle="--", linewidth=1.6, label="CDM basis")
                # Shaded horizontal band for ± std
                if math.isfinite(b_std) and b_std > 0:
                    ax.axhspan(b_mean - b_std, b_mean + b_std, color="#555555", alpha=0.10, linewidth=0)

            if not have_any and b is None:
                plt.close(fig)
                if not quiet:
                    print(f"[SKIP] {dskey} / {cdm}: no Accuracy data.")
                continue

            # Labels, grid, legend
            ds_label = ds.get("_label", dskey)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            ax.set_xlabel("Number of submitted questions (t)", fontsize=18)
            ax.set_ylabel("Accuracy", fontsize=18)
            ax.grid(True, linestyle=":", linewidth=0.7, alpha=0.6)
            leg = ax.legend(frameon=True, fontsize=12, loc="lower center")
            leg.get_frame().set_alpha(0.9)
            plt.tight_layout()

            base = os.path.join(outdir, f"acc_{dskey}_{cdm.replace(' ','_')}")
            for ext in formats:
                try:
                    fig.savefig(f"{base}.{ext}", dpi=300, bbox_inches="tight")
                    if not quiet:
                        print(f"[SAVE] {base}.{ext}")
                except Exception as e:
                    print(f"[WARN] Failed to save {base}.{ext}: {e}", file=sys.stderr)
            plt.close(fig)

# ---------- LaTeX generation (single-CDM tables) ----------
def latex_header_single_cdm(metric_label, cdm_name, colsep_pt):
    method_cols = " ".join(["c"] * len(METHOD_ORDER))
    methods_mc = " & ".join([f"\\multicolumn{{1}}{{c}}{{{METHOD_LABELS[m]}}}" for m in METHOD_ORDER])
    return (
f"""\\begin{{table*}}[t]
\\centering
\\scriptsize
\\setlength{{\\tabcolsep}}{{{colsep_pt:.2f}pt}}
\\renewcommand{{\\arraystretch}}{{1.0}}
\\resizebox{{\\linewidth}}{{!}}{{%
\\begin{{tabular}}{{ll {method_cols}}}
\\toprule
Dataset & \\makecell{{Number of\\\\ submitted\\\\ questions ($t$)}} & {methods_mc} \\\\
\\midrule
"""
    )

def latex_footer_single_cdm(metric_label, cdm_name):
    cap = (f"{metric_label.upper()} results across datasets using {cdm_name} and varying numbers of "
           f"submitted questions ($t$). The best (by the chosen objective) mean $\\pm$ std are bold; ties use a tight tier rule.")
    tag = f"{metric_label}-{cdm_name.replace(' ','-').lower()}"
    return (
f"""\\bottomrule
\\end{{tabular}}
}}
\\caption{{{cap}}}
\\label{{tab:results-{tag}}}
\\end{{table*}}
"""
    )

def _dataset_has_any_data_for_cdm(ds_block, cdm_filter):
    steps = ds_block["steps"]; cells = ds_block["cells"]
    for st in steps:
        for m in METHOD_ORDER:
            if (st, m, cdm_filter) in cells:
                return True
    return False

def render_table_for_metric_single_cdm(results, metric_key, objective_for_metric, cdm_filter,
                                       tier_rel, tier_abs, colsep_pt):
    available_ds = list(results.keys())
    ordered_ds = [ds for ds in DATASET_ORDER if ds in results] + \
                 [ds for ds in sorted(available_ds) if ds not in DATASET_ORDER]

    out = [latex_header_single_cdm(metric_key, cdm_filter, colsep_pt)]
    dash = r"\textemdash"
    any_row = False

    for ds in ordered_ds:
        ds_block = results[ds]
        if not _dataset_has_any_data_for_cdm(ds_block, cdm_filter):
            continue

        label = ds_block["_label"]
        steps = ds_block["steps"]
        cells = ds_block["cells"]

        for idx, step in enumerate(steps):
            pairs, means_only = [], []
            for m in METHOD_ORDER:
                tup = cells.get((step, m, cdm_filter))
                if isinstance(tup, dict) and (metric_key in tup):
                    mean, std = tup[metric_key]
                    pairs.append((mean, std)); means_only.append(mean)
                else:
                    pairs.append(None); means_only.append(None)

            bold_idx = best_indices_tier(means_only, objective_for_metric, tier_rel=tier_rel, tier_abs=tier_abs)

            row_lead = f"{label} & {step+1}" if isinstance(step, int) and idx == 0 else (f"        & {step+1}" if isinstance(step, int) else f"{label} & {step}" if idx==0 else f"        & {step}")
            row_cells = []
            for ci, p in enumerate(pairs):
                if p is None:
                    row_cells.append(dash)
                else:
                    mean, std = p
                    cell = fmt_mean_std(mean, std)
                    if ci in bold_idx:
                        cell = r"\textbf{" + cell + "}"
                    row_cells.append(cell)

            out.append(row_lead + " & " + " & ".join(row_cells) + r" \\")
            any_row = True

        out.append(r"\midrule")

    if out and out[-1] == r"\midrule":
        out.pop()

    if not any_row:
        out.append(r"\multicolumn{2}{l}{\emph{No data available}} \\")
    out.append(latex_footer_single_cdm(metric_key, cdm_filter))
    return "\n".join(out)

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser(description="Emit LaTeX tables and Accuracy curves per (dataset, CDM), plus CDM-basis reference lines.")
    ap.add_argument("--dir", default="./logs/", help="Directory containing *_metrics_summary.csv (input)")
    ap.add_argument("--outdir", default="../data/", help="Directory to write LaTeX files")
    ap.add_argument("--metric", default="mi_acc",
                    help="Single metric to output (e.g., mi_acc). If not set, outputs ALL metrics found.")
    ap.add_argument("--maximize", default="mi_acc", help="Comma-separated metrics to maximize (overrides defaults).")
    ap.add_argument("--minimize", default="", help="Comma-separated metrics to minimize (overrides defaults).")
    ap.add_argument("--cdms", default="CD-BPR,NCDM",
                    help="Comma-separated CDMs to emit (subset of: CD-BPR,NCDM); order respected.")
    ap.add_argument("--tier-rel", type=float, default=0.0,
                    help="Relative tier tolerance (fraction). Default 0.0 (strict).")
    ap.add_argument("--tier-abs", type=float, default=0.0,
                    help="Absolute tier tolerance. Default 0.0 (strict).")
    ap.add_argument("--colsep", type=float, default=4.0,
                    help="LaTeX \\tabcolsep in points (default: 4.0). Larger => more space between columns.")

    # New: figure outputs
    ap.add_argument("--fig-outdir", default="../data/", help="Output directory for Accuracy figures")
    ap.add_argument("--fig-formats", default="pdf", help="Comma-separated formats for figures (e.g., png,pdf)")
    ap.add_argument("--no-stdout", action="store_true", help="Do not print tables to stdout; only write files.")
    ap.add_argument("--quiet", action="store_true", help="Reduce warnings/info logs.")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    max_set = parse_objective_overrides(args.maximize)
    min_set = parse_objective_overrides(args.minimize)

    # Methods (curves)
    results, metrics_seen = build_results(args.dir, target_metric=args.metric, verbose=not args.quiet)
    if not results:
        print("[ERROR] No usable results found.", file=sys.stderr); sys.exit(2)

    # CDM-basis references
    baselines = build_cdm_basis(args.dir, verbose=not args.quiet)

    # ---- Tables (same as before) ----
    metrics_to_do = [args.metric] if args.metric else sorted(metrics_seen)
    want_cdms = [c.strip() for c in args.cdms.split(",") if c.strip()]
    for c in list(want_cdms):
        if c not in ("CD-BPR","NCDM"):
            if not args.quiet:
                print(f"[WARN] Unknown CDM '{c}' ignored; use 'CD-BPR' or 'NCDM'.", file=sys.stderr)
            want_cdms.remove(c)
    if not want_cdms:
        print("[ERROR] No valid CDMs requested.", file=sys.stderr); sys.exit(3)

    emitted_any = False
    for m in metrics_to_do:
        filtered, _ = build_results(args.dir, target_metric=m, verbose=False)
        if not filtered:
            if not args.quiet:
                print(f"[WARN] No data for metric '{m}', skipping.", file=sys.stderr)
            continue

        obj = decide_objective(m, max_set, min_set)  # (note: order doesn't matter here)
        for cdm in want_cdms:
            tex = render_table_for_metric_single_cdm(filtered, m, obj, cdm,
                                                     tier_rel=args.tier_rel, tier_abs=args.tier_abs,
                                                     colsep_pt=args.colsep)
            if not args.no_stdout:
                sys.stdout.write(tex + "\n\n")
            fname = f"results-{m}-{cdm.replace(' ','_')}-tierrel{args.tier_rel:.4g}-tierabs{args.tier_abs:.4g}-colsep{args.colsep:.2f}.tex"
            outpath = os.path.join(args.outdir, fname)
            try:
                with open(outpath, "w") as f:
                    f.write(tex)
                if not args.quiet:
                    print(f"[WRITE] {outpath}", file=sys.stderr)
            except Exception as e:
                print(f"[WARN] Failed to write {outpath}: {e}", file=sys.stderr)
            emitted_any = True

    if not emitted_any:
        print("[ERROR] No tables emitted (no matching metrics).", file=sys.stderr)

    # ---- Accuracy curves per (dataset, CDM) with CDM-basis line ----
    fig_formats = tuple(s.strip().lower() for s in args.fig_formats.split(",") if s.strip()) or ("png","pdf")
    plot_accuracy_per_pair(results, baselines, outdir=args.fig_outdir, formats=fig_formats, quiet=args.quiet)

if __name__ == "__main__":
    main()
