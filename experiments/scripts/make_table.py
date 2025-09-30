#!/usr/bin/env python3
import os, re, csv, sys, argparse, math, glob
from collections import defaultdict

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
# Accept: CAT_launch_<dataset>_<cdm>_<meta>_<exp>_metrics_summary.csv
# where tokens may contain letters/digits/hyphens/mixed case.
# --- Filename parsing (robust) ---
# Accepts meta names with underscores (Beta_cd, Approx_GAP, etc.)
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
    # Fallback for weird edge cases
    parts = name.split("_")
    if len(parts) >= 6 and name.endswith("_metrics_summary.csv") and parts[0]=="CAT" and parts[1]=="launch":
        dataset = parts[2]
        cdm = parts[3]
        meta = "_".join(parts[4:-2])  # meta may contain underscores
        exp  = parts[-2]
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

def fmt_mean_std(mean, std):
    return f"{mean:.3f} $\\pm$ {std:.3f}"

def load_csv(path):
    rows = []
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)
    return rows

# Objective helpers
def parse_objective_overrides(s):
    if not s: return set()
    return {m.strip() for m in s.split(",") if m.strip()}

DEFAULT_MINIMIZE = {"rmse","mae","pc-er"}
DEFAULT_MAXIMIZE = set()  # all others default to maximize

def decide_objective(metric, min_set, max_set):
    if metric in min_set: return "min"
    if metric in max_set: return "max"
    return "min" if metric in DEFAULT_MINIMIZE else "max"

def best_indices(values, objective):
    valid = [(i, v) for i, v in enumerate(values) if v is not None]
    if not valid: return set()
    if objective == "min":
        bestv = min(v for _, v in valid)
    else:
        bestv = max(v for _, v in valid)
    return {i for i, v in valid if v == bestv}

# ---------- Load & aggregate ----------
def build_results(dirpath, target_metric=None, verbose=True):
    """
    results[dataset_key]['_label'] = display_name
    results[dataset_key]['steps'] = sorted_steps
    results[dataset_key]['cells'][(step, method_key, cdm_label)] = (mean,std)
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

        # Skip CDM_basis summaries explicitly (not experimental methods)
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
            if not metric: continue
            metrics_seen.add(metric)
            if target_metric and metric != target_metric:
                continue

            step_raw = row.get("step", "").strip()
            try:
                step = int(step_raw)
            except Exception:
                step = step_raw  # keep as-is if non-integer

            mean = row.get("mean"); std = row.get("std")
            if not (is_finite_number(mean) and is_finite_number(std)):
                continue
            mean = float(mean); std = float(std)

            ds = results.setdefault(dataset_key, {"_label": dataset_label, "steps": set(), "cells": {}})
            ds["steps"].add(step)
            ds["cells"][(step, method_key, cdm_label)] = (mean, std)

    # finalize step ordering per dataset
    for dskey, d in results.items():
        try:
            d["steps"] = sorted(d["steps"], key=lambda x: (isinstance(x, str), x))
        except Exception:
            d["steps"] = sorted(list(d["steps"]))
    return results, metrics_seen

# ---------- LaTeX generation ----------
def latex_header(metric_label):
    return (
f"""\\begin{{table*}}[t]
\\centering
\\scriptsize
\\setlength{{\\tabcolsep}}{{0.5pt}}
\\renewcommand{{\\arraystretch}}{{1.0}}
\\resizebox{{\\linewidth}}{{!}}{{%
\\begin{{tabular}}{{%
  ll
  @{{\\quad}}
  c@{{\\hskip 4pt}}c
  @{{\\quad\\quad}}
  c@{{\\hskip 4pt}}c
  @{{\\quad\\quad}}
  c@{{\\hskip 4pt}}c
  @{{\\quad\\quad}}
  c@{{\\hskip 4pt}}c
  @{{\\quad\\quad}}
  c@{{\\hskip 4pt}}c
}}
\\toprule
Dataset & \\makecell{{Number of\\\\ submitted\\\\ questions ($t$)}}
  & \\multicolumn{{2}}{{c}}{{{METHOD_LABELS['naive']}}}
  & \\multicolumn{{2}}{{c}}{{{METHOD_LABELS['maml']}}}
  & \\multicolumn{{2}}{{c}}{{{METHOD_LABELS['approxgap']}}}
  & \\multicolumn{{2}}{{c}}{{{METHOD_LABELS['betacd']}}}
  & \\multicolumn{{2}}{{c}}{{{METHOD_LABELS['method']}}} \\\\
\\cmidrule(lr){{3-4}}
\\cmidrule(lr){{5-6}}
\\cmidrule(lr){{7-8}}
\\cmidrule(lr){{9-10}}
\\cmidrule(lr){{11-12}}
 &  & NCDM & CD-BPR 
  & NCDM & CD-BPR 
  & NCDM & CD-BPR 
  & NCDM & CD-BPR 
  & NCDM & CD-BPR \\\\
\\midrule
"""
    )

def latex_footer(metric_label):
    cap = f"{metric_label.upper()} results across datasets, two CDMs (NCDM and CD-BPR), and varying numbers of submitted questions ($t$). The best (according to the metric objective) mean $\\pm$ std in each row is in bold."
    return (
f"""\\bottomrule
\\end{{tabular}}
}}
\\caption{{{cap}}}
\\label{{tab:results-{metric_label}}}
\\end{{table*}}
"""
    )

def render_table_for_metric(results, metric_key, objective_for_metric):
    available_ds = list(results.keys())
    ordered_ds = [ds for ds in DATASET_ORDER if ds in results] + [ds for ds in sorted(available_ds) if ds not in DATASET_ORDER]
    out = [latex_header(metric_key)]
    dash = r"\textemdash"
    for ds in ordered_ds:
        label = results[ds]["_label"]
        steps = results[ds]["steps"]
        cells = results[ds]["cells"]
        for idx, step in enumerate(steps):
            pairs = []; means_only = []
            for m in METHOD_ORDER:
                for cdm_disp in ["NCDM", "CD-BPR"]:
                    tup = cells.get((step, m, cdm_disp))
                    if tup:
                        mean, std = tup
                        pairs.append((mean, std)); means_only.append(mean)
                    else:
                        pairs.append(None); means_only.append(None)
            bold_idx = best_indices(means_only, objective_for_metric)
            row = [f"{label} & {step}" if idx==0 else f"        & {step}"]
            for ci, p in enumerate(pairs):
                if p is None:
                    row.append(dash)
                else:
                    mean, std = p
                    cell = fmt_mean_std(mean, std)
                    if ci in bold_idx:
                        cell = r"\textbf{" + cell + "}"
                    row.append(cell)
            out.append(" & ".join(row) + r" \\")
        out.append(r"\midrule")
    if out and out[-1] == r"\midrule":
        out.pop()
    out.append(latex_footer(metric_key))
    return "\n".join(out)

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=".", help="Directory containing *_metrics_summary.csv")
    ap.add_argument("--metric", default=None, help="Single metric to output (e.g., mi_acc). If not set, outputs all.")
    ap.add_argument("--maximize", default="", help="Comma-separated metrics to maximize (overrides defaults).")
    ap.add_argument("--minimize", default="", help="Comma-separated metrics to minimize (overrides defaults).")
    ap.add_argument("--quiet", action="store_true", help="Reduce warnings/info logs.")
    args = ap.parse_args()

    max_set = parse_objective_overrides(args.maximize)
    min_set = parse_objective_overrides(args.minimize)

    results, metrics_seen = build_results(args.dir, target_metric=args.metric, verbose=not args.quiet)
    if not results:
        print("[ERROR] No usable results found.", file=sys.stderr); sys.exit(2)

    metrics_to_do = [args.metric] if args.metric else sorted(metrics_seen)
    emitted_any = False
    for m in metrics_to_do:
        filtered, _ = build_results(args.dir, target_metric=m, verbose=False)
        if not filtered:
            if not args.quiet:
                print(f"[WARN] No data for metric '{m}', skipping.", file=sys.stderr)
            continue
        obj = decide_objective(m, min_set, max_set)
        sys.stdout.write(render_table_for_metric(filtered, m, obj))
        sys.stdout.write("\n\n")
        emitted_any = True

    if not emitted_any:
        print("[ERROR] No tables emitted (no matching metrics).", file=sys.stderr)
        sys.exit(3)

if __name__ == "__main__":
    main()
