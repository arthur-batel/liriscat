from collections import defaultdict
from typing import List

import pandas as pd
import random
import numpy as np
import torch

import seaborn as sns

from sklearn.model_selection import train_test_split
from tqdm import tqdm

import sys
sys.path.append("../../")

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from IMPACT import utils as utils_IMPACT
utils_IMPACT.set_seed(0)

import gc
import json
import matplotlib.pyplot as plt

from torch.utils.data import TensorDataset, DataLoader

cat_absolute_path = os.path.abspath('../../')

from micat import utils as utils_micat

def ecdf_percentiles(X, d, value_per_dim):
    """
    For each dimension, computes the percentile of a given value within that dimension's distribution.
    X: np.ndarray of shape (n_users, n_dim)
    value_per_dim: array-like, shape (n_dim,), value[i] = value in dimension i
    Returns: percentiles, array of shape (n_dim,), between 0 and 1
    """
    percentiles = []
    n_users, n_dim = X.shape
    
    values = X[:, d]
    v = value_per_dim
    # Empirical CDF: proportion of values <= v
    return (((values <= v).sum() / len(values))*100).floor().int().item()



        
import numpy as np
import torch
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.transforms import blended_transform_factory as _bt

import numpy as np
import torch
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.transforms import blended_transform_factory as _bt

def plot_embedding_distribution(
    i, j, t_np, test_emb, bins=50, xlim=None, ylim=None,
    test_data=None, train_emb=None, q_ids=None, u_ids=None, l_ids=None
):
    skills = {
        1: "Property of inequality", 2: "Methods of data sampling",
        3: "Geometric progression", 4: "Function versus equation",
        5: "Solving triangle", 6: "Principles of data analysis",
        7: "Classical probability theory", 8: "Linear programming",
        9: "Definitions of algorithm", 10: "Algorithm logic",
        11: "Arithmetic progression", 12: "Spatial imagination",
        13: "Abstract summarization", 14: "Reasoning and demonstration",
        15: "Calculation", 16: "Data handling"
    }

    # --- student selection (as in your code)
    test_users = torch.tensor(list(test_data.users_id))
    u_test = (-(test_emb[:, :, torch.tensor([i, j])][:, test_users, :]).std(axis=0).sum(dim=1)).argsort()
    users = test_users[u_test][:10]

    v = test_data.df[test_data.df['user_id'].isin(users.numpy())]
    w = v[test_data.df['dimension_id'] == 0]
    xg = w.groupby('user_id')['correct'].mean()
    u = int(xg.index[xg.argmax()])

    # Grades
    gi_mean = test_data.df[(test_data.df['user_id'] == u) & (test_data.df['dimension_id'] == i)]['correct'].mean()
    gj_mean = test_data.df[(test_data.df['user_id'] == u) & (test_data.df['dimension_id'] == j)]['correct'].mean()
    try: g_i = int((gi_mean - 1) * 5)
    except: g_i = 0
    try: g_j = int((gj_mean - 1) * 5)
    except: g_j = 0
    print(f"Student {u} : Category {i}: {g_i}/100, Category {j}: {g_j}/100")

    

    # --- background KDE
    x = t_np[:, i]
    y = t_np[:, j]
    g = sns.jointplot(x=x, y=y, kind="kde", fill=False, color="black", levels=8)
    ax = g.ax_joint

    # trajectory + label
    
    ax.scatter(test_emb[:, u, i], test_emb[:, u, j], s=15, color='tab:red')

    if (u_ids is not None) and (l_ids is not None):
        print(l_ids[u_ids == u])
    if (u_ids is not None) and (q_ids is not None):
        print(q_ids[u_ids == u])
        
    ax.plot(test_emb[:, u, i], test_emb[:, u, j], color='tab:red',
            label=f"Student {u}. Grades: cat. {i}, {g_i}/5; cat. {j}, {g_j}/5")

    # Honor provided limits, then read authoritative limits from the joint axes
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()

    # Freeze autoscaling so later artists don't shift limits
    ax.set_autoscale_on(False)

    # --- transforms for marginals (data on variable axis, axes on density axis)
    bx = _bt(g.ax_marg_x.transData, g.ax_marg_x.transAxes)  # (x=data, y=axes)
    by = _bt(g.ax_marg_y.transAxes, g.ax_marg_y.transData)  # (x=axes, y=data)  


    # --- per-step guides (DRAWN IN DATA COORDS ON THE JOINT AXES)
    # vertical segment: from (px, py) up to top y1
    # horizontal segment: from (px, py) right to x1
    for idx in range(16):
        px = float(test_emb[idx, u, i])
        py = float(test_emb[idx, u, j])

        # vertical from the exact (px, py) to the top
        ax.plot([px, px], [py, y1], linestyle=':',
                color=('red' if idx == 5 else 'black'), alpha=0.75, zorder=1)

        # horizontal only for idx 0 and 15 (per your logic)
        if idx in (0, 15):
            ax.plot([px, x1], [py, py], linestyle=':',
                    color='black', alpha=0.75, zorder=1)

        # x-marginal tick (axes y-coords so it can go above 1.0 if you like)
        g.ax_marg_x.plot([px, px], [0.0, 0.8], transform=bx, linestyle=':',
                         color=('red' if idx == 5 else 'black'), alpha=0.75, clip_on=False)

        # step index above x-marginal
        g.ax_marg_x.text(px, 0.8, str(idx + 1), transform=bx,
                         color=('red' if idx == 5 else 'black'), fontsize=9.5,
                         ha='center', va='bottom',
                         bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.1'))

        # percentile label every 3 steps (x-marginal)
        try:
            perf_x = ecdf_percentiles(train_emb, i, px)
        except Exception:
            perf_x = None
        if (idx % 3 == 0) and (perf_x is not None):
            g.ax_marg_x.text(px, 0.55, f"{perf_x}%", transform=bx,
                             color=('red' if idx == 8 else 'black'), fontsize=9.5,
                             ha='center', va='top',
                             bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.15'))

        # y-marginal ticks/labels at idx 0 and 15
        if idx in (0, 15):
            g.ax_marg_y.plot([0.0, 1.0], [py, py], transform=by, linestyle=':',
                             color='black', alpha=0.75, clip_on=False)
            g.ax_marg_y.text(1.0, py, str(idx + 1), transform=by, color='black', fontsize=11,
                             ha='left', va='center',bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.15'))
            try:
                perf_y = ecdf_percentiles(train_emb, j, py)
                g.ax_marg_y.text(0.70, py, f"{perf_y}%", transform=by, color='black', fontsize=11,
                                 ha='right', va='center',
                                 bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.15'))
            except Exception:
                pass

    

    # --- global averages (data coords for joint; blended for marginals)
    px_avg = float(np.nanmean(t_np[:, i]))
    py_avg = float(np.nanmean(t_np[:, j]))
    ax.plot([px_avg, test_emb[0, u, i]], [py_avg, test_emb[0, u, j]], color="black", zorder=1)
    ax.scatter(px_avg, py_avg, s=15, color="black", zorder=2)

    # vertical from (px_avg, py_avg) to top; horizontal from (px_avg, py_avg) to right
    ax.plot([px_avg, px_avg], [py_avg, y1], linestyle=':', color='black', alpha=0.75, zorder=1)

    # x-marginal vertical guide up to 1.30 (above the panel)
    g.ax_marg_x.plot([px_avg, px_avg], [0.0, 1.1], transform=bx, linestyle=':',
                     color='black', alpha=0.75, clip_on=False)
    g.ax_marg_x.text(px_avg, 1.1, "Init (t=0): \nAvg train value", transform=bx,
                     ha='center', va='bottom', fontsize=11,
                     bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.15'))

    # static marginal hints
    g.ax_marg_x.text(1.10, 0.85, "Number of\nsubmitted questions",
                     transform=g.ax_marg_x.transAxes, ha='left', va='center', fontsize=12,
                     bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.15'))
    g.ax_marg_x.text(1.10, 0.35, "> x% of the\ntraining students",
                     transform=g.ax_marg_x.transAxes, ha='left', va='center', fontsize=12,
                     bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.15'))

    # labels
    cat_i = skills[i + 1]
    cat_j = skills[j + 1]
    g.set_axis_labels(f'Profile dimension {i}:\nmath category "{cat_i}"',
                      f'Profile dimension {j}:\nmath category "{cat_j}"', fontsize=13)

    plt.tight_layout()
    plt.legend(framealpha=1, fontsize=12)
    plt.savefig("../data/students_distrib.pdf", bbox_inches="tight")
    plt.show()


    
import numpy as np, seaborn as sns, matplotlib.pyplot as plt, torch
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable

_cmap_rg = LinearSegmentedColormap.from_list("RedGreen", ["#d62728","#2ca02c"])

import numpy as np
import torch
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib.cm import ScalarMappable

import numpy as np
import torch
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
import numpy as np
import torch
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.transforms import blended_transform_factory as _bt
from sklearn.neighbors import KernelDensity

def plot_embedding_distribution_flow(
    i, j, t_np, test_emb, bins=50, xlim=None, ylim=None, test_data=None, train_emb=None,
    step_to_show=15,flow_anchor="end",flow_bins=30,min_count=3, stream_density=1.2,
    overlay_time_idx=-1, overlay_color="C2", overlay_label="test profile distrib.\n (t=15)",
    add_colorbar=True
):
    skills = {
        1: "Property of inequality", 2: "Methods of data sampling",
        3: "Geometric progression", 4: "Function versus equation",
        5: "Solving triangle", 6: "Principles of data analysis",
        7: "Classical probability theory", 8: "Linear programming",
        9: "Definitions of algorithm", 10: "Algorithm logic",
        11: "Arithmetic progression", 12: "Spatial imagination",
        13: "Abstract summarization", 14: "Reasoning and demonstration",
        15: "Calculation", 16: "Data handling"
    }

    # --- background KDE from t_np ---
    t_np = t_np.detach().cpu().numpy() if isinstance(t_np, torch.Tensor) else np.asarray(t_np)
    x_bg, y_bg = t_np[:, i], t_np[:, j]
    g = sns.jointplot(x=x_bg, y=y_bg, kind="kde", fill=False, color="black", levels=8)
    ax = g.ax_joint
    if xlim is not None: ax.set_xlim(xlim)
    if ylim is not None: ax.set_ylim(ylim)

    # freeze limits then compute borders
    x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
    ax.set_autoscale_on(False)

    # --- global averages (joint + marginals) ---
    px_avg = float(np.nanmean(t_np[:, i]))
    py_avg = float(np.nanmean(t_np[:, j]))
    ax.vlines(px_avg, py_avg, y1, linestyles=':', color='black', alpha=0.6, zorder=1)
    ax.scatter(px_avg, py_avg, s=15, color='black', zorder=2)

    bx = _bt(g.ax_marg_x.transData, g.ax_marg_x.transAxes)   # x in data, y in axes
    by = _bt(g.ax_marg_y.transAxes, g.ax_marg_y.transData)   # x in axes, y in data
    g.ax_marg_x.plot([px_avg, px_avg], [0, 1], transform=bx, linestyle=':', color='black', alpha=0.6, clip_on=False)
    g.ax_marg_x.text(px_avg, 1.06, "Init (t=0): \nAvg train value", transform=bx, ha='center', va='bottom', fontsize=11,
                     bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.2'))


    try:
        perf_x_avg = ecdf_percentiles(train_emb, i, px_avg)
        perf_y_avg = ecdf_percentiles(train_emb, j, py_avg)
        g.ax_marg_x.text(px_avg, 0.4, f"{perf_x_avg}%", transform=bx, ha='center', va='bottom', fontsize=11,
                         bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.15'))

    except:
        pass

    # Authoritative limits from the joint, then sync to marginals
    if xlim is not None:
        g.ax_joint.set_xlim(xlim)
    if ylim is not None:
        g.ax_joint.set_ylim(ylim)
    xlim = g.ax_joint.get_xlim()
    ylim = g.ax_joint.get_ylim()
    g.ax_marg_x.set_xlim(xlim)
    g.ax_marg_y.set_ylim(ylim)

    # >>> ADD THIS LINE <<<
    test_emb = test_emb.detach().cpu().numpy() if isinstance(test_emb, torch.Tensor) else np.asarray(test_emb)


    T, N, D = test_emb.shape
    Xt, Yt = test_emb[:, :, i], test_emb[:, :, j]

    dX, dY = Xt[1:] - Xt[:-1], Yt[1:] - Yt[:-1]
    X0, Y0, X1, Y1 = Xt[:-1], Yt[:-1], Xt[1:], Yt[1:]

    if flow_anchor == "start":
        Xa, Ya = X0, Y0
    elif flow_anchor == "end":
        Xa, Ya = X1, Y1
    else:
        Xa, Ya = 0.5 * (X0 + X1), 0.5 * (Y0 + Y1)

    step_ids = np.repeat(np.arange(1, T)[:, None], N, axis=1).reshape(-1)
    Xa_f = Xa.reshape(-1)
    Ya_f = Ya.reshape(-1)
    dX_f = dX.reshape(-1)
    dY_f = dY.reshape(-1)

    m = (
        np.isfinite(Xa_f) & np.isfinite(Ya_f) &
        np.isfinite(dX_f) & np.isfinite(dY_f) &
        (Xa_f >= xlim[0]) & (Xa_f <= xlim[1]) &
        (Ya_f >= ylim[0]) & (Ya_f <= ylim[1])
    )
    Xa_f, Ya_f, dX_f, dY_f, step_ids = Xa_f[m], Ya_f[m], dX_f[m], dY_f[m], step_ids[m]

    x_edges = np.linspace(xlim[0], xlim[1], int(flow_bins) + 1)
    y_edges = np.linspace(ylim[0], ylim[1], int(flow_bins) + 1)
    Xc, Yc = np.meshgrid(
        0.5 * (x_edges[:-1] + x_edges[1:]),
        0.5 * (y_edges[:-1] + y_edges[1:])
    )
    U = np.zeros_like(Xc)
    V = np.zeros_like(Yc)
    C = np.zeros_like(Xc, dtype=int)
    Tval = np.zeros_like(Xc, dtype=float)

    if Xa_f.size > 0:
        xi = np.clip(np.digitize(Xa_f, x_edges) - 1, 0, len(x_edges) - 2)
        yi = np.clip(np.digitize(Ya_f, y_edges) - 1, 0, len(y_edges) - 2)
        for k in range(Xa_f.size):
            r, c = yi[k], xi[k]
            U[r, c] += dX_f[k]
            V[r, c] += dY_f[k]
            C[r, c] += 1
            Tval[r, c] += step_ids[k]

        valid = C >= min_count
        if np.any(valid):
            U_plot = np.where(valid, U / np.maximum(C, 1), 0.0)
            V_plot = np.where(valid, V / np.maximum(C, 1), 0.0)
            T_plot = np.where(valid, Tval / np.maximum(C, 1), np.nan)

            tmin, tmax = np.nanmin(T_plot), np.nanmax(T_plot)
            if np.isfinite(tmin) and np.isfinite(tmax) and (tmax > tmin):
                norm_values = (T_plot - tmin) / (tmax - tmin)
            else:
                norm_values = np.zeros_like(T_plot)

            g.ax_joint.streamplot(
                Xc, Yc, U_plot, V_plot,
                density=stream_density,
                color=norm_values, cmap=_cmap_rg,
                linewidth=1.5, arrowsize=1.5
            )

            from mpl_toolkits.axes_grid1 import make_axes_locatable


        if add_colorbar and np.isfinite(tmin) and np.isfinite(tmax) and (tmax > tmin):
            # Create the ScalarMappable for the same normalization and cmap
            sm = ScalarMappable(Normalize(vmin=tmin, vmax=tmax), cmap=_cmap_rg)
            sm.set_array([])
        
            # --- Reserve a right gutter totally outside the JointGrid ---
            fig = g.fig  # seaborn JointGrid figure
            sp = fig.subplotpars
        
            # Shrink the grid to make room on the far right (no overlap with y-marginal)
            # 0.86 means the JointGrid uses 86% of the figure width; the remaining 14% is free.
            fig.subplots_adjust(right=0.86)
        
            # Add a dedicated colorbar axes in that free right gutter.
            # Align vertically with the JointGrid: from sp.bottom to sp.top.
            cax_left = 1.0# must be > the 'right' value above; tweak if needed
            cax_width = 0.03  # thickness of the colorbar
            cax = fig.add_axes([cax_left, sp.bottom+ (sp.top - sp.bottom)*0.05, cax_width, (sp.top - sp.bottom)*0.8])
        
            cb = fig.colorbar(sm, cax=cax)
        
            lo = int(np.ceil(tmin))
            hi = int(np.floor(tmax))
            ticks = [lo] if hi <= lo else np.linspace(lo, hi, num=min(6, hi - lo + 1), dtype=int)
            cb.set_ticks(ticks)
            cb.set_label("Number of submitted questions (t)")




    # labels + legend (original style)
    cat_i, cat_j = skills[i + 1], skills[j + 1]
    g.set_axis_labels(f'Profile dimension {i}:\n math category "{cat_i}"',
                      f'Profile dimension {j}: \nmath category "{cat_j}"', fontsize=11)

    plt.tight_layout()
    plt.legend(framealpha=1, fontsize=20)
    plt.savefig(f"../data/students_comp{i}{j}.pdf", bbox_inches="tight")
    plt.show()



def plot_embedding_distribution_flow2(
    i, j,
    t_np,
    test_emb,
    *,
    bins=50,
    xlim=None, ylim=None,
    test_data=None, train_emb=None,
    flow_bins=30, min_count=3, stream_density=1.2,
    overlay_time_idx=-1, overlay_color="C2", overlay_label="test profile distrib.\n (t=15)",
    add_colorbar=True, flow_anchor="end"
):
    import numpy as np
    import torch
    import seaborn as sns
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    from matplotlib.cm import ScalarMappable
    from matplotlib.transforms import blended_transform_factory as _bt
    from sklearn.neighbors import KernelDensity

    if isinstance(test_emb, torch.Tensor):
        test_emb = test_emb.detach().cpu().numpy()
    t_np = np.asarray(t_np)

    x_bg, y_bg = t_np[:, i], t_np[:, j]

    _cmap_rg = LinearSegmentedColormap.from_list(
        "rg_no_yellow", [(0.0, "#d73027"), (1.0, "#1a9850")]
    )

    g = sns.jointplot(x=x_bg, y=y_bg, kind="kde", fill=False, color="black", levels=8)

    # Authoritative limits from the joint, then sync to marginals
    if xlim is not None:
        g.ax_joint.set_xlim(xlim)
    if ylim is not None:
        g.ax_joint.set_ylim(ylim)
    xlim = g.ax_joint.get_xlim()
    ylim = g.ax_joint.get_ylim()
    g.ax_marg_x.set_xlim(xlim)
    g.ax_marg_y.set_ylim(ylim)

    T, N, D = test_emb.shape
    Xt, Yt = test_emb[:, :, i], test_emb[:, :, j]

    dX, dY = Xt[1:] - Xt[:-1], Yt[1:] - Yt[:-1]
    X0, Y0, X1, Y1 = Xt[:-1], Yt[:-1], Xt[1:], Yt[1:]

    if flow_anchor == "start":
        Xa, Ya = X0, Y0
    elif flow_anchor == "end":
        Xa, Ya = X1, Y1
    else:
        Xa, Ya = 0.5 * (X0 + X1), 0.5 * (Y0 + Y1)

    step_ids = np.repeat(np.arange(1, T)[:, None], N, axis=1).reshape(-1)
    Xa_f = Xa.reshape(-1)
    Ya_f = Ya.reshape(-1)
    dX_f = dX.reshape(-1)
    dY_f = dY.reshape(-1)

    m = (
        np.isfinite(Xa_f) & np.isfinite(Ya_f) &
        np.isfinite(dX_f) & np.isfinite(dY_f) &
        (Xa_f >= xlim[0]) & (Xa_f <= xlim[1]) &
        (Ya_f >= ylim[0]) & (Ya_f <= ylim[1])
    )
    Xa_f, Ya_f, dX_f, dY_f, step_ids = Xa_f[m], Ya_f[m], dX_f[m], dY_f[m], step_ids[m]

    x_edges = np.linspace(xlim[0], xlim[1], int(flow_bins) + 1)
    y_edges = np.linspace(ylim[0], ylim[1], int(flow_bins) + 1)
    Xc, Yc = np.meshgrid(
        0.5 * (x_edges[:-1] + x_edges[1:]),
        0.5 * (y_edges[:-1] + y_edges[1:])
    )
    U = np.zeros_like(Xc)
    V = np.zeros_like(Yc)
    C = np.zeros_like(Xc, dtype=int)
    Tval = np.zeros_like(Xc, dtype=float)

    if Xa_f.size > 0:
        xi = np.clip(np.digitize(Xa_f, x_edges) - 1, 0, len(x_edges) - 2)
        yi = np.clip(np.digitize(Ya_f, y_edges) - 1, 0, len(y_edges) - 2)
        for k in range(Xa_f.size):
            r, c = yi[k], xi[k]
            U[r, c] += dX_f[k]
            V[r, c] += dY_f[k]
            C[r, c] += 1
            Tval[r, c] += step_ids[k]

        valid = C >= min_count
        if np.any(valid):
            U_plot = np.where(valid, U / np.maximum(C, 1), 0.0)
            V_plot = np.where(valid, V / np.maximum(C, 1), 0.0)
            T_plot = np.where(valid, Tval / np.maximum(C, 1), np.nan)

            tmin, tmax = np.nanmin(T_plot), np.nanmax(T_plot)
            if np.isfinite(tmin) and np.isfinite(tmax) and (tmax > tmin):
                norm_values = (T_plot - tmin) / (tmax - tmin)
            else:
                norm_values = np.zeros_like(T_plot)

            g.ax_joint.streamplot(
                Xc, Yc, U_plot, V_plot,
                density=stream_density,
                color=norm_values, cmap=_cmap_rg,
                linewidth=1.5, arrowsize=1.5
            )

            if add_colorbar and np.isfinite(tmin) and np.isfinite(tmax) and (tmax > tmin):
                sm = ScalarMappable(Normalize(vmin=tmin, vmax=tmax), cmap=_cmap_rg)
                sm.set_array([])
                cb = g.figure.colorbar(sm, ax=g.ax_joint, fraction=0.046, pad=0.04)
                lo = int(np.ceil(tmin))
                hi = int(np.floor(tmax))
                ticks = [lo] if hi <= lo else np.linspace(lo, hi, num=min(6, hi - lo + 1), dtype=int)
                cb.set_ticks(ticks)
                cb.set_label("Number of submitted questions (t)")

    t_sel = max(0, min(T - 1, overlay_time_idx if overlay_time_idx >= 0 else (T + overlay_time_idx)))
    x_last = Xt[t_sel]
    y_last = Yt[t_sel]
    x_last = x_last[np.isfinite(x_last)]
    y_last = y_last[np.isfinite(y_last)]

    # Blended transforms for marginals (x=data, y=axes) and (x=axes, y=data)
    bx = _bt(g.ax_marg_x.transData, g.ax_marg_x.transAxes)
    by = _bt(g.ax_marg_y.transAxes, g.ax_marg_y.transData)

    px_avg = float(np.nanmean(t_np[:, i]))
    py_avg = float(np.nanmean(t_np[:, j]))
    g.ax_joint.scatter(px_avg, py_avg, s=15, color="black", zorder=2)
    g.ax_joint.plot([px_avg, px_avg], [py_avg, ylim[1]], linestyle=':', color='black', alpha=0.75, zorder=1)

    # Works now: data x, axes y (can extend above 1.0)
    g.ax_marg_x.plot([px_avg, px_avg], [0.0, 1.1], linestyle=':', transform=bx,
                     color='black', alpha=0.75, clip_on=False)
    g.ax_marg_x.text(px_avg, 1.1, "Init (t=0): \nAvg train value", transform=bx,
                     ha='center', va='bottom', fontsize=11,
                     bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.15'))

    def _bandwidth_scott(z):
        n = max(1, z.size)
        s = np.nanstd(z, ddof=1) if n > 1 else 1.0
        return max(1e-3 * (np.nanmax(z) - np.nanmin(z) + 1e-9), s * (n ** (-1/5)))

    skills = {
        1: "Property of inequality", 2: "Methods of data sampling",
        3: "Geometric progression", 4: "Function versus equation",
        5: "Solving triangle", 6: "Principles of data analysis",
        7: "Classical probability theory", 8: "Linear programming",
        9: "Definitions of algorithm", 10: "Algorithm logic",
        11: "Arithmetic progression", 12: "Spatial imagination",
        13: "Abstract summarization", 14: "Reasoning and demonstration",
        15: "Calculation", 16: "Data handling"
    }
    g.set_axis_labels(
        f'Profile dimension {i+1}: "{skills.get(i+1, f"Dim {i+1}")}"',
        f'Profile dimension {j+1}: "{skills.get(j+1, f"Dim {j+1}")}"',
        fontsize=11
    )

    plt.tight_layout()
    g.figure.savefig(f"../data/students_flow{i}{j}.png", dpi=400, bbox_inches="tight")
    plt.show()



import seaborn as sns, matplotlib.pyplot as plt, torch, numpy as np
from matplotlib.transforms import blended_transform_factory as _bt

import numpy as np, seaborn as sns, matplotlib.pyplot as plt, torch

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import torch
from matplotlib.transforms import blended_transform_factory as _bt

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import torch
from matplotlib.transforms import blended_transform_factory as _bt

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import torch
from matplotlib.transforms import blended_transform_factory as _bt

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import torch
from matplotlib.transforms import blended_transform_factory as _bt

def plot_embedding_distribution_comp(
    i, j, t_np, test_emb, bins=50, xlim=None, ylim=None, test_data=None, train_emb=None,
    step_to_show=15
):
    skills = {
        1: "Property of inequality", 2: "Methods of data sampling",
        3: "Geometric progression", 4: "Function versus equation",
        5: "Solving triangle", 6: "Principles of data analysis",
        7: "Classical probability theory", 8: "Linear programming",
        9: "Definitions of algorithm", 10: "Algorithm logic",
        11: "Arithmetic progression", 12: "Spatial imagination",
        13: "Abstract summarization", 14: "Reasoning and demonstration",
        15: "Calculation", 16: "Data handling"
    }

    # --- background KDE from t_np ---
    t_np = t_np.detach().cpu().numpy() if isinstance(t_np, torch.Tensor) else np.asarray(t_np)
    x_bg, y_bg = t_np[:, i], t_np[:, j]
    g = sns.jointplot(x=x_bg, y=y_bg, kind="kde", fill=False, color="black", levels=8)
    ax = g.ax_joint
    if xlim is not None: ax.set_xlim(xlim)
    if ylim is not None: ax.set_ylim(ylim)

    # --- original student selection (one per “level” on dim i) ---
    if isinstance(test_emb, torch.Tensor):
        temb_t = test_emb.detach().cpu()
    else:
        temb_t = torch.from_numpy(np.asarray(test_emb))

    test_users = torch.tensor(list(test_data.users_id))
    u_test = (-(temb_t[:, :, torch.tensor([i, j])][:, test_users, :]).std(axis=0).sum(dim=1)).argsort()
    users_sorted = test_users[u_test]  # candidate pool for replacements (in priority order)

    v = test_data.df[test_data.df['user_id'].isin(users_sorted.numpy())]
    w_i = v[test_data.df['dimension_id'] == i]
    x_i = w_i.groupby('user_id')['correct'].mean()
    w_j = v[test_data.df['dimension_id'] == j]
    x_j = w_i.groupby('user_id')['correct'].mean()  # (kept as in your original)

    # initial trio: top / middle / bottom on dim i (original logic)
    u0 = int(x_i.index[x_i.argmax()])                  # high on i
    u2 = int(x_i.index[(-x_i).argmax()])               # low on i
    u1 = int(x_i[x_i != 0].index[((x_i.mean() - x_i[x_i != 0]).abs() - (x_j.mean() - x_j).abs()).argmax()]) \
         if (x_i != 0).any() else u0                   # middle-ish

    # --- ensure the three are distinct; if not, replace duplicates from users_sorted ---
    init_trio = [u0, u1, u2]
    if len(set(init_trio)) < 3:
        unique = []
        # helper: iterate over users_sorted as replacement pool
        pool = [int(u) for u in users_sorted.numpy().tolist()]
        for idx, u in enumerate(init_trio):
            if u not in unique:
                unique.append(u)
            else:
                # find next candidate not yet used
                repl = None
                for cand in pool:
                    if cand not in unique:
                        repl = cand
                        break
                unique.append(repl if repl is not None else u)
        # if still not unique (extreme edge-case), pad from full test_users
        if len(set(unique)) < 3:
            for cand in [int(u) for u in test_users.numpy().tolist()]:
                if cand not in unique:
                    unique.append(cand)
                if len(set(unique[:3])) == 3:
                    break
        users_sel = unique[:3]
    else:
        users_sel = init_trio

    # --- plotting trajectories/points ---
    test_emb_np = temb_t.numpy()
    T, N, _ = test_emb_np.shape
    final_idx = min(step_to_show, T - 1)

    colors = ['tab:green', 'tab:blue', 'tab:red']
    for c_idx, u in enumerate(users_sel):
        xs = test_emb_np[:, u, i]
        ys = test_emb_np[:, u, j]

        gi_mean = test_data.df[(test_data.df['user_id'] == u) & (test_data.df['dimension_id'] == i)]['correct'].mean()
        gj_mean = test_data.df[(test_data.df['user_id'] == u) & (test_data.df['dimension_id'] == j)]['correct'].mean()
        try: g_i = int((gi_mean - 1) * 5)
        except: g_i = 0
        try: g_j = int((gj_mean - 1) * 5)
        except: g_j = 0

        ax.plot(xs, ys, color=colors[c_idx % 3],
                label=f"Student {u}. Grades: cat.{i}, {g_i}/5; cat.{j}, {g_j}/5")
        ax.scatter(xs, ys, s=15, color=colors[c_idx % 3], zorder=2)

    # freeze limits then compute borders
    x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
    ax.set_autoscale_on(False)

    # --- global averages (joint + marginals) ---
    px_avg = float(np.nanmean(t_np[:, i]))
    py_avg = float(np.nanmean(t_np[:, j]))
    ax.vlines(px_avg, py_avg, y1, linestyles=':', color='black', alpha=0.6, zorder=1)
    ax.hlines(py_avg, px_avg, x1, linestyles=':', color='black', alpha=0.6, zorder=1)
    ax.scatter(px_avg, py_avg, s=15, color='black', zorder=2)

    bx = _bt(g.ax_marg_x.transData, g.ax_marg_x.transAxes)   # x in data, y in axes
    by = _bt(g.ax_marg_y.transAxes, g.ax_marg_y.transData)   # x in axes, y in data
    g.ax_marg_x.plot([px_avg, px_avg], [0, 1], transform=bx, linestyle=':', color='black', alpha=0.6, clip_on=False)
    g.ax_marg_y.plot([0, 1.1], [py_avg, py_avg], transform=by, linestyle=':', color='black', alpha=0.6, clip_on=False)
    g.ax_marg_x.text(px_avg, 1.06, "Init (t=0): \nAvg train value", transform=bx, ha='center', va='bottom', fontsize=13,
                     bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.2'))
    g.ax_marg_y.text(1.1, py_avg, "Init (t=0): \nAvg train value", transform=by, ha='left', va='center', fontsize=13,
                     bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.2'))

    try:
        perf_x_avg = ecdf_percentiles(train_emb, i, px_avg)
        perf_y_avg = ecdf_percentiles(train_emb, j, py_avg)
        g.ax_marg_x.text(px_avg, 0.4, f"{perf_x_avg}%", transform=bx, ha='center', va='bottom', fontsize=11,
                         bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.15'))
        g.ax_marg_y.text(0.7, py_avg, f"{perf_y_avg}%", transform=by, ha='right', va='center', fontsize=11,
                         bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.15'))
    except:
        pass

    for c_idx, u in enumerate(users_sel):
        xs = [px_avg,test_emb_np[0, u, i]]
        ys = [py_avg,test_emb_np[0, u, j]]

        ax.plot(xs, ys, color="black", zorder=1)

    # --- guides at the exact final point for each selected student (joint + marginals) ---
    for c_idx, u in enumerate(users_sel):
        px = float(test_emb_np[final_idx, u, i])
        py = float(test_emb_np[final_idx, u, j])

        ax.hlines(py, px, x1, linestyles=':', color=colors[c_idx % 3], alpha=0.9, zorder=1)
        ax.vlines(px, py, y1, linestyles=':', color=colors[c_idx % 3], alpha=0.9, zorder=1)
        ax.scatter([px], [py], s=36, color=colors[c_idx % 3], edgecolor='white', linewidth=0.8, zorder=3)

        g.ax_marg_x.plot([px, px], [0, 1], transform=bx, linestyle=':', color=colors[c_idx % 3], alpha=0.9, clip_on=False)
        g.ax_marg_y.plot([0, 1], [py, py], transform=by, linestyle=':', color=colors[c_idx % 3], alpha=0.9, clip_on=False)
        g.ax_marg_x.text(px, 0.9, str(final_idx + 1), transform=bx, color=colors[c_idx % 3],
                         ha='center', va='center', fontsize=11,
                         bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.15'))
        g.ax_marg_y.text(0.8, py, str(final_idx + 1), transform=by, color=colors[c_idx % 3],
                         ha='left', va='center', fontsize=11,
                         bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.15'))

        try:
            perf_x = ecdf_percentiles(train_emb, i, px)
            perf_y = ecdf_percentiles(train_emb, j, py)
            g.ax_marg_x.text(px, 0.4, f"{perf_x}%", transform=bx, color=colors[c_idx % 3],
                             ha='center', va='bottom', fontsize=11,
                             bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.15'))
            g.ax_marg_y.text(0.7, py, f"{perf_y}%", transform=by, color=colors[c_idx % 3],
                             ha='right', va='center', fontsize=11,
                             bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.15'))
        except:
            pass

    # static marginal hints
    g.ax_marg_x.text(1.1, 0.85, "Number of\nsubmitted questions", transform=g.ax_marg_x.transAxes,
                     ha='left', va='center', fontsize=13,
                     bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.15'))
    g.ax_marg_x.text(1.1, 0.35, "> x% of the\ntraining students", transform=g.ax_marg_x.transAxes,
                     ha='left', va='center', fontsize=13,
                     bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.15'))

    # labels + legend (original style)
    cat_i, cat_j = skills[i + 1], skills[j + 1]
    g.set_axis_labels(f'Profile dimension {i}:\nmath category "{cat_i}"',
                      f'Profile dimension {j}:\nmath category "{cat_j}"', fontsize=14)

    plt.tight_layout()
    plt.legend(framealpha=1, fontsize=12)
    plt.savefig(f"../data/students_comp{i}{j}.pdf", bbox_inches="tight")
    plt.show()





def stat_unique(data: pd.DataFrame, key):
    """
    Calculate statistics on unique values in a DataFrame column or combination of columns.

    :param data: A pandas DataFrame.
    :type data: pd.DataFrame
    :param key: The column name or list of column names
    :type key: str or List[str]
    :return: None
    :rtype: None
    """
    if key is None:
        print('Total length: {}'.format(len(data)))
    elif isinstance(key, str):
        print('Number of unique {}: {}'.format(key, len(data[key].unique())))
    elif isinstance(key, list):
        print('Number of unique [{}]: {}'.format(','.join(key), len(data.drop_duplicates(key, keep='first'))))
        
def remove_duplicates(data: pd.DataFrame, key_attrs: List[str], agg_attrs: List[str]):
    """
    Remove duplicates from a DataFrame based on specified key attributes while aggregating other attributes.

    :param data: Dataset as a pandas.DataFrame
    :type data: pd.DataFrame
    :param key_attrs: Attributes to group by for identifying unique records
    :type key_attrs: List[str]
    :param agg_attrs: Attributes to aggregate in a set for every unique key
    :type agg_attrs: List[str]
    :return: DataFrame with duplicates removed and specified attributes aggregated
    :rtype: pd.DataFrame
    """
    special_attributes = key_attrs.copy()
    special_attributes.extend(agg_attrs)
    d = {}
    for agg_attr in agg_attrs:
        d.update({agg_attr: set})
    d.update({col: 'first' for col in data.columns if col not in special_attributes})
    data = data.groupby(key_attrs).agg(d).reset_index()
    return data

def split_data_vertically_unique_fold(quadruplet, valid_prop):
    """
    Split data (list of double) into train and validation sets.

    :param quadruplet: List of triplets (sid, qid, score)
    :type quadruplet: list
    :param valid_prop: Fraction of the validation set
    :type valid_prop: float
    :param least_test_length: Minimum number of items a student must have to be included in the test set.
    :type least_test_length: int or None
    :return: Train and validation sets.
    :rtype: list, list, list
    """
    df = pd.DataFrame(quadruplet, columns=["student_id", "item_id", "correct", "dimension_id"])
    df.columns = ["student_id", "item_id", "correct", "dimension_id"]
    df_grouped = df.groupby(['student_id', 'item_id']).agg(
        answers=('correct', list),
        dimensions=('dimension_id', list)
    ).reset_index()

    train_list = []
    valid_list = []

    for i_group, group in df_grouped.groupby('student_id'):
        train_part, valid_part = train_test_split(group, test_size=valid_prop, shuffle=True)
        train_list.append(train_part)
        valid_list.append(valid_part)

    train = pd.concat(train_list, ignore_index=True)
    valid = pd.concat(valid_list, ignore_index=True)

    train_expanded = train.explode(['answers', 'dimensions'])
    valid_expanded = valid.explode(['answers', 'dimensions'])

    return train_expanded, valid_expanded

def are_student_sets_equal(train, valid):
    i_train = train['item_id'].unique()
    i_valid= valid['item_id'].unique()
    return len(set(i_train) - (set(i_valid)))==0

def are_pair_student_equal(train, valid):
    r_train = train.apply(lambda x: str(x['student_id']) + '_' + str(x['item_id']), axis=1).unique()
    r_valid = valid.apply(lambda x: str(x['student_id']) + '_' + str(x['item_id']), axis=1).unique()
    return len(set(r_train) - (set(r_valid))) == 0

def quadruplet_format(data: pd.DataFrame):
    """
    Convert DataFrame into a list of quadruplets with correct data types.

    :param data: Dataset containing columns 'user_id', 'item_id', 'correct', and 'dimension_id'
    :type data: pd.DataFrame
    :return: List of quadruplets [sid, qid, score, dim]
    :rtype: list
    """
    # Ensure data types
    data['user_id'] = data['user_id'].astype(int)
    data['item_id'] = data['item_id'].astype(int)
    data['dimension_id'] = data['dimension_id'].astype(int)
    # 'correct' column might be float or int, depending on your data

    # Extract columns as lists
    user_ids = data['user_id'].tolist()
    item_ids = data['item_id'].tolist()
    corrects = data['correct'].tolist()
    dimension_ids = data['dimension_id'].tolist()

    # Combine columns into a list of quadruplets
    quadruplets = list(zip(user_ids, item_ids, corrects, dimension_ids))

    # Convert each quadruplet to a list
    quadruplets = [list(quad) for quad in quadruplets]

    return quadruplets


def densify(data: pd.DataFrame, grp_by_attr: str, count_attr: str, thd: int):
    """
    Filter out groups in a DataFrame based on a count threshold.

    :param data: Dataset
    :type data: pd.DataFrame
    :param grp_by_attr: Attribute used for grouping
    :type grp_by_attr: str
    :param count_attr: Attribute used for counting within groups
    :type count_attr: str
    :param thd: Threshold for group count (groups with less than thd filtered)
    :type thd: int
    :return: DataFrame with groups filtered out
    """
    n = data.groupby(grp_by_attr)[count_attr].nunique()
    filter = n[n < thd].index.tolist()
    print(f'filter {len(filter)} '+grp_by_attr)

    if len(filter) >0 :
        return data[~data[grp_by_attr].isin(filter)], len(filter)
    else : return data, 0

def create_q2k(data: pd.DataFrame):
    """
    Create mappings from item IDs to sets of dimension IDs and vice versa.

    :param data: Dataset containing 'item_id' and 'dimension_id' columns
    :type data: pd.DataFrame
    :return: item to knowledge mapping and knowledge to item mapping
    :rtype: Dict[str, Set[str]], Dict[str, Set[str]]
    """
    q2k = {}
    table = data.drop_duplicates(subset=["dimension_id","item_id"])
    for i, row in table.iterrows():
        q = int(row['item_id'])
        l = q2k.get(q,[])
        l.append(str(int(row['dimension_id'])))
        q2k[q] = l

    # get knowledge to item map
    k2q = {}
    for q, ks in q2k.items():
        for k in ks:
            k2q.setdefault(k, set())
            k2q[k].add(q)
    return q2k, k2q

def encode_attr(data: pd.DataFrame, attr:str):
    """
    Encode categorical attribute values with numerical IDs.

    :param data: Dataset
    :type data: pd.DataFrame
    :param attr: Attribute to renumber
    :type attr: str
    :return: Encoded DataFrame and mapping from attribute to numerical IDs
    :rtype: pd.DataFrame, Dict[str, int]
    """

    attr2n = {}
    cnt = 0
    for i, row in data.iterrows():
        if row[attr] not in attr2n:
            attr2n[row[attr]] = cnt
            cnt += 1

    data.loc[:, attr] = data.loc[:, attr].apply(lambda x: attr2n[x])
    return data.astype({attr:int}), attr2n


def parse_data(data):
    """
    Parse data into student-based and item-based datasets.

    :param data: List of triplets (sid, qid, score)
    :type data: pd.DataFrame
    :return: Student-based and item-based datasets
    :rtype: defaultdict(dict), defaultdict(dict)
    """

    stu_data = defaultdict(lambda: defaultdict(dict))
    ques_data = defaultdict(lambda: defaultdict(dict))
    for i, row in data.iterrows():
        sid = row.user_id
        qid = row.item_id
        correct = row.correct
        stu_data[sid][qid] = correct
        ques_data[qid][sid] = correct
    return stu_data, ques_data

def quadruplet_format(data: pd.DataFrame):
    """
    Convert DataFrame into a list of quadruplets with correct data types.

    :param data: Dataset containing columns 'user_id', 'item_id', 'correct', and 'dimension_id'
    :type data: pd.DataFrame
    :return: List of quadruplets [sid, qid, score, dim]
    :rtype: list
    """
    # Ensure data types
    data['user_id'] = data['user_id'].astype(int)
    data['item_id'] = data['item_id'].astype(int)
    data['dimension_id'] = data['dimension_id'].astype(int)
    # 'correct' column might be float or int, depending on your data

    # Extract columns as lists
    user_ids = data['user_id'].tolist()
    item_ids = data['item_id'].tolist()
    corrects = data['correct'].tolist()
    dimension_ids = data['dimension_id'].tolist()

    # Combine columns into a list of quadruplets
    quadruplets = list(zip(user_ids, item_ids, corrects, dimension_ids))

    # Convert each quadruplet to a list
    quadruplets = [list(quad) for quad in quadruplets]

    return quadruplets




def one_hot_encoding(df,response_range_dict):
    # Pre-calculate num_copies for each q_name
    df = df[df['item_id'].isin(response_range_dict.keys())]

    df['r_range'] = df['item_id'].map(response_range_dict).astype(int)

    # Initialize an empty list to store the duplicated DataFrames
    dfs = []

    # Vectorized operation to duplicate rows
    for _, row in tqdm(df.iterrows(), total=df.shape[0]):
        r_range = int(row['r_range'])
        c = int(row['correct'])

        # Create a binary list using numpy
        c_binary_list = np.zeros(r_range, dtype=int)
        c_binary_list[c - 1] = 1

        # Duplicate the row r_range times
        duplicated_rows = pd.DataFrame([row] * r_range)
        duplicated_rows['correct_binary'] = c_binary_list

        # Update item_id with the new values
        duplicated_rows['item_id'] = duplicated_rows['item_id'].astype(
            str) + '_' + duplicated_rows.index.astype(str)

        dfs.append(duplicated_rows)

    # Concatenate all the DataFrames in the list
    return pd.concat(dfs, ignore_index=True)

def rescaling_dict(metadata: pd.DataFrame,q2n):
    """
    Computes response_range_dict, min_response_dict and max_response_dict

    """
    response_range_dict = {}
    min_response_dict = {}
    max_response_dict = {}

    # Iterate over the DataFrame
    for i, row in metadata.iterrows():
        # Extract item ID, min_response, and max_response from the row
        try:
            item_id = q2n[row["Variable Name"]]
            min_response = row["min_response"]
            max_response = row["max_response"]

            # Calculate response range and store it in the dictionary
            response_range = max_response - min_response
            response_range_dict[item_id] = response_range
            min_response_dict[item_id] = min_response
            max_response_dict[item_id] = max_response
        except KeyError as e:
            print(f'{e} were removed from dataset')
    return [response_range_dict, min_response_dict, max_response_dict]

def get_modalities_nb(data, metadata) :

    tensor_data = torch.zeros((metadata['num_user_id'], metadata['num_item_id']), dtype=torch.double)
    sid = torch.from_numpy(data['user_id'].to_numpy()).long()
    qid = torch.from_numpy(data['item_id'].to_numpy()).long()
    val = torch.from_numpy(data['correct'].to_numpy())
    
    tensor_data.index_put_((sid, qid), val)

    R_t = tensor_data
    R_t = R_t.T - 1
    
    nb_modalities = torch.zeros(metadata['num_item_id'], dtype=torch.long)
    
    for item_i, logs in enumerate(R_t):
        unique_logs = torch.unique(logs)
        delta_min = torch.min(
            torch.abs(unique_logs.unsqueeze(0) - unique_logs.unsqueeze(1)) + torch.eye(unique_logs.shape[0]))
        nb_modalities[item_i] = (torch.round(1 / delta_min) + 1).long()
    return nb_modalities

def split_users(df, folds_nb=5, seed=0) :
    """
    k-fold cross validationsplit of users

    """

    users_idx = df['user_id'].unique()
    N = len(users_idx) // 5
    random.Random(seed).shuffle(users_idx)

    train = [[] for _ in range(folds_nb)]
    valid = [[] for _ in range(folds_nb)]
    test = [[] for _ in range(folds_nb)]

    for i_fold in range(folds_nb):
        test_fold, valid_fold = (i_fold - 1) % 5, i_fold

        test_users = users_idx[test_fold * N: (test_fold + 1) * N]
        valid_users = users_idx[valid_fold * N: (valid_fold + 1) * N]
        train_indices = [idx for idx in range(len(users_idx))]
        train_indices = [idx for idx in train_indices if idx //
                         N != test_fold and idx // N != valid_fold]
        train_users = [int(users_idx[idx]) for idx in train_indices]

        train[i_fold] = df[df['user_id'].isin(users_idx[train_users])]
        valid[i_fold] = df[df['user_id'].isin(users_idx[valid_users])]
        test[i_fold] = df[df['user_id'].isin(users_idx[test_users])]



    return train, valid, test

def split_data_horizontally(df):
    train = []
    valid = []

    for i_group, group in df.groupby('user_id'):
        group_idxs = group.index.values

        train_item_idx, valid_item_idx = train_test_split(group_idxs, test_size=0.2, shuffle=True)

        train.extend(group.loc[train_item_idx].values.tolist())
        valid.extend(group.loc[valid_item_idx].values.tolist())

    return train, valid

def save_df_to_csv(data, path):
    """
    Save list of triplets (sid, qid, score) to a CSV file.
    :param data: List of triplets (sid, qid, score)
    :type data: pd.DataFrame
    :param path: Path to CSV file
    :type path: str
    """
    data.to_csv(path, index=False)


def get_metadata(data: pd.DataFrame, keys: List[str]) -> dict:
    m = {}
    for attr in keys:
        m["num_"+attr] = len(data[attr].unique())
    return m

def convert_to_records(data):
    df = data.rename(columns={
        'student_id': 'user_id',
        'answers': 'correct',
        'dimensions': 'dimension_id'
    })
    return df.to_records(index=False, column_dtypes={'user_id': int, 'item_id': int, 'correct': float, 'dimension_id': int})


def load_dataset_resources(config, base_path: str = "../2-preprocessed_data/"):
    concept_map = json.load(open(f'../datasets/2-preprocessed_data/{config["dataset_name"]}_concept_map.json', 'r'))
    concept_map = {int(k): [int(x) for x in v] for k, v in concept_map.items()}
    #parameter
    metadata = json.load(open(f'../datasets/2-preprocessed_data/{config["dataset_name"]}_metadata.json', 'r'))
    nb_modalities = torch.load(f'../datasets/2-preprocessed_data/{config["dataset_name"]}_nb_modalities.pkl',weights_only=True)
    return concept_map, metadata, nb_modalities

def vertical_data(config, i_fold):

    train = pd.read_csv(
    f'../datasets/2-preprocessed_data/{config["dataset_name"]}_vert_train_{i_fold}.csv',
    encoding='utf-8', dtype={'student_id': int, 'item_id': int, "correct": float,
                                                             "dimension_id": int})
    valid= pd.read_csv(
    f'../datasets/2-preprocessed_data/{config["dataset_name"]}_vert_valid_{i_fold}.csv',
    encoding='utf-8', dtype={'student_id': int, 'item_id': int, "correct": float,
                                                             "dimension_id": int})

    return convert_to_records(train), convert_to_records(valid)


def load_dataset(config) :
    gc.collect()
    torch.cuda.empty_cache()
    # read datasets
    train_data, valid_data = utils_micat.prepare_dataset(config, i_fold=0)
    # read datasets ressources
    concept_map, metadata, nb_modalities= load_dataset_resources(config)

    return train_data,valid_data,concept_map,metadata


def tarjan_scc(adj_matrix, idx_to_genres):
    n = len(adj_matrix)
    index = [-1] * n
    lowlink = [0] * n
    on_stack = [False] * n
    stack = []
    sccs = []
    current_index = [0]  # use a list to allow modifications in closure

    def strongconnect(v):
        # Set the depth index for v
        index[v] = current_index[0]
        lowlink[v] = current_index[0]
        current_index[0] += 1
        stack.append(v)
        on_stack[v] = True

        # Consider successors of v
        for w in range(n):
            if adj_matrix[v][w] == 1:  # edge from v to w
                if index[w] == -1:
                    # Successor w has not yet been visited
                    strongconnect(w)
                    lowlink[v] = min(lowlink[v], lowlink[w])
                elif on_stack[w]:
                    # Successor w is on the stack, so it's part of the current SCC
                    lowlink[v] = min(lowlink[v], index[w])

        # If v is a root node, pop the stack and generate an SCC
        if lowlink[v] == index[v]:
            # Start a new strongly connected component
            scc = []
            while True:
                w = stack.pop()
                on_stack[w] = False
                scc.append(idx_to_genres[w])
                if w == v:
                    break
            sccs.append(scc)

    for v in range(n):
        if index[v] == -1:
            strongconnect(v)

    return sccs