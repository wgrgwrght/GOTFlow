"""
util_functions.py

Reusable helper functions for GOTFlow analysis:
- heatmaps across states / edges
- OT gene-shift using *cached* transport plans (no Sinkhorn recomputation)
- KM plots (per-gene grid, per-edge grid)
- univariate Cox + forest plot

Assumptions:
- got is a fitted GOTFlow instance.
- got.transport_plan_[(S,T)] exists if drift_mode="barycentric" during fit.
- got.bin_ids exists and aligns with the training data X used in got.fit.
"""

from __future__ import annotations

import re
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

# lifelines (survival)
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.plotting import add_at_risk_counts
from lifelines.statistics import logrank_test



# ---------------------------------------------------------------------
# Heatmap across states (K rows) - NO re-selection if top_n=None
# ---------------------------------------------------------------------

def heatmap_top_features_by_state(
    values: pd.DataFrame,          # samples x features
    state_ids: np.ndarray,         # length n_samples
    save_path: str | None = None,
    top_n: int | None = 30,
    grouping: str = "median",
    title: str = "",
    row_cluster: bool = False,
    col_cluster: bool = True,
    figsize=(18, 8),
    cmap="RdBu_r",
    center: float = 0.0,
):
    """
    Aggregate per-state (median/mean), then clustermap.
    If top_n is None: keep all columns (no variance-based reselection).
    """
    df = values.copy()
    df["state"] = np.asarray(state_ids).astype(int)

    if grouping == "median":
        state_scores = df.groupby("state").median(numeric_only=True)
    else:
        state_scores = df.groupby("state").mean(numeric_only=True)

    state_scores = state_scores.sort_index()

    if top_n is not None and top_n < state_scores.shape[1]:
        top_feats = state_scores.var(axis=0).sort_values(ascending=False).head(int(top_n)).index
        heat = state_scores[top_feats]
    else:
        heat = state_scores

    g = sns.clustermap(
        heat,
        cmap=cmap,
        center=center,
        col_cluster=bool(col_cluster),
        row_cluster=bool(row_cluster),
        linewidths=0.2,
        figsize=figsize,
    )
    plt.suptitle(title, y=1.02)
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.show()
    return heat


# ---------------------------------------------------------------------
# OT plan + gene shift
# ---------------------------------------------------------------------

def _get_edge_list_from_states(state_ids: np.ndarray):
    uniq = np.sort(np.unique(np.asarray(state_ids).astype(int)))
    return [(int(uniq[i]), int(uniq[i+1])) for i in range(len(uniq)-1)]

def edge_gene_deltas_from_cached_plan(
    got,
    X_gene: np.ndarray,   # (N, G) aligned to got.bin_ids order
    S: int,
    T: int,
):
    """
    Use cached OT plan got.transport_plan_[(S,T)] to compute:
      ybar_gene = (P @ X_gene[tgt_idx]) / row_mass
      delta_gene = ybar_gene - X_gene[src_idx]
    Returns: src_idx, ybar_gene, delta_gene, mass
    """
    if not hasattr(got, "transport_plan_") or got.transport_plan_ is None:
        raise ValueError("got.transport_plan_ missing. Fit with drift_mode='barycentric'.")

    key = (int(S), int(T))
    if key not in got.transport_plan_:
        raise KeyError(f"Missing cached plan for edge {key}. Ensure GOTFlow.fit used drift_mode='barycentric'.")

    if got.bin_ids is None:
        raise ValueError("got.bin_ids missing.")

    P = np.asarray(got.transport_plan_[key], float)  # (n_src, n_tgt)
    src_idx = np.where(np.asarray(got.bin_ids) == int(S))[0]
    tgt_idx = np.where(np.asarray(got.bin_ids) == int(T))[0]

    if src_idx.size == 0 or tgt_idx.size == 0:
        raise ValueError(f"Empty edge {key}: n_src={src_idx.size}, n_tgt={tgt_idx.size}")

    if P.shape != (src_idx.size, tgt_idx.size):
        raise ValueError(
            f"Cached plan shape {P.shape} does not match indices "
            f"(n_src={src_idx.size}, n_tgt={tgt_idx.size}) for edge {key}."
        )

    mass = P.sum(axis=1) + 1e-12
    ybar_gene = (P @ X_gene[tgt_idx]) / mass[:, None]
    delta_gene = ybar_gene - X_gene[src_idx]
    return src_idx, ybar_gene, delta_gene, mass


def edge_shift_heatmap(
    got,
    X_gene: np.ndarray,
    gene_cols: list[str],
    state_ids: np.ndarray,
    top_genes: int = 30,
    aggregate: str = "mean",   # mean|median over source samples
    signed: bool = True,
    weight_by_mass: bool = True,
    save_path: str | None = None,
    title: str = "Per-edge gene shift",
    cmap: str = "RdBu_r",
    center: float = 0.0,
    figsize=(18, 6),
):
    """
    Heatmap with rows = edges (K-1) and cols = genes.
    Uses cached got.transport_plan_ .
    Returns: (heat, edge_df, top_list)
    """
    edges = _get_edge_list_from_states(state_ids)
    if not edges:
        raise ValueError("Need at least 2 states to form edges.")

    edge_rows = []
    edge_names = []

    for (S, T) in edges:
        _, _, delta_gene, mass = edge_gene_deltas_from_cached_plan(got, X_gene, S, T)
        D = delta_gene if signed else np.abs(delta_gene)

        if weight_by_mass and mass is not None:
            w = mass / (mass.sum() + 1e-12)
            agg_vec = (w[:, None] * D).sum(axis=0)
        else:
            agg_vec = np.median(D, axis=0) if aggregate == "median" else D.mean(axis=0)

        edge_rows.append(agg_vec)
        edge_names.append(f"{S}→{T} (n={D.shape[0]})")

    edge_df = pd.DataFrame(np.vstack(edge_rows), index=edge_names, columns=gene_cols)

    score = edge_df.abs().mean(axis=0)
    top_list = score.sort_values(ascending=False).head(int(top_genes)).index.tolist()
    heat = edge_df[top_list]

    g = sns.clustermap(
        heat,
        cmap=cmap,
        center=center,
        linewidths=0.2,
        linecolor="lightgray",
        col_cluster=True,
        row_cluster=False,
        figsize=figsize,
        cbar_kws={"label": "mean Δ gene (edge)"},
    )
    plt.suptitle(title, y=1.02)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

    return heat, edge_df, top_list


# ---------------------------------------------------------------------
# KM plots
# ---------------------------------------------------------------------

def plot_km_grid_by_gene(
    df: pd.DataFrame,
    genes_set: list[str],
    time_col: str,
    event_col: str,
    split: str = "median",
    q: float = 0.5,
    censor_time: float | None = 3652.0,
    min_group_n: int = 20,
    ncols: int = 3,
    figsize_per_subplot=(4.6, 3.6),
    pvalue_fmt: str = ".2e",
    suptitle: str | None = None,
    save_path: str | None = None,
    show_at_risk: bool = True,
    low_color: str = "blue",
    high_color: str = "red",
):
    missing = [g for g in genes_set if g not in df.columns]
    if missing:
        raise ValueError(f"Missing genes in df columns: {missing}")

    d = df.copy()
    d[time_col] = pd.to_numeric(d[time_col], errors="coerce")
    d[event_col] = pd.to_numeric(d[event_col], errors="coerce")
    d = d.dropna(subset=[time_col, event_col])
    d[event_col] = d[event_col].astype(int)

    if censor_time is not None:
        ct = float(censor_time)
        t0 = d[time_col].to_numpy(float)
        e0 = d[event_col].to_numpy(int)
        d[time_col] = np.minimum(t0, ct)
        d[event_col] = ((e0 == 1) & (t0 <= ct)).astype(int)

    n = len(genes_set)
    ncols = max(1, int(ncols))
    nrows = int(np.ceil(n / ncols))

    fig_w = figsize_per_subplot[0] * ncols
    fig_h = figsize_per_subplot[1] * nrows
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(fig_w, fig_h))
    axes = np.array(axes).reshape(-1)

    results = []
    for i, gene in enumerate(genes_set):
        ax = axes[i]
        x = pd.to_numeric(d[gene], errors="coerce")
        mask = x.notna()

        if mask.sum() < 2 * min_group_n:
            ax.set_title(f"{gene}  p=NA")
            ax.axis("off")
            results.append({"gene": gene, "p_value": np.nan})
            continue

        xg = x[mask]
        tg = d.loc[mask, time_col].to_numpy()
        eg = d.loc[mask, event_col].to_numpy()

        thr = float(np.nanmedian(xg.to_numpy())) if split == "median" else float(np.nanquantile(xg.to_numpy(), q))
        low = (xg < thr).to_numpy()
        high = ~low

        if low.sum() < min_group_n or high.sum() < min_group_n:
            ax.set_title(f"{gene}  p=NA")
            ax.axis("off")
            results.append({"gene": gene, "p_value": np.nan})
            continue

        lr = logrank_test(tg[low], tg[high], event_observed_A=eg[low], event_observed_B=eg[high])
        pval = float(lr.p_value)

        km_low = KaplanMeierFitter()
        km_high = KaplanMeierFitter()
        km_low.fit(tg[low], eg[low], label=f"Low (<{thr:.3g})")
        km_high.fit(tg[high], eg[high], label=f"High (≥{thr:.3g})")

        km_low.plot(ax=ax, ci_show=False, linewidth=2.0, color=low_color)
        km_high.plot(ax=ax, ci_show=False, linewidth=2.0, linestyle="--", color=high_color)

        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.25)
        ax.set_title(f"{gene}  p={format(pval, pvalue_fmt)}")
        xlabel = time_col if censor_time is None else f"{time_col} (censored at {int(censor_time)})"
        ax.set_xlabel(xlabel)
        ax.set_ylabel("S(t)")
        ax.legend(fontsize=8, frameon=True)

        if show_at_risk:
            add_at_risk_counts(km_low, km_high, ax=ax)

        results.append({"gene": gene, "p_value": pval, "threshold": thr, "n_low": int(low.sum()), "n_high": int(high.sum())})

    for j in range(n, axes.size):
        axes[j].axis("off")

    if suptitle:
        fig.suptitle(suptitle, y=1.02)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.show()
    return fig, axes[:n], pd.DataFrame(results).sort_values("p_value")


def plot_km_subplots_per_edge(
    df: pd.DataFrame,
    state_ids: np.ndarray,            # aligned row-for-row with df
    time_col: str,
    event_col: str,
    censor_time: float | None = 3652.0,
    ncols: int = 2,
    figsize_per_subplot=(5.2, 4.2),
    suptitle: str | None = None,
    pvalue_fmt: str = ".2e",
    min_group_n: int = 25,
    low_color: str = "blue",
    high_color: str = "red",
    save_path: str | None = None,
):
    state_ids = np.asarray(state_ids).reshape(-1)
    if state_ids.shape[0] != df.shape[0]:
        raise ValueError("state_ids must align row-for-row with df.")

    t = pd.to_numeric(df[time_col], errors="coerce")
    e = pd.to_numeric(df[event_col], errors="coerce")
    keep = t.notna() & e.notna()

    d = df.loc[keep].copy()
    d[time_col] = pd.to_numeric(d[time_col], errors="coerce")
    d[event_col] = pd.to_numeric(d[event_col], errors="coerce").astype(int)
    state_ids_f = state_ids[keep.to_numpy()]

    if censor_time is not None:
        ct = float(censor_time)
        t0 = d[time_col].to_numpy(float)
        e0 = d[event_col].to_numpy(int)
        d[time_col] = np.minimum(t0, ct)
        d[event_col] = ((e0 == 1) & (t0 <= ct)).astype(int)

    edges = _get_edge_list_from_states(state_ids_f)
    n_edges = len(edges)

    ncols = max(1, int(ncols))
    nrows = int(np.ceil(n_edges / ncols))
    fig_w = figsize_per_subplot[0] * ncols
    fig_h = figsize_per_subplot[1] * nrows
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(fig_w, fig_h))
    axes = np.array(axes).reshape(-1)

    rows = []
    for i, (S, T) in enumerate(edges):
        ax = axes[i]
        mask_S = state_ids_f == S
        mask_T = state_ids_f == T
        nS, nT = int(mask_S.sum()), int(mask_T.sum())

        if nS < min_group_n or nT < min_group_n:
            ax.set_title(f"Edge {S}→{T}\n(log-rank p=NA, nS={nS}, nT={nT})")
            ax.axis("off")
            rows.append({"edge": f"{S}->{T}", "n_S": nS, "n_T": nT, "p_value": np.nan})
            continue

        tS = d.loc[mask_S, time_col].to_numpy()
        eS = d.loc[mask_S, event_col].to_numpy()
        tT = d.loc[mask_T, time_col].to_numpy()
        eT = d.loc[mask_T, event_col].to_numpy()

        lr = logrank_test(tS, tT, event_observed_A=eS, event_observed_B=eT)
        pval = float(lr.p_value)

        kmS = KaplanMeierFitter()
        kmT = KaplanMeierFitter()
        kmS.fit(tS, eS, label=f"State {S} (n={nS})")
        kmT.fit(tT, eT, label=f"State {T} (n={nT})")

        kmS.plot(ax=ax, ci_show=False, linewidth=2.0, color=low_color)
        kmT.plot(ax=ax, ci_show=False, linewidth=2.0, linestyle="--", color=high_color)

        ax.set_title(f"Edge {S}→{T}\n(log-rank p={format(pval, pvalue_fmt)}, nS={nS}, nT={nT})")
        xlabel = time_col if censor_time is None else f"{time_col} (censored at {float(censor_time):.0f} days)"
        ax.set_xlabel(xlabel)
        ax.set_ylabel("S(t)")
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, frameon=True)
        add_at_risk_counts(kmS, kmT, ax=ax)

        rows.append({"edge": f"{S}->{T}", "n_S": nS, "n_T": nT, "p_value": pval})

    for j in range(n_edges, axes.size):
        axes[j].axis("off")

    if suptitle:
        fig.suptitle(suptitle, y=1.02)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.show()
    return fig, axes[:n_edges], pd.DataFrame(rows).sort_values("p_value")


# ---------------------------------------------------------------------
# Cox univariate + forest
# ---------------------------------------------------------------------

def _bh_fdr(pvals: np.ndarray) -> np.ndarray:
    pvals = np.asarray(pvals, float)
    n = pvals.size
    order = np.argsort(pvals)
    ranked = pvals[order]
    q = ranked * n / (np.arange(1, n + 1))
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0, 1)
    out = np.empty_like(q)
    out[order] = q
    return out


def cox_univariate_top_genes(
    df: pd.DataFrame,
    genes: list[str],
    time_col: str,
    event_col: str,
    censor_time: float | None = 3652.0,
    standardize: bool = True,
    min_n: int = 25,
):
    d = df.copy()
    d[time_col] = pd.to_numeric(d[time_col], errors="coerce")
    d[event_col] = pd.to_numeric(d[event_col], errors="coerce")
    d = d.dropna(subset=[time_col, event_col])
    d[event_col] = d[event_col].astype(int)

    if censor_time is not None:
        ct = float(censor_time)
        t0 = d[time_col].to_numpy(float)
        e0 = d[event_col].to_numpy(int)
        d[time_col] = np.minimum(t0, ct)
        d[event_col] = ((e0 == 1) & (t0 <= ct)).astype(int)

    rows = []
    for g in genes:
        if g not in d.columns:
            rows.append({"gene": g, "status": "missing_column"})
            continue
        x = pd.to_numeric(d[g], errors="coerce")
        dd = d[[time_col, event_col]].copy()
        dd[g] = x
        dd = dd.dropna(subset=[g])
        if dd.shape[0] < min_n:
            rows.append({"gene": g, "status": f"too_few_samples(n={dd.shape[0]})"})
            continue

        if standardize:
            mu = dd[g].mean()
            sd = dd[g].std(ddof=0)
            if sd == 0 or not np.isfinite(sd):
                rows.append({"gene": g, "status": "zero_variance"})
                continue
            dd[g] = (dd[g] - mu) / sd

        cph = CoxPHFitter()
        try:
            cph.fit(dd, duration_col=time_col, event_col=event_col, robust=True)
            s = cph.summary.loc[g]
            rows.append({
                "gene": g,
                "n": int(dd.shape[0]),
                "HR": float(np.exp(s["coef"])),
                "coef": float(s["coef"]),
                "CI95_low": float(np.exp(s["coef lower 95%"])),
                "CI95_high": float(np.exp(s["coef upper 95%"])),
                "p": float(s["p"]),
                "status": "ok",
            })
        except Exception as e:
            rows.append({"gene": g, "n": int(dd.shape[0]), "status": f"fit_failed: {type(e).__name__}"})

    res = pd.DataFrame(rows)
    ok = res["status"].eq("ok")
    if ok.any():
        res.loc[ok, "q"] = _bh_fdr(res.loc[ok, "p"].to_numpy())
        res = res.sort_values(["status", "p"], ascending=[True, True])
    return res


def plot_cox_forest(
    cox_tbl: pd.DataFrame,
    max_genes: int = 20,
    sort_by: str = "p",     # "p"|"q"|"HR"
    only_significant: bool = False,
    q_thresh: float = 0.05,
    p_thresh: float = 0.05,
    title: str = "Univariate Cox forest",
    figsize=(9, 8),
    show_q: bool = True,
    save_path: str | None = None,
):
    df = cox_tbl.copy()
    if "status" in df.columns:
        df = df[df["status"] == "ok"].copy()
    if df.empty:
        raise ValueError("No rows with status=='ok' to plot.")

    if only_significant:
        if show_q and "q" in df.columns:
            df = df[df["q"] <= q_thresh].copy()
        else:
            df = df[df["p"] <= p_thresh].copy()
    if df.empty:
        raise ValueError("No genes remain after significance filtering.")

    if sort_by == "q" and "q" in df.columns:
        df = df.sort_values("q")
    elif sort_by == "HR":
        df = df.sort_values("HR", ascending=False)
    else:
        df = df.sort_values("p")

    df = df.head(int(max_genes)).iloc[::-1].reset_index(drop=True)
    y = np.arange(df.shape[0])

    hr = df["HR"].to_numpy(float)
    lo = df["CI95_low"].to_numpy(float)
    hi = df["CI95_high"].to_numpy(float)
    xerr = np.vstack([hr - lo, hi - hr])

    fig, ax = plt.subplots(figsize=figsize)
    ax.errorbar(hr, y, xerr=xerr, fmt="o", capsize=3, elinewidth=1.5, markersize=5)
    ax.axvline(1.0, linestyle="--", linewidth=1)
    ax.grid(axis="x", alpha=0.25)
    ax.set_xlabel("Hazard Ratio (HR) with 95% CI")
    ax.set_title(title)

    labels = []
    for _, r in df.iterrows():
        if show_q and "q" in df.columns and pd.notna(r.get("q", np.nan)):
            labels.append(f"{r['gene']} (p={r['p']:.2e}, q={r['q']:.2e})")
        else:
            labels.append(f"{r['gene']} (p={r['p']:.2e})")

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)

    xmin = np.nanmin(lo)
    xmax = np.nanmax(hi)
    if np.isfinite(xmin) and np.isfinite(xmax):
        pad = 0.05 * (xmax - xmin + 1e-12)
        ax.set_xlim(max(0, xmin - pad), xmax + pad)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.show()
    return fig, ax
