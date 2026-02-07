import pandas as pd
import numpy as np
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import matplotlib
import math
import argparse
from pathlib import Path
from scipy.special import erfinv
from typing import Literal

params: dict[str, str] = {'axes.titlesize':'18',
        'xtick.labelsize':'16',
        'ytick.labelsize':'16',
        'font.size':'19',
        'legend.fontsize':'medium',
        'lines.linewidth':'2.5',
        'font.weight':'normal',
        'lines.markersize':'14',
        'text.latex.preamble': r'\usepackage{amsfonts}',
        'lines.markerfacecolor':'none'
        }
matplotlib.rcParams.update(params)
plt.rcParams["mathtext.fontset"] = "cm"
plt.rc('text', usetex=False)
plt.rc('font', family='serif')
markers: list[str] = ['s', 'd', 'X', 'o', 'v', '*', '^', '8', 'h', '+', 'P', 'X', 'D']


def plot_results_example_audit(df: pd.DataFrame, lst_protocol: list, epsilon: float, k: int) -> None:
    df_eps: pd.DataFrame = df.loc[(df.epsilon == epsilon) & (df.k == k)]
    dic_eps: dict[str, float] = df_eps.groupby('protocol')['eps_emp'].mean().to_dict()
    dic_stds: dict[str, float] = df_eps.groupby('protocol')['eps_emp'].std().to_dict()

    # seed ごとに (GRR_eps + logRmax) を作ってから平均・std を取る
    grr: pd.DataFrame = df_eps[df_eps["protocol"] == "GRR"][["seed", "eps_emp"]]
    lr: pd.DataFrame = df_eps[["seed", "logRmax_eff"]].drop_duplicates("seed")
    grr_plus: pd.DataFrame = grr.merge(lr, on="seed", how="inner")
    grr_plus["sum"] = grr_plus["eps_emp"] + grr_plus["logRmax_eff"]
    grr_plus_mean: float = float(grr_plus["sum"].mean())
    grr_plus_std: float = float(grr_plus["sum"].std(ddof=1)) if len(grr_plus) > 1 else 0.0
    # 表示順に 1 本追加
    lst_protocol = lst_protocol + ["GRR +\n$\\log R_{\\max}$"]
    dic_eps["GRR +\n$\\log R_{\\max}$"] = grr_plus_mean
    dic_stds["GRR +\n$\\log R_{\\max}$"] = grr_plus_std

    # Get empirical eps
    values: list[float] = [dic_eps[key] for key in lst_protocol] 
    stds: list[float] = [dic_stds[key] for key in lst_protocol]

    n: int = len(lst_protocol)

    # Plotting
    plt.figure(figsize=(3.8, 3.8))
    plt.grid(color='grey', linestyle='dashdot', linewidth=0.5, zorder=0)
    plt.xlim(-0.5, n - 0.5)
    plt.hlines(epsilon, -0.5, n - 0.5, label='Theoretical $\\varepsilon$', color ='red', linestyle='dashed')
    plt.bar(range(len(lst_protocol)), values, zorder=10, width=0.65)
    plt.xticks(range(len(lst_protocol)), lst_protocol, rotation = 45)
    plt.errorbar(range(len(lst_protocol)), values, yerr=stds, ecolor='black', capsize=5, zorder=50, linestyle='None')

    plt.ylabel('Estimated $\\varepsilon_{emp}$')    
    plt.xlabel('LDP Frequency Estimation Protocols')
    plt.savefig('results/fig_results_summary_audit.pdf', dpi=500, bbox_inches = 'tight',pad_inches = 0.1)
    plt.gcf().subplots_adjust(bottom=0.32)
    return plt.show()

def plot_results_pure_ldp_protocols(df: pd.DataFrame, analysis: str, lst_protocol: list, lst_eps: list, lst_k: list) -> None:
    # LRT 比較パネルを 1 枚追加したいので、描画用にだけ 1 要素増やす
    add_compare_panel: bool = "LRT" in lst_protocol
    proto_panels: list[str] = lst_protocol + (["LRT_compare"] if add_compare_panel else [])
    
    # 動的に行数を決定
    ncols: int = 2
    nrows: int = math.ceil(len(proto_panels) / ncols)
    fig, ax = plt.subplots(nrows, ncols, figsize=(12, 3.5*nrows), sharey=True)
    plt.subplots_adjust(wspace=0.25, hspace=0.5)

    # ax を常に 2 次元配列として扱う
    ax = np.atleast_2d(ax)

    c = 0 # column
    for row, protocol in enumerate(proto_panels):
        r, c = divmod(row, ncols)
        ax[r, c].yaxis.set_tick_params(which='both', labelbottom=True)
        ax[r, c].grid(color='grey', linestyle='dashdot', linewidth=0.5)
        ax[r, c].plot(lst_eps, label='Theoretical $\\varepsilon$', color ='black', linestyle='dashed')
        
        mkr_idx = 0
        for k in lst_k:
            results_k: list[float] = []
            variation_k: list[float] = []

            # --- LRT 用：lower/upper の平均系列 ---
            lrt_lo_mean: list[float] = []
            lrt_hi_mean: list[float] = []

            for epsilon in lst_eps:
                df_eps = df.loc[(df.protocol == protocol) & (df.epsilon == epsilon) & (df.k == k)]['eps_emp'].clip(0)
                # LRT_compare も中身は LRT を使って描く
                proto_for_data: str = 'LRT' if protocol == 'LRT_compare' else protocol
                df_eps: pd.Series = df.loc[(df.protocol == proto_for_data) & (df.epsilon == epsilon) & (df.k == k)]['eps_emp'].clip(0)
                results_k.append(df_eps.mean())
                variation_k.append(df_eps.std())

                # LRT パネルのときだけ lower/upper を集計
                if protocol == "LRT":
                    df_lo = df.loc[
                        (df.protocol == "LRT") & (df.epsilon == epsilon) & (df.k == k),
                        "eps_lower",
                    ].astype(float)         # type: ignore
                    df_hi = df.loc[
                        (df.protocol == "LRT") & (df.epsilon == epsilon) & (df.k == k),
                        "eps_upper",
                    ].astype(float)         # type: ignore

                    lrt_lo_mean.append(df_lo.mean())
                    lrt_hi_mean.append(df_hi.mean())
            
            std_minus = np.array(results_k) - np.array(variation_k)
            std_plus = np.array(results_k) + np.array(variation_k)        
            ax[r, c].fill_between(range(len(lst_eps)), std_minus, std_plus, alpha=0.3)
            ax[r, c].plot(results_k, label = 'k={}'.format(k), marker = markers[mkr_idx])

            # LRT のみ：lower/upper を同色で追加描画（marker/linestyle を変える）
            if protocol == "LRT":
                x_idx = np.arange(len(lst_eps), dtype=int)

                # marker は lower=▽ upper=△
                ax[r, c].plot(
                    x_idx,
                    np.array(lrt_lo_mean, dtype=float),
                    linestyle=":",
                    color=ax[r, c].lines[-1].get_color(),
                    marker="v",
                    label=f"k={k} (lower)",
                )
                ax[r, c].plot(
                    x_idx,
                    np.array(lrt_hi_mean, dtype=float),
                    linestyle=":",
                    color=ax[r, c].lines[-1].get_color(),
                    marker="^",
                    label=f"k={k} (upper)",
                )
            
            mkr_idx+=1
            
            # --- 比較パネルならオレンジ線（GRR + ln R_max）を重ねる ---
            if protocol == 'LRT_compare':
                df_grr_all: pd.DataFrame = df[df['protocol'] == 'GRR'][['seed','epsilon','eps_emp']]
                lr_all: pd.DataFrame     = df[df['protocol']=='LRT'][['seed','logRmax_eff']].drop_duplicates('seed')
                y_mean, y_std = [], []
                for epsilon in lst_eps:
                    grr_e  = df_grr_all[df_grr_all['epsilon'] == epsilon][['seed','eps_emp']]
                    merged = grr_e.merge(lr_all, on='seed', how='inner')
                    if merged.empty:
                            y_mean.append(np.nan); y_std.append(np.nan)
                    else:
                        s = (merged['eps_emp'] + merged['logRmax_eff']).to_numpy()
                        y_mean.append(float(np.mean(s)))
                        y_std.append(float(np.std(s, ddof=1)) if s.size > 1 else 0.0)

                x_idx = np.arange(len(lst_eps))
                y_mean = np.array(y_mean, dtype=float)
                y_std  = np.array(y_std, dtype=float)
                ax[r, c].plot(x_idx, y_mean, color='tab:orange', marker='o',
                            label='RR +\n$\\log R_{\\max}$')
                ax[r, c].fill_between(x_idx, y_mean - y_std, y_mean + y_std,
                                    color='tab:orange', alpha=0.2)
            
            # --- LRT_tilde1 があり，比較パネルなら緑線（LRT_tilde1: tilde y=1 only）を重ねる ---
            if protocol == 'LRT_compare' and ('LRT_tilde1' in lst_protocol):
                # eps_emp mean±std
                y_mean, y_std = [], []
                y_lo, y_hi = [], []

                for epsilon in lst_eps:
                    d_emp: pd.Series = df.loc[
                        (df["protocol"] == "LRT_tilde1") & (df["epsilon"] == epsilon) & (df["k"] == k),
                        "eps_emp",
                    ].astype(float)  # type: ignore
                    d_lo = df.loc[
                        (df["protocol"] == "LRT_tilde1") & (df["epsilon"] == epsilon) & (df["k"] == k),
                        "eps_lower",
                    ].astype(float)  # type: ignore
                    d_hi = df.loc[
                        (df["protocol"] == "LRT_tilde1") & (df["epsilon"] == epsilon) & (df["k"] == k),
                        "eps_upper",
                    ].astype(float)  # type: ignore

                    y_mean.append(float(d_emp.mean()) if not d_emp.empty else np.nan)
                    y_std.append(float(d_emp.std(ddof=1)) if d_emp.size > 1 else 0.0)
                    y_lo.append(float(d_lo.mean()) if not d_lo.empty else np.nan)
                    y_hi.append(float(d_hi.mean()) if not d_hi.empty else np.nan)

                x_idx = np.arange(len(lst_eps))
                y_mean = np.array(y_mean, dtype=float)
                y_std  = np.array(y_std, dtype=float)
                y_lo   = np.array(y_lo, dtype=float)
                y_hi   = np.array(y_hi, dtype=float)

                # 緑で描く（
                ax[r, c].plot(
                    x_idx, y_mean, color="green", marker="o",
                    label="LRT ($\\tilde y=1$ only)"
                )
                ax[r, c].fill_between(
                    x_idx, y_mean - y_std, y_mean + y_std,
                    color="green", alpha=0.15
                )
                # lower/upper（同じ緑）
                ax[r, c].plot(
                    x_idx, y_lo, linestyle=":", color="green", marker="v",
                    label="LRT ($\\tilde y=1$ only, lower)"
                )
                ax[r, c].plot(
                    x_idx, y_hi, linestyle=":", color="green", marker="^",
                    label="LRT ($\\tilde y=1$ only, upper)"
                )

            # --- 比較パネルなら紫/ピンク線（LRT_indirect）を重ねる ---
            if protocol == "LRT_compare" and ("LRT_indirect" in lst_protocol):
                # eps_emp mean±std, eps_lower/upper mean
                y_mean, y_std = [], []
                y_lo, y_hi = [], []

                for epsilon in lst_eps:
                    d_emp: pd.Series = df.loc[
                        (df["protocol"] == "LRT_indirect")
                        & (df["epsilon"] == epsilon)
                        & (df["k"] == k),
                        "eps_emp",
                    ].astype(float)  # type: ignore
                    d_lo: pd.Series = df.loc[
                        (df["protocol"] == "LRT_indirect")
                        & (df["epsilon"] == epsilon)
                        & (df["k"] == k),
                        "eps_lower",
                    ].astype(float)  # type: ignore
                    d_hi: pd.Series = df.loc[
                        (df["protocol"] == "LRT_indirect")
                        & (df["epsilon"] == epsilon)
                        & (df["k"] == k),
                        "eps_upper",
                    ].astype(float)  # type: ignore

                    y_mean.append(float(d_emp.mean()) if not d_emp.empty else np.nan)
                    y_std.append(float(d_emp.std(ddof=1)) if d_emp.size > 1 else 0.0)
                    y_lo.append(float(d_lo.mean()) if not d_lo.empty else np.nan)
                    y_hi.append(float(d_hi.mean()) if not d_hi.empty else np.nan)

                x_idx = np.arange(len(lst_eps))
                y_mean = np.array(y_mean, dtype=float)
                y_std = np.array(y_std, dtype=float)
                y_lo = np.array(y_lo, dtype=float)
                y_hi = np.array(y_hi, dtype=float)

                # 色は紫（tab:purple）or ピンク（tab:pink）
                base_color = "tab:purple"

                # eps_emp（点線）
                ax[r, c].plot(
                    x_idx,
                    y_mean,
                    color=base_color,
                    marker="o",
                    linestyle=":",
                    label="LRT (indirect, emp)",
                )
                ax[r, c].fill_between(
                    x_idx, y_mean - y_std, y_mean + y_std, color=base_color, alpha=0.12
                )

                # eps_lower（実線・太め）
                ax[r, c].plot(
                    x_idx,
                    y_lo,
                    color=base_color,
                    linestyle="-",
                    linewidth=2.5,
                    marker="v",
                    label="LRT (indirect, lower)",
                )

                # eps_upper（点線）
                ax[r, c].plot(
                    x_idx,
                    y_hi,
                    color=base_color,
                    linestyle=":",
                    marker="^",
                    label="LRT (indirect, upper)",
                )

        ax[r, c].set_yscale('log')
        ax[r, c].set_xticks(range(len(lst_eps)))
        ax[r, c].set_xticklabels(lst_eps)
        ax[r, c].set_title(protocol, fontsize=20)
        ax[r, c].set_ylabel('Estimated $\\varepsilon_{emp}$')
        ax[r, c].set_xlabel('Theoretical $\\varepsilon$')

    # 余ったパネルを非表示
    total_panels: int = nrows * ncols
    for j in range(len(proto_panels), total_panels):
        r, c = divmod(j, ncols)
        ax[r, c].set_visible(False)

    # 反例
    # ax[0, 0].legend(columnspacing=0.8, ncol=8, loc='upper center', bbox_to_anchor=(1.05, 1.5))
    # 全パネルの凡例項目を集約して上部に表示
    handles, labels = [], []
    for axes in ax.flat:
        h, l = axes.get_legend_handles_labels()
        for hh, ll in zip(h, l):
            if ll not in labels:  # 重複除去
                handles.append(hh); labels.append(ll)
    # fig.legend(handles, labels, columnspacing=0.8, ncol=8,
    #         loc='upper center', bbox_to_anchor=(0.5, 1.01))
    # fig.legend(handles, labels, loc="lower right")
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.02),
            ncol=min(8, len(labels)), columnspacing=0.8)
    plt.savefig('results/fig_results_'+analysis+'.pdf', dpi=500, bbox_inches = 'tight',pad_inches = 0.1)

    return plt.show()


def guideline1_suggested_N(
    c: float, alpha: float, omega: float = 0.005, beta: float | None = None
) -> int:
    c_val = float(np.clip(c, 1e-12, 1 - 1e-12))
    beta_val = float(alpha if beta is None else beta)
    beta_val: float = min(max(beta_val, 1e-6), 0.49)

    base1: float = 2.0 * (1.0 - c_val) / (omega**2 * c_val)
    base2: float = 8.0 * (1.0 - c_val) / c_val
    erf_arg = float(np.clip(1.0 - 2.0 * beta_val, -0.999999, 0.999999))
    erf_sq: float = float(erfinv(erf_arg)) ** 2

    return int(np.ceil(max(base1, base2) * erf_sq))


def guideline1_c_boundary_for_fixed_N(
    *, N_fixed: int, alpha: float, omega: float, beta: float | None = None
) -> float:
    """
    Guideline1: N >= max(2(1-c)/(omega^2 c), 8(1-c)/c) * (erf^{-1}(1-2beta))^2
    を満たす境界 c* を返す。
    A = max(2/omega^2, 8), E = (erf^{-1}(1-2beta))^2
    => c* = (A*E) / (N + A*E)
    """
    if N_fixed <= 0:
        raise ValueError("N_fixed must be positive.")

    beta_val = float(alpha if beta is None else beta)
    beta_val: float = min(max(beta_val, 1e-6), 0.49)

    A1: float = 2.0 / (float(omega) ** 2)
    inv_val = float(erfinv(float(np.clip(1.0 - 2.0 * beta_val, -0.999999, 0.999999))))
    A2 = 8.0
    A: float = max(A1, A2)
    E: float = inv_val**2

    c_star = float((A * E) / (float(N_fixed) + (A * E)))
    # safety clip
    return float(np.clip(c_star, 1e-12, 1.0 - 1e-12))


def plot_sweep_gap_from_upper_with_errorbars(
    df: pd.DataFrame,
    *,
    is_N: bool = True,  # True: x=N, False: x=c
    epsilon_theoretical: float = 1.0,
    k: int = 2,
    # guideline line for N-sweep
    c_for_guideline: float = 1e-2,
    alpha: float = 1e-2,
    omega: float = 0.005,
    shift_list: list[float] | None = None,
    outpath: str = "results/fig_sweep_gap_from_upper.pdf",
) -> None:
    """
    y-axis: (GRR + logRmax_eff) - estimate   (mean ± 1σ over seeds)

    LRT (default color):
      - ○ solid : upper - eps_emp
      - ▽ dotted: upper - eps_lower
      - △ dotted: upper - eps_upper

    LRT_decomp (tab:purple)  [data protocol name is still "LRT_indirect"]:
      - ▽ solid : upper - eps_lower
      - △ dotted: upper - eps_upper

    If is_N:
      x-axis = N (log scale) + vertical guideline line at N_guideline(c_for_guideline)
    else (c-sweep):
      x-axis = c (log scale) + vertical boundary line c* such that Guideline1(N>=...) is satisfied for fixed N
    """
    xcol: Literal['N'] | Literal['c'] = "N" if is_N else "c"

    # --- filter fixed params ---
    df0: pd.DataFrame = df.copy()
    df0 = df0[df0["k"] == k]
    df0 = df0[np.isclose(df0["epsilon"].astype(float), float(epsilon_theoretical))]

    if shift_list is None:
        shift_list = sorted(df0["mean_shift"].dropna().unique().tolist())

    # --- Guideline lines ---
    N_guideline: int | None = None
    c_boundary: float | None = None

    if is_N:
        # N-sweep: existing guideline (vertical line at N_guideline for given c)
        # (you already have guideline1_suggested_N; keep using it)
        N_guideline = guideline1_suggested_N(
            c=c_for_guideline, alpha=alpha, omega=omega, beta=alpha
        )
    else:
        # c-sweep: infer fixed N from df
        N_vals: list[int] = sorted(set(int(v) for v in df0["N"].dropna().unique().tolist()))
        if len(N_vals) != 1:
            raise ValueError(f"c-sweep expects fixed N, but got N values: {N_vals}")
        N_fixed: int = N_vals[0]
        c_boundary = guideline1_c_boundary_for_fixed_N(
            N_fixed=N_fixed, alpha=alpha, omega=omega, beta=alpha
        )

    # --- build upper = GRR + logRmax_eff (per seed, per (shift,x)) ---
    grr: pd.DataFrame = df0[df0["protocol"] == "GRR"][["mean_shift", xcol, "seed", "eps_emp"]].copy()
    lr: pd.DataFrame = (
        df0[["mean_shift", xcol, "seed", "logRmax_eff"]]
        .drop_duplicates(["mean_shift", xcol, "seed"])
        .copy()
    )

    upper: pd.DataFrame = grr.merge(lr, on=["mean_shift", xcol, "seed"], how="inner")
    upper["upper"] = upper["eps_emp"].astype(float) + upper["logRmax_eff"].astype(float)
    upper = upper[["mean_shift", xcol, "seed", "upper"]]

    # --- helper: aggregate gap mean/std over seeds ---
    def agg_gap(df_gap: pd.DataFrame, label: str) -> pd.DataFrame:
        g: pd.DataFrame = df_gap.groupby(["mean_shift", xcol], as_index=False).agg(
            mean=("gap", "mean"),
            std=("gap", lambda x: float(np.std(x, ddof=1)) if len(x) > 1 else 0.0),
        )
        g["series"] = label
        return g

    # ===== LRT =====
    lrt: pd.DataFrame = df0[df0["protocol"] == "LRT"][
        ["mean_shift", xcol, "seed", "eps_emp", "eps_lower", "eps_upper"]
    ].copy()
    lrt = lrt.merge(upper, on=["mean_shift", xcol, "seed"], how="inner")

    lrt_emp: pd.DataFrame = lrt.copy()
    lrt_emp["gap"] = lrt_emp["upper"] - lrt_emp["eps_emp"]
    g_lrt_emp: pd.DataFrame = agg_gap(lrt_emp, "LRT_emp")

    lrt_lo: pd.DataFrame = lrt.dropna(subset=["eps_lower"]).copy()
    lrt_lo["gap"] = lrt_lo["upper"] - lrt_lo["eps_lower"]
    g_lrt_lo: pd.DataFrame = agg_gap(lrt_lo, "LRT_lower")

    lrt_hi: pd.DataFrame = lrt.dropna(subset=["eps_upper"]).copy()
    lrt_hi["gap"] = lrt_hi["upper"] - lrt_hi["eps_upper"]
    g_lrt_hi: pd.DataFrame = agg_gap(lrt_hi, "LRT_upper")

    # ===== LRT_decomp (data protocol name is "LRT_indirect") =====
    ind: pd.DataFrame = df0[df0["protocol"] == "LRT_indirect"][
        ["mean_shift", xcol, "seed", "eps_lower", "eps_upper"]
    ].copy()
    ind = ind.merge(upper, on=["mean_shift", xcol, "seed"], how="inner")

    ind_lo: pd.DataFrame = ind.dropna(subset=["eps_lower"]).copy()
    ind_lo["gap"] = ind_lo["upper"] - ind_lo["eps_lower"]
    g_ind_lo: pd.DataFrame = agg_gap(ind_lo, "DECOMP_lower")

    ind_hi: pd.DataFrame = ind.dropna(subset=["eps_upper"]).copy()
    ind_hi["gap"] = ind_hi["upper"] - ind_hi["eps_upper"]
    g_ind_hi: pd.DataFrame = agg_gap(ind_hi, "DECOMP_upper")

    agg: pd.DataFrame = pd.concat(
        [g_lrt_emp, g_lrt_lo, g_lrt_hi, g_ind_lo, g_ind_hi], ignore_index=True
    )

    # --- plotting ---
    fig, axes = plt.subplots(
        1, len(shift_list), figsize=(5.2 * len(shift_list), 3.8), sharey=True
    )
    if len(shift_list) == 1:
        axes: list = [axes]

    for ax, shift in zip(axes, shift_list):
        sub: pd.DataFrame = agg[np.isclose(agg["mean_shift"], float(shift))].copy().sort_values(xcol)

        ax.axhline(0.0, linestyle="--", linewidth=1.0)

        if is_N and N_guideline is not None:
            ax.axvline(float(N_guideline), linestyle="--", linewidth=1.0)
        if (not is_N) and (c_boundary is not None):
            ax.axvline(float(c_boundary), linestyle="--", linewidth=1.0)

        ax.grid(True, linestyle="dashdot", linewidth=0.5)

        # --- LRT (default color) ---
        s: pd.DataFrame = sub[sub["series"] == "LRT_emp"]
        ln: list = ax.plot(
            s[xcol], s["mean"], marker="o", linestyle="-", label=r"LRT: Upper - eps$_\mathrm{emp}"
        )
        c_lrt: str = ln[0].get_color()
        ax.errorbar(
            s[xcol], s["mean"], yerr=s["std"], linestyle="None", capsize=3, color=c_lrt
        )

        for lab, mkr in [("LRT_lower", "v"), ("LRT_upper", "^")]:
            s2: pd.DataFrame = sub[sub["series"] == lab]
            ax.plot(
                s2[xcol],
                s2["mean"],
                linestyle=":",
                marker=mkr,
                color=c_lrt,
                label=r"LRT: Upper - eps$_\mathrm{lower}$" if mkr == "v" else r"LRT: Upper - eps$_\mathrm{upper}$",
            )
            ax.errorbar(
                s2[xcol],
                s2["mean"],
                yerr=s2["std"],
                linestyle="None",
                capsize=3,
                color=c_lrt,
            )

        # --- LRT_decomp (purple) ---
        base = "tab:purple"
        s3: pd.DataFrame = sub[sub["series"] == "DECOMP_lower"]
        ax.plot(
            s3[xcol],
            s3["mean"],
            linestyle="-",
            marker="v",
            color=base,
            linewidth=2.5,
            label="LRT_decomp: upper - eps_lower",
        )
        ax.errorbar(
            s3[xcol],
            s3["mean"],
            yerr=s3["std"],
            linestyle="None",
            capsize=3,
            color=base,
        )

        s4: pd.DataFrame = sub[sub["series"] == "DECOMP_upper"]
        ax.plot(
            s4[xcol],
            s4["mean"],
            linestyle=":",
            marker="^",
            color=base,
            label="LRT_decomp: upper - eps_upper",
        )
        ax.errorbar(
            s4[xcol],
            s4["mean"],
            yerr=s4["std"],
            linestyle="None",
            capsize=3,
            color=base,
        )

        ax.set_xscale("log")
        ax.set_xlabel("N (nb_trials)" if is_N else "c")
        ax.set_title(f"mean_shift = {shift}")

        # --- annotate boundary meaning (optional but helpful) ---
        if (not is_N) and (c_boundary is not None):
            # text near top
            y_top: float = ax.get_ylim()[1]
            ax.text(
                float(c_boundary),
                y_top,
                "",
                ha="left",
                va="top",
            )

    axes[0].set_ylabel("estimate error")

    # legend: avoid overlap
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        bbox_to_anchor=(0.5, 1.05),
        frameon=True,
    )
    fig.tight_layout(rect=(0.04, 0.0, 1.0, 0.80))

    plt.savefig(outpath, dpi=300, bbox_inches="tight", pad_inches=0.1)
    plt.show()

def plot_results_approx_ldp_protocols(df: pd.DataFrame, analysis: str, lst_protocol: list, lst_eps: list, lst_k: list):

    fig, ax = plt.subplots(3, 2, figsize=(12, 10.5), sharey=True)
    plt.subplots_adjust(wspace=0.25, hspace=0.5)


    c = 0 # column
    for row, protocol in enumerate(lst_protocol):
        
        r: int = row // 2
        if c>1:
            c=0
        ax[r, c].yaxis.set_tick_params(which='both', labelbottom=True)
        ax[r, c].grid(color='grey', linestyle='dashdot', linewidth=0.5)
        if protocol != "GM":
            ax[r, c].plot(lst_eps, label='Theoretical $\epsilon$', color ='black', linestyle='dashed')
        else:
            ax[r, c].plot([eps for eps in lst_eps if eps <= 1], label='Theoretical $\epsilon$', color ='black', linestyle='dashed')

        mkr_idx = 0
        for k in lst_k:

            results_k = []
            variation_k = []
            if protocol != "GM":
                for epsilon in lst_eps:
                    df_eps = df.loc[(df.protocol == protocol) & (df.epsilon == epsilon) & (df.k == k)]['eps_emp'].clip(0)
                    results_k.append(df_eps.mean())
                    variation_k.append(df_eps.std())
            else:
                for epsilon in [eps for eps in lst_eps if eps <= 1]:
                    df_eps = df.loc[(df.protocol == protocol) & (df.epsilon == epsilon) & (df.k == k)]['eps_emp'].clip(0)
                    results_k.append(df_eps.mean())
                    variation_k.append(df_eps.std())
            
            
            std_minus = np.array(results_k) - np.array(variation_k)
            std_plus = np.array(results_k) + np.array(variation_k)        
            ax[r, c].fill_between(range(len(variation_k)), std_minus, std_plus, alpha=0.3)
            ax[r, c].plot(results_k, label = 'k={}'.format(k), marker = markers[mkr_idx])
            
            mkr_idx+=1

        ax[r, c].set_yscale('log')
        ax[r, c].set_xticks(range(len(lst_eps)))
        ax[r, c].set_xticklabels(lst_eps)
        ax[r, c].set_title(protocol + ', $\\delta=1e^{-5}$', fontsize=20)
        ax[r, c].set_ylabel('Estimated $\\epsilon_{emp}$')
        ax[r, c].set_xlabel('Theoretical $\\epsilon$')
        ax[r, c].set_yticks([1e-1, 1e0, 1e1])  

        c += 1

        ax[0, 0].legend(columnspacing=0.3, ncol=8, loc='upper center', bbox_to_anchor=(1.05, 1.5))
        plt.savefig('results/fig_results_'+analysis+'.pdf', dpi=500, bbox_inches = 'tight',pad_inches = 0.1)

    return plt.show()


def plot_results_approx_ldp_delta_impact(df_delta_imp: pd.DataFrame, analysis: str, lst_protocol: list, lst_eps: list, lst_k: list, lst_delta: list):
    
    df_pure = pd.read_csv('results/ldp_audit_results_main_pure_ldp_protocols.csv')    
    df_approx = pd.read_csv('results/ldp_audit_results_main_approx_ldp_protocols.csv')
    df = pd.concat([df_pure, df_approx, df_delta_imp], axis=0)

    bar_width = 0.17  # width of bars
    eps_positions = np.arange(len(lst_eps))  # position of groups
    n_delta = len(lst_delta) + 1  # Include df_pure in the delta count

    for k in lst_k:
        print('--------------------------------k={}--------------------------------'.format(k))

        fig, ax = plt.subplots(3, 2, figsize=(10, 9), sharey=True)
        plt.subplots_adjust(wspace=0.3, hspace=0.6)
        for i, protocol in enumerate(lst_protocol):
            r, c = divmod(i, 2)  
            
            ax[r, c].yaxis.set_tick_params(which='both', labelbottom=True)
            ax[r, c].grid(color='grey', linestyle='dashdot', linewidth=0.5, zorder=0)

            # Adjust delta_positions to accommodate all bars, including the one for df_pure
            delta_positions = [eps_positions + (i - n_delta / 2) * bar_width for i in range(n_delta)]
            
            test_delta = []
            for delta_idx, delta in enumerate([None] + lst_delta):  # Add None for df_pure
                
                results_k = []
                variation_k = []
                for epsilon in lst_eps:
                    if delta_idx == 0:  # Handle df_pure
                        df_eps = df_pure.loc[(df_pure['protocol'] == protocol[1:]) & (df_pure['epsilon'] == epsilon) & (df_pure['k'] == k)]['eps_emp'].clip(0)
                        
                    else:
                        df_eps = df.loc[(df['protocol'] == protocol) & (df['epsilon'] == epsilon) & (df['delta'] == delta) & (df['k'] == k)]['eps_emp'].clip(0)
    
                    results_k.append(df_eps.mean() if df_eps.mean() > 0 else 0)
                    variation_k.append(df_eps.std() if df_eps.std() > 0 else 0)
                
                test_delta.append(results_k)
                label = '$\\delta=0$' if delta_idx == 0 else f'$\\delta=${"{:.0e}".format(delta)}'
                ax[r, c].bar(delta_positions[delta_idx], results_k, width=bar_width, label=label, zorder=5)
                ax[r, c].errorbar(delta_positions[delta_idx], results_k, yerr=variation_k, fmt='none', capsize=4, color='black', zorder=5)

            ax[r, c].set_xticks(eps_positions)
            ax[r, c].set_xticklabels(lst_eps)
            ax[r, c].set_yticks(lst_eps)
            ax[r, c].set_ylim(0, 1.05)  
            ax[r, c].set_title(protocol, fontsize=20)
            ax[r, c].set_ylabel('Estimated $\\epsilon_{emp}$')
            ax[r, c].set_xlabel('Theoretical $\\epsilon$')

        # Adjust legend for the entire figure, considering the extra "Pure" category
        handles, labels = ax[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, columnspacing=0.8, handlelength=1.5, loc='upper center', bbox_to_anchor=(0.49, 1.0), ncol=n_delta)
        plt.savefig('results/fig_results_'+analysis+'_k_'+str(k)+'.pdf', dpi=500, bbox_inches = 'tight',pad_inches = 0.1)
        plt.show()
    
    return plt.show()


def plot_results_lho_protocol(df_lho, analysis, lst_k, lst_g):
    
    plt.figure(figsize=(6, 3))
    plt.grid(color='grey', linestyle='dashdot', linewidth=0.5)
    mkr_idx = 0
    for k in lst_k:
        results_k = []
        variation_k = []
        for g in lst_g:
            df_eps = df_lho.loc[(df_lho.g == g) & (df_lho.k == k)]['eps_emp'].clip(0)
            results_k.append(df_eps.mean())
            variation_k.append(df_eps.std())

        std_minus = np.array(results_k) - np.array(variation_k)
        std_plus = np.array(results_k) + np.array(variation_k)        
        plt.fill_between(range(len(lst_g)), std_minus, std_plus, alpha=0.3)
        plt.plot(results_k, label = 'k={}'.format(k), marker = markers[mkr_idx])

        mkr_idx+=1

    plt.xticks(range(len(lst_g)), lst_g)
    plt.ylabel('Estimated $\epsilon_{emp}$')    

    plt.xlabel('Hash domain $g$')
    plt.legend(columnspacing=0.8, ncol=3, loc='upper center', bbox_to_anchor=(0.49, 1.48))
    plt.savefig('results/fig_results_'+analysis+'.pdf', dpi=500, bbox_inches = 'tight',pad_inches = 0.1)

    return plt.show()


def plot_results_longitudinal_pure_ldp_protocols(df_pure_long: pd.DataFrame, df_approx_long: pd.DataFrame, pure_ldp_protocols: list, approx_lst_protocol: list, 
                                                analysis: str, lst_eps: list, lst_k: list, lst_tau: list, eps_ub: float):
    
    df_seq = pd.concat([df_pure_long, df_approx_long], axis=0)
    lst_protocol = pure_ldp_protocols + approx_lst_protocol

    for k in lst_k:
        print('--------------------------------k={}--------------------------------'.format(k))
        
        fig, ax = plt.subplots(2, 2, figsize=(14, 9), sharey=True)
        plt.subplots_adjust(wspace=0.2, hspace=0.45)

        c = 0 # column
        for row, epsilon in enumerate(lst_eps):
            
            r = row // 2
            if c>1:
                c=0
            
            ax[r, c].yaxis.set_tick_params(which='both', labelbottom=True)
            ax[r, c].grid(color='grey', linestyle='dashdot', linewidth=0.5)
            ax[r, c].plot(epsilon*lst_tau, label='Theoretical $\\tau\epsilon$ (Sequential Composition)', color ='black', linestyle='dashed')
            ax[r, c].hlines(y=eps_ub, xmin=0, xmax=len(lst_tau)-1, label='Optimal $\epsilon_{OPT}$ (Monte Carlo Upper Bound)', color='red', linestyle='dotted')

            mkr_idx = 0
            for protocol in lst_protocol:

                results_k = []
                variation_k = []
                for idx_tau, tau in enumerate(lst_tau):
                    df_eps = df_seq.loc[(df_seq.protocol == protocol) & (df_seq.tau == tau) & (df_seq.k == k) & (df_seq.epsilon == epsilon)]['eps_emp'].clip(0)
                    results_k.append(df_eps.mean())
                    variation_k.append(df_eps.std())
                    
                std_minus = np.array(results_k) - np.array(variation_k)
                std_plus = np.array(results_k) + np.array(variation_k)        
                ax[r, c].fill_between(range(len(lst_tau)), std_minus, std_plus, alpha=0.3)
                ax[r, c].plot(results_k, label = protocol, marker = markers[mkr_idx])

                mkr_idx+=1

            ax[r, c].set_yscale('log')
            ax[r, c].set_xticks(range(len(lst_tau)))
            ax[r, c].set_xticklabels(lst_tau)
            ax[r, c].set_title('Per report $\epsilon={}$'.format(epsilon), fontsize=20)
            ax[r, c].set_ylabel('Estimated $\epsilon_{emp}$')
            ax[r, c].set_xlabel('Number of Data Collections $\\tau$')
            c += 1

        ax[0, 0].legend(columnspacing=0.3, handlelength=1.5, ncol=6, loc='upper center', bbox_to_anchor=(1.03, 1.55))
        plt.savefig('results/fig_results_'+analysis+'_k_'+str(k)+'.pdf', dpi=500, bbox_inches = 'tight',pad_inches = 0.1)
        plt.show()

    return plt.show()

def plot_results_multidimensional(df: pd.DataFrame, lst_protocol: list, analysis: str, lst_eps: list, lst_k: list, lst_d: list):

    for k in lst_k:
        print('--------------------------------k={}--------------------------------'.format(k))

        fig, ax = plt.subplots(1, 2, figsize=(11.4, 2.8), sharey=True)
        plt.subplots_adjust(wspace=0.25, hspace=0.2)
        
        for r, d in enumerate(lst_d):
            
            ax[r].yaxis.set_tick_params(which='both', labelbottom=True)
            ax[r].grid(color='grey', linestyle='dashdot', linewidth=0.5)
            ax[r].plot(lst_eps, label='Theoretical $\epsilon$', color ='black', linestyle='dashed')

            mkr_idx=0
            for protocol in lst_protocol:
                results_k = []
                variation_k = []
                for epsilon in lst_eps:
                    df_eps = df.loc[(df.protocol == protocol) & (df.k == k) & (df.epsilon == epsilon) & (df.d == d)]['eps_emp'].clip(0)
                    results_k.append(df_eps.mean())
                    variation_k.append(df_eps.std())

                std_minus = np.array(results_k) - np.array(variation_k)
                std_plus = np.array(results_k) + np.array(variation_k)        
                ax[r].fill_between(range(len(lst_eps)), std_minus, std_plus, alpha=0.3)
                ax[r].plot(results_k, label = lst_protocol[mkr_idx], marker = markers[mkr_idx])
                mkr_idx+=1 

            ax[r].set_yscale('log')
            ax[r].set_xticks(range(len(lst_eps)))
            ax[r].set_xticklabels(lst_eps)

            ax[r].set_title('$d={}$'.format(d))
            ax[r].set_ylabel('Estimated $\epsilon_{emp}$')
            ax[r].set_xlabel('Theoretical $\epsilon$')

        ax[0].legend(columnspacing=0.3, ncol=3, loc='upper center', bbox_to_anchor=(1.1, 1.65))
        plt.savefig('results/fig_results_'+analysis+'_k_'+str(k)+'.pdf', dpi=500, bbox_inches = 'tight',pad_inches = 0.1)
        plt.show()

    return plt.show()


def plot_result_pure_ldp_package(df: pd.DataFrame, analysis: str, lst_protocol: list, lst_eps: list, lst_k: list):

    fig, ax = plt.subplots(1, 2, figsize=(11.4, 2.8))
    plt.subplots_adjust(wspace=0.25, hspace=0.2)

    # Define the coordinates for the box
    box_x = -0.22  # x-coordinate of the box
    box_y = .23  # y-coordinate of the box
    box_width = 1.6  # width of the box
    box_height = 0.9  # height of the box

    for r, protocol in enumerate(lst_protocol):
        
        ax[r].grid(color='grey', linestyle='dashdot', linewidth=0.5)
        ax[r].plot(lst_eps, label='Theoretical $\epsilon$', color ='black', linestyle='dashed')
        mkr_idx = 0
        for k in lst_k:
            results_k = []
            variation_k = []
            for epsilon in lst_eps:
                df_eps = df.loc[(df.protocol == protocol) & (df.epsilon == epsilon) & (df.k == k)]['eps_emp'].clip(0)
                results_k.append(df_eps.mean())
                variation_k.append(df_eps.std())
            
            std_minus = np.array(results_k) - np.array(variation_k)
            std_plus = np.array(results_k) + np.array(variation_k)        
            ax[r].fill_between(range(len(lst_eps)), std_minus, std_plus, alpha=0.3)
            ax[r].plot(results_k, label = 'k={}'.format(k), marker = markers[mkr_idx])
            mkr_idx+=1
        
        ax[r].set_yscale('log')
        ax[r].set_xticks(range(len(lst_eps)))
        ax[r].set_xticklabels(lst_eps)
        
        ax[r].set_title('SUE' if 'SUE' in protocol else 'OUE')
        ax[r].set_ylabel('Estimated $\epsilon_{emp}$')
        
        # Add the box to the plot
        box = Rectangle((box_x, box_y), box_width, box_height, fill=False, edgecolor='chocolate', linewidth=2)
        ax[r].add_patch(box)
        ax[r].set_xticklabels(lst_eps)
        ax[r].set_xlabel('Theoretical $\epsilon$')

    ax[0].legend(columnspacing=0.3, ncol=53, loc='upper center', bbox_to_anchor=(1.05, 1.45))
    plt.savefig('results/fig_results_'+analysis+'.pdf', dpi=500, bbox_inches = 'tight',pad_inches = 0.1)
    
    return plt.show()

def _ensure_float(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _series_mean_std(
    df: pd.DataFrame, protocol: str, k: int, value_col: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    d = df[(df["protocol"] == protocol) & (df["k"] == k)].copy()
    if d.empty:
        return np.array([]), np.array([]), np.array([])
    g = d.groupby("epsilon")[value_col]
    eps = g.mean().index.to_numpy(dtype=float)
    mean = g.mean().to_numpy(dtype=float)
    std = g.std(ddof=1).fillna(0.0).to_numpy(dtype=float)
    return eps, mean, std


def _grr_plus_logrmax_mean_std(
    df: pd.DataFrame, k: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    seed ごとに GRR(eps_emp) + logRmax_eff を作ってから epsilon ごとに mean/std
    """
    grr = df[(df["protocol"] == "GRR") & (df["k"] == k)][
        ["seed", "epsilon", "eps_emp"]
    ].copy()
    # logRmax_eff は LRT 行から seed ごとに取る（epsilon に依存しない想定）
    lr = (
        df[(df["protocol"] == "LRT") & (df["k"] == k)][["seed", "logRmax_eff"]]
        .drop_duplicates("seed")
        .copy()
    )
    merged = grr.merge(lr, on="seed", how="inner")
    if merged.empty:
        return np.array([]), np.array([]), np.array([])
    merged["sum"] = merged["eps_emp"] + merged["logRmax_eff"]
    g = merged.groupby("epsilon")["sum"]
    eps = g.mean().index.to_numpy(dtype=float)
    mean = g.mean().to_numpy(dtype=float)
    std = g.std(ddof=1).fillna(0.0).to_numpy(dtype=float)
    return eps, mean, std


def _nice_step_and_max(ymax: float, n_ticks: int) -> tuple[float, float]:
    """
    ymax に対して、人間が読みやすい tick 間隔 step と、
    step の倍数に切り上げた ymax_nice を返す。
    """
    if not np.isfinite(ymax) or ymax <= 0:
        return 1.0, 1.0

    raw_step: float = ymax / max(n_ticks - 1, 1)
    exp: float = 10 ** math.floor(math.log10(raw_step))
    frac: float = raw_step / exp

    # 1, 2, 2.5, 5, 10 のどれかに丸める（見やすい）
    if frac <= 1.0:
        nice = 1.0
    elif frac <= 2.0:
        nice = 2.0
    elif frac <= 2.5:
        nice = 2.5
    elif frac <= 5.0:
        nice = 5.0
    else:
        nice = 10.0

    step: float = nice * exp
    ymax_nice: float = step * math.ceil(ymax / step)
    return step, ymax_nice


def plot_lrt_compare_2x2(
    csv_paths: list[str],
    outpath: str = "results/fig_lrt_compare_2x2.pdf",
    panel_titles: list[str] | None = None,
    k: int | None = None,
    yscale: str = "log",  # "log" or "linear"
) -> None:
    """
    4つのCSVを読み込み, 2x2で LRT_compare パネルのみを描く
    - 理論: y = epsilon
    - LRT: eps_emp の mean±std + lower/upper(mean)
    - GRR + logRmax: mean±std
    """
    if len(csv_paths) != 4:
        raise ValueError("csv_paths must contain exactly 4 files (for 2x2).")

    if panel_titles is None:
        # ファイル名から雑にタイトル生成
        panel_titles = [Path(p).stem for p in csv_paths]
    if len(panel_titles) != 4:
        raise ValueError("panel_titles must have length 4.")

    dfs: list[pd.DataFrame] = []
    for p in csv_paths:
        d: pd.DataFrame = pd.read_csv(p)
        d = _ensure_float(
            d,
            [
                "seed",
                "k",
                "epsilon",
                "eps_emp",
                "logRmax_eff",
                "eps_lower",
                "eps_upper",
            ],
        )
        dfs.append(d)

    # k を自動決定（指定がなければ最頻値）
    if k is None:
        ks = dfs[0]["k"].dropna().astype(int)
        if ks.empty:
            raise ValueError(
                "k could not be inferred from the first csv (column k missing/empty)."
            )
        k = int(ks.value_counts().idxmax())

    fig, ax = plt.subplots(2, 2, figsize=(10.8, 7.6), sharey=True)
    ax = np.atleast_2d(ax)

    for idx, (df, title) in enumerate(zip(dfs, panel_titles)):
        r, ccol = divmod(idx, 2)
        a = ax[r, ccol]
        a.grid(color="grey", linestyle="dashdot", linewidth=0.5)

        # x 軸（epsilon のユニーク値）
        eps_list = np.sort(df["epsilon"].dropna().unique().astype(float))
        x = np.arange(len(eps_list), dtype=int)
        m = {float(e): i for i, e in enumerate(eps_list)}

        # --- theoretical y = epsilon ---
        a.plot(
            x,
            eps_list,
            linestyle="dashed",
            color="black",
            label="Theoretical $\\varepsilon$",
            zorder=1,
        )

        # =========================
        # LRT（完全）: lower=実線, upper=点線
        # =========================
        eps_lo, lo_mean, _ = _series_mean_std(df, "LRT", k, "eps_lower")
        eps_hi, hi_mean, _ = _series_mean_std(df, "LRT", k, "eps_upper")

        line_color = None
        if eps_lo.size > 0:
            xlo = np.array([m[float(e)] for e in eps_lo], dtype=int)
            ln = a.plot(
                xlo,
                lo_mean,
                marker="s",
                linestyle="-",
                label="LRT (lower)",
                zorder=10,
            )
            line_color = ln[0].get_color()
        else:
            line_color = "tab:blue"

        if eps_hi.size > 0:
            xhi = np.array([m[float(e)] for e in eps_hi], dtype=int)
            a.plot(
                xhi,
                hi_mean,
                linestyle=":",
                marker="^",
                color=line_color,
                label="LRT (upper)",
                zorder=11,
            )

        # =========================
        # LRT_indirect（分離監査 = LRT-Decomp）
        #   lower=実線(太め), upper=点線
        # =========================
        base_color = "tab:purple"

        eps_lo_d, d_lo_mean, _ = _series_mean_std(df, "LRT_indirect", k, "eps_lower")
        if eps_lo_d.size > 0:
            xlo = np.array([m[float(e)] for e in eps_lo_d], dtype=int)
            a.plot(
                xlo,
                d_lo_mean,
                color=base_color,
                linestyle="-",
                linewidth=2.5,
                marker="v",
                label="LRT_decomp (lower)",
                zorder=21,
            )

        eps_hi_d, d_hi_mean, _ = _series_mean_std(df, "LRT_indirect", k, "eps_upper")
        if eps_hi_d.size > 0:
            xhi = np.array([m[float(e)] for e in eps_hi_d], dtype=int)
            a.plot(
                xhi,
                d_hi_mean,
                color=base_color,
                linestyle=":",
                marker="^",
                label="LRT_decomp (upper)",
                zorder=21,
            )

        # =========================
        # GRR + logRmax（オレンジ）: 最前面
        # =========================
        df_k = df.loc[df["k"] == k].copy()

        df_grr_all = df_k.loc[
            df_k["protocol"] == "GRR", ["seed", "epsilon", "eps_emp"]
        ].dropna()

        lr_all = df_k.loc[:, ["seed", "logRmax_eff"]].dropna().drop_duplicates("seed")

        if not df_grr_all.empty and not lr_all.empty:
            y_mean, y_std = [], []
            for eps in eps_list:
                grr_e = df_grr_all.loc[
                    df_grr_all["epsilon"] == eps, ["seed", "eps_emp"]
                ]
                merged = grr_e.merge(lr_all, on="seed", how="inner")
                if merged.empty:
                    y_mean.append(np.nan)
                    y_std.append(np.nan)
                else:
                    s = (merged["eps_emp"] + merged["logRmax_eff"]).to_numpy(
                        dtype=float
                    )
                    y_mean.append(float(np.mean(s)))
                    y_std.append(float(np.std(s, ddof=1)) if s.size > 1 else 0.0)

            y_mean = np.array(y_mean, dtype=float)
            y_std = np.array(y_std, dtype=float)

            a.plot(
                x,
                y_mean,
                color="tab:orange",
                marker="o",
                label="RR + $\\log R_{\\max}$",
                zorder=100,
            )
            a.fill_between(
                x,
                y_mean - y_std,
                y_mean + y_std,
                color="tab:orange",
                alpha=0.20,
                zorder=90,
            )

        if yscale == "log":
            a.set_yscale("log")

        a.set_title(title, fontsize=25)
        a.set_xticks(x)
        a.set_xticklabels([str(e) for e in eps_list])

        if r == 1:
            a.set_xlabel("Theoretical $\\varepsilon$", fontsize=25)
        else:
            a.set_xlabel("")
            a.tick_params(axis="x", labelbottom=False)
        if ccol == 0:
            a.set_ylabel("Estimated $\\varepsilon$ (CI)", fontsize=25)

    # 図全体の凡例（重複除去）
    handles, labels = [], []
    for a in ax.flat:
        h, l = a.get_legend_handles_labels()
        for hh, ll in zip(h, l):
            if ll not in labels:
                handles.append(hh)
                labels.append(ll)

    ncol = 3
    plt.subplots_adjust(top=0.82)

    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=ncol,
        frameon=True,
        columnspacing=0.9,
        handletextpad=0.6,
        fontsize=20,
        borderaxespad=0.2,
    )

    outpath = str(outpath)
    Path(outpath).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, dpi=500, bbox_inches="tight", pad_inches=0.1)
    plt.show()


def plot_results_example_audit_2x2(
    csv_paths: list[str],
    outpath: str = "results/fig_results_summary_audit_2x2.pdf",
    panel_titles: list[str] | None = None,
    epsilon: float = 1.0,
    k: int | None = None,
    lst_protocol: list[str] | None = None,
) -> None:
    """
    4つのCSVを読み込み, 各CSVについて (epsilon, k) を固定した棒グラフを 2x2 で描く。
    利用CSVは plot_lrt_compare_2x2 と同じでOK。

    棒:
        - GRR (mean ± std)
        - LRT (mean ± std)
        - GRR + logRmax_eff (seed ごとに足してから mean ± std)
    参照線:
        - theoretical epsilon (赤破線)
    """
    if len(csv_paths) != 4:
        raise ValueError("csv_paths must contain exactly 4 files (for 2x2).")

    if panel_titles is None:
        panel_titles = [Path(p).stem for p in csv_paths]
    if len(panel_titles) != 4:
        raise ValueError("panel_titles must have length 4.")

    if lst_protocol is None:
        # ここがバーの表示順（必要なら好きに変えてOK）
        lst_protocol = ["GRR", "LRT"]

    dfs: list[pd.DataFrame] = []
    for p in csv_paths:
        d: pd.DataFrame = pd.read_csv(p)
        d = _ensure_float(
            d,
            [
                "seed",
                "k",
                "epsilon",
                "eps_emp",
                "logRmax_eff",
                "eps_lower",
                "eps_upper",
            ],
        )
        dfs.append(d)

    # k を自動決定（指定がなければ最頻値）
    if k is None:
        ks: pd.Series[int] = dfs[0]["k"].dropna().astype(int)
        if ks.empty:
            raise ValueError(
                "k could not be inferred from the first csv (column k missing/empty)."
            )
        k = int(ks.value_counts().idxmax())

    fig, ax = plt.subplots(2, 2, figsize=(10.8, 7.6), sharey=False)
    ax = np.atleast_2d(ax)
    plt.subplots_adjust(wspace=0.25, hspace=0.35)

    # ★ 行ごとの ymax を集計する（上段=0行、下段=1行）
    row_ymax: dict[int, float] = {0: 0.0, 1: 0.0}

    for idx, (df, title) in enumerate(zip(dfs, panel_titles)):
        r, c = divmod(idx, 2)
        a = ax[r, c]

        # --- plot_results_example_audit と同じ集計 ---
        df_eps: pd.DataFrame = df.loc[(df.epsilon == epsilon) & (df.k == k)]

        dic_eps: dict[str, float] = (
            df_eps.groupby("protocol")["eps_emp"].mean().to_dict()
        )
        dic_stds: dict[str, float] = (
            df_eps.groupby("protocol")["eps_emp"].std().to_dict()
        )

        # seed ごとに (GRR_eps + logRmax) を作ってから平均・std を取る
        grr: pd.DataFrame = df_eps[df_eps["protocol"] == "GRR"][["seed", "eps_emp"]]
        lr: pd.DataFrame = df_eps[["seed", "logRmax_eff"]].drop_duplicates("seed")
        grr_plus: pd.DataFrame = grr.merge(lr, on="seed", how="inner")
        grr_plus["sum"] = grr_plus["eps_emp"] + grr_plus["logRmax_eff"]
        grr_plus_mean: float = (
            float(grr_plus["sum"].mean()) if not grr_plus.empty else float("nan")
        )
        grr_plus_std: float = (
            float(grr_plus["sum"].std(ddof=1)) if len(grr_plus) > 1 else 0.0
        )

        # 表示順に 1 本追加
        proto_names: list[str] = lst_protocol + ["GRR +\n$\\log R_{\\max}$"]
        dic_eps["GRR +\n$\\log R_{\\max}$"] = grr_plus_mean
        dic_stds["GRR +\n$\\log R_{\\max}$"] = grr_plus_std

        values: list[float] = [
            float(dic_eps.get(key, float("nan"))) for key in proto_names
        ]
        stds: list[float] = [float(dic_stds.get(key, 0.0)) for key in proto_names]

        # このパネルで棒+誤差の最大値を更新（行ごとに集計）
        panel_max: float = float(np.nanmax(np.array(values) + np.array(stds)))
        row_ymax[r] = max(row_ymax[r], panel_max)

        n: int = len(proto_names)
        x: np.ndarray = np.arange(n, dtype=int)

        # --- 描画 ---
        a.grid(color="grey", linestyle="dashdot", linewidth=0.5, zorder=0)
        a.set_xlim(-0.5, n - 0.5)

        a.hlines(
            epsilon,
            -0.5,
            n - 0.5,
            label="Theoretical $\\varepsilon$",
            color="red",
            linestyle="dashed",
            zorder=3,
        )
        a.bar(x, values, zorder=10, width=0.65)
        a.errorbar(
            x,
            values,
            yerr=stds,
            ecolor="black",
            capsize=5,
            zorder=50,
            linestyle="None",
        )

        a.set_xticks(x)
        a.set_title(title, fontsize=25)

        # --- y-axis control (row-wise) ---
        if r == 0:
            # 上段：ズーム表示
            a.set_ylim(0, 2.0)
            a.set_yticks([0, 0.5, 1.0, 1.5, 2.0])
        else:
            # 下段：全体表示
            a.set_ylim(0, 16)
            a.set_yticks([0, 5, 10, 15])

        if c == 0:
            a.set_ylabel("Estimated $\\varepsilon_{emp}$", fontsize=25)
        if r == 1:
            a.set_xticks(x)
            a.set_xticklabels(proto_names, rotation=45, ha="right")
        else:
            # a.set_xticks([])          # ★ 0,1,2 を消す
            a.set_xticklabels([])     # ★ 念のため
            # a.set_xlabel("LDP Frequency Estimation Protocols", fontsize=20)
            a.set_xlabel("")

    # 行ごとの y 軸範囲と ticks を動的に決める
    pad_top: float = 1.10
    pad_bottom: float = 1.15

    top_target: float = row_ymax[0] * pad_top
    top_target = max(top_target, epsilon * 1.3)
    bot_target: float = row_ymax[1] * pad_bottom

    # 上段は細かく、下段は粗く
    top_step, top_ylim = _nice_step_and_max(top_target, n_ticks=5)
    bot_step, bot_ylim = _nice_step_and_max(bot_target, n_ticks=4)

    top_ticks = np.arange(0.0, top_ylim + 0.5 * top_step, top_step)
    bot_ticks = np.arange(0.0, bot_ylim + 0.5 * bot_step, bot_step)

    # 上段(0行)と下段(1行)へ適用
    for j in range(2):
        ax[0, j].set_ylim(0.0, top_ylim)
        ax[0, j].set_yticks(top_ticks)

        ax[1, j].set_ylim(0.0, bot_ylim)
        ax[1, j].set_yticks(bot_ticks)

    # 図全体の凡例（重複除去）
    handles, labels = [], []
    for a in ax.flat:
        h, l = a.get_legend_handles_labels()
        for hh, ll in zip(h, l):
            if ll not in labels:
                handles.append(hh)
                labels.append(ll)
    fig.legend(handles, labels, loc="lower right")

    outpath = str(outpath)
    Path(outpath).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, dpi=500, bbox_inches="tight", pad_inches=0.1)
    plt.show()

def _plot_sweep_gap_panel(
    ax: plt.Axes,  # type: ignore
    df: pd.DataFrame,
    *,
    is_N: bool,
    mean_shift: float,
    epsilon_theoretical: float,
    k: int,
    c_for_guideline: float,
    alpha: float,
    omega: float,
    fixed_value: float | None = None,
    fixed_label: str | None = None, 
) -> tuple[list, list]:
    # TODO: fixed 値のパラメータを引数化する
    xcol: Literal['N'] | Literal['c'] = "N" if is_N else "c"

    # --- filter fixed params + mean_shift ---
    df0: pd.DataFrame = df.copy()
    df0 = df0[df0["k"] == k]
    df0 = df0[np.isclose(df0["epsilon"].astype(float), float(epsilon_theoretical))]
    df0 = df0[np.isclose(df0["mean_shift"].astype(float), float(mean_shift))]

    # --- infer fixed value if not given  ---
    if fixed_value is None:
        if is_N:
            # N-sweep → c が固定
            if "c" in df0.columns:
                c_vals = sorted(df0["c"].dropna().unique())
                if len(c_vals) == 1:
                    fixed_value = float(c_vals[0])
                    fixed_label = "c"
        else:
            # c-sweep → N が固定
            if "N" in df0.columns:
                N_vals = sorted(df0["N"].dropna().unique())
                if len(N_vals) == 1:
                    fixed_value = float(N_vals[0])
                    fixed_label = "N"

    # --- Guideline vertical line (orange) ---
    N_guideline: int | None = None
    c_boundary = None
    if is_N:
        N_guideline = guideline1_suggested_N(
            c=c_for_guideline, alpha=alpha, omega=omega, beta=alpha
        )
    else:
        N_vals: list[int] = sorted(set(int(v) for v in df0["N"].dropna().unique().tolist()))
        if len(N_vals) != 1:
            raise ValueError(f"c-sweep expects fixed N, but got N values: {N_vals}")
        c_boundary = guideline1_c_boundary_for_fixed_N(
            N_fixed=N_vals[0], alpha=alpha, omega=omega, beta=alpha
        )

    # --- build upper ---
    grr: pd.DataFrame = df0[df0["protocol"] == "GRR"][["seed", xcol, "eps_emp"]].copy()
    lr: pd.DataFrame = df0[["seed", xcol, "logRmax_eff"]].drop_duplicates(["seed", xcol]).copy()

    upper: pd.DataFrame = grr.merge(lr, on=["seed", xcol], how="inner")
    upper["upper"] = upper["eps_emp"] + upper["logRmax_eff"]
    upper = upper[["seed", xcol, "upper"]]

    def _agg_gap(df_gap: pd.DataFrame, label: str) -> pd.DataFrame:
        g: pd.DataFrame = df_gap.groupby([xcol], as_index=False).agg(
            mean=("gap", "mean"),
            std=("gap", lambda x: float(np.std(x, ddof=1)) if len(x) > 1 else 0.0),
        )
        g["series"] = label
        return g

    # ===== LRT (完全) =====
    lrt: pd.DataFrame = df0[df0["protocol"] == "LRT"][["seed", xcol, "eps_lower", "eps_upper"]].copy()
    lrt = lrt.merge(upper, on=["seed", xcol], how="inner")

    lrt_lo = lrt.dropna(subset=["eps_lower"]).copy()
    lrt_lo["gap"] = lrt_lo["upper"] - lrt_lo["eps_lower"]
    g_lrt_lo = _agg_gap(lrt_lo, "LRT_lower")

    lrt_hi = lrt.dropna(subset=["eps_upper"]).copy()
    lrt_hi["gap"] = lrt_hi["upper"] - lrt_hi["eps_upper"]
    g_lrt_hi = _agg_gap(lrt_hi, "LRT_upper")

    # ===== LRT_decomp =====
    ind = df0[df0["protocol"] == "LRT_indirect"][
        ["seed", xcol, "eps_lower", "eps_upper"]
    ].copy()
    ind = ind.merge(upper, on=["seed", xcol], how="inner")

    ind_lo = ind.dropna(subset=["eps_lower"]).copy()
    ind_lo["gap"] = ind_lo["upper"] - ind_lo["eps_lower"]
    g_ind_lo = _agg_gap(ind_lo, "DECOMP_lower")

    ind_hi = ind.dropna(subset=["eps_upper"]).copy()
    ind_hi["gap"] = ind_hi["upper"] - ind_hi["eps_upper"]
    g_ind_hi = _agg_gap(ind_hi, "DECOMP_upper")

    agg = pd.concat([g_lrt_lo, g_lrt_hi, g_ind_lo, g_ind_hi], ignore_index=True)
    agg = agg.sort_values(xcol)

    # --- plotting ---
    ax.axhline(0.0, linestyle="--", linewidth=1.0)
    if is_N and N_guideline is not None:
        ax.axvline(
            float(N_guideline), linestyle="--", linewidth=1.2, color="tab:orange"
        )
    if (not is_N) and c_boundary is not None:
        ax.axvline(float(c_boundary), linestyle="--", linewidth=1.2, color="tab:orange")

    ax.grid(True, linestyle="dashdot", linewidth=0.5)

    # LRT (blue)
    s = agg[agg["series"] == "LRT_lower"]
    ln = ax.plot(
        s[xcol],
        s["mean"],
        linestyle="-",
        marker="v",
        label=r"LRT: Upper $-$ $\hat{\varepsilon}_\mathrm{lower}$",
    )
    c_lrt = ln[0].get_color()
    ax.errorbar(
        s[xcol], s["mean"], yerr=s["std"], linestyle="None", capsize=3, color=c_lrt
    )

    s2: pd.DataFrame = agg[agg["series"] == "LRT_upper"]
    ax.plot(
        s2[xcol],
        s2["mean"],
        linestyle=":",
        marker="^",
        color=c_lrt,
        label=r"LRT: Upper $-$ $\hat{\varepsilon}_\mathrm{upper}$",
    )
    ax.errorbar(
        s2[xcol], s2["mean"], yerr=s2["std"], linestyle="None", capsize=3, color=c_lrt
    )

    # LRT_decomp (purple)
    base = "tab:purple"
    s3: pd.DataFrame = agg[agg["series"] == "DECOMP_lower"]
    ax.plot(
        s3[xcol],
        s3["mean"],
        linestyle="-",
        marker="v",
        color=base,
        linewidth=2.5,
        label=r"LRT_Decomp: Upper $-$ $\hat{\varepsilon}_\mathrm{lower}$",
    )
    ax.errorbar(
        s3[xcol], s3["mean"], yerr=s3["std"], linestyle="None", capsize=3, color=base
    )

    s4: pd.DataFrame = agg[agg["series"] == "DECOMP_upper"]
    ax.plot(
        s4[xcol],
        s4["mean"],
        linestyle=":",
        marker="^",
        color=base,
        label=r"LRT_Decomp: Upper $-$ $\hat{\varepsilon}_\mathrm{upper}$",
    )
    ax.errorbar(
        s4[xcol], s4["mean"], yerr=s4["std"], linestyle="None", capsize=3, color=base
    )

    ax.set_xscale("log")
    if fixed_value is not None and fixed_label is not None:
        ax.set_xlabel(f"{xcol} (fixed {fixed_label} = {fixed_value:g})", fontsize=25)
    else:
        ax.set_xlabel(xcol, fontsize=25)


    h, l = ax.get_legend_handles_labels()
    return h, l


def plot_N_c_sweep_2x2(
    *,
    csv_N: str,
    csv_c: str,
    outpath: str,
    epsilon_theoretical: float = 1.0,
    k: int = 2,
    c_for_guideline: float = 1e-2,
    alpha: float = 1e-2,
    omega: float = 0.005,
    shift_list: list[float] | None = None,
) -> None:
    dfN = pd.read_csv(csv_N)
    dfC = pd.read_csv(csv_c)

    if shift_list is None:
        shift_list = sorted(
            set(dfN["mean_shift"].dropna().astype(float))
            & set(dfC["mean_shift"].dropna().astype(float))
        )
    shift_list = shift_list[:2]  # [0.1, 0.5]

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.2), sharey=True)
    axes = np.atleast_2d(axes)

    all_handles, all_labels = [], []

    for c_idx, shift in enumerate(shift_list):
        # ---- row 0: N-sweep ----
        h, l = _plot_sweep_gap_panel(
            axes[0, c_idx],
            dfN,
            is_N=True,
            mean_shift=float(shift),
            epsilon_theoretical=epsilon_theoretical,
            k=k,
            c_for_guideline=c_for_guideline,
            alpha=alpha,
            omega=omega,
        )
        axes[0, c_idx].set_title(f"mean_shift = {shift}", fontsize=22)
        axes[1, c_idx].set_title(f"mean_shift = {shift}", fontsize=22)

        # ---- row 1: c-sweep ----
        h2, l2 = _plot_sweep_gap_panel(
            axes[1, c_idx],
            dfC,
            is_N=False,
            mean_shift=float(shift),
            epsilon_theoretical=epsilon_theoretical,
            k=k,
            c_for_guideline=c_for_guideline,
            alpha=alpha,
            omega=omega,
        )

        for hh, ll in zip(h + h2, l + l2):
            if ll not in all_labels:
                all_handles.append(hh)
                all_labels.append(ll)

    # row labels
    axes[0, 0].set_ylabel(r"Upper $-$ $\hat{\varepsilon}_\mathrm{CI}$", fontsize=22)
    axes[1, 0].set_ylabel(r"Upper $-$ $\hat{\varepsilon}_\mathrm{CI}$", fontsize=22)

    # legend
    fig.legend(
        all_handles,
        all_labels,
        loc="upper center",
        ncol=2,
        bbox_to_anchor=(0.5, 1.02),
        frameon=True,
        fontsize=21,
    )

    fig.tight_layout(rect=(0.04, 0.0, 1.0, 0.90))
    Path(outpath).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.show()


def _main_cli() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    # ---- subcommand: lrt_compare（今まで通り） ----
    p1: argparse.ArgumentParser = sub.add_parser("lrt_compare")
    p1.add_argument(
        "-csv_list",
        "--csv_list",
        nargs="+",
        required=True,
        help="CSV files (exactly 4) for the 2x2 panels.",
    )
    p1.add_argument(
        "--titles",
        nargs="*",
        default=None,
        help="Optional 4 titles for panels (same order as csv_list).",
    )
    p1.add_argument(
        "--out",
        default="results/fig_lrt_compare_2x2.pdf",
        help="Output pdf path.",
    )
    p1.add_argument(
        "--k",
        type=int,
        default=None,
        help="Domain size k. If omitted, inferred from first CSV.",
    )
    p1.add_argument(
        "--yscale",
        choices=["log", "linear"],
        default="log",
        help="Y-axis scale.",
    )

    # ---- subcommand: audit_2x2（新規） ----
    p2 = sub.add_parser("audit_2x2")
    p2.add_argument(
        "-csv_list",
        "--csv_list",
        nargs="+",
        required=True,
        help="CSV files (exactly 4) for the 2x2 panels.",
    )
    p2.add_argument(
        "--titles",
        nargs="*",
        default=None,
        help="Optional 4 titles for panels (same order as csv_list).",
    )
    p2.add_argument(
        "--out",
        default="results/fig_results_summary_audit_2x2.pdf",
        help="Output pdf path.",
    )
    p2.add_argument(
        "--epsilon",
        type=float,
        default=1.0,
        help="Fixed theoretical epsilon to visualize (e.g., 1.0).",
    )
    p2.add_argument(
        "--k",
        type=int,
        default=None,
        help="Domain size k. If omitted, inferred from first CSV.",
    )

    # ---- subcommand: N_c_sweep（新規） ----
    p3 = sub.add_parser("N_c_sweep")
    p3.add_argument(
        "--csv_N",
        default="./results/20260116/N_sweep_eps=1.0_c=0.01_k=2_seeds=5.csv",
        help="N-sweep result csv path.",
    )
    p3.add_argument(
        "--csv_c",
        default="./results/20260116/c_sweep_N=500000_eps=1.0_k=2_seeds=5.csv",
        help="c-sweep result csv path.",
    )
    p3.add_argument(
        "--out",
        default="results/fig_N_c_sweep_2x2.pdf",
        help="Output pdf path.",
    )
    p3.add_argument("--epsilon", type=float, default=1.0)
    p3.add_argument("--k", type=int, default=2)
    p3.add_argument("--alpha", type=float, default=1e-2)
    p3.add_argument("--omega", type=float, default=0.005)
    p3.add_argument(
        "--c_for_guideline",
        type=float,
        default=1e-2,
        help="For N-sweep guideline line: N_guideline(c_for_guideline).",
    )
    p3.add_argument(
        "--shifts",
        type=float,
        nargs="*",
        default=None,
        help="Optional mean_shift list (use first two for 2x2). e.g. --shifts 0.1 0.5",
    )

    args: argparse.Namespace = parser.parse_args()

    if args.cmd == "lrt_compare":
        csvs: list[str] = args.csv_list
        if len(csvs) != 4:
            raise ValueError("Please pass exactly 4 csv files for 2x2.")
        titles = args.titles
        if titles is not None and len(titles) != 4:
            raise ValueError("--titles must have exactly 4 strings when provided.")
        plot_lrt_compare_2x2(
            csv_paths=csvs,
            outpath=args.out,
            panel_titles=titles,
            k=args.k,
            yscale=args.yscale,
        )

    elif args.cmd == "audit_2x2":
        csvs: list[str] = args.csv_list
        if len(csvs) != 4:
            raise ValueError("Please pass exactly 4 csv files for 2x2.")
        titles = args.titles
        if titles is not None and len(titles) != 4:
            raise ValueError("--titles must have exactly 4 strings when provided.")
        plot_results_example_audit_2x2(
            csv_paths=csvs,
            outpath=args.out,
            panel_titles=titles,
            epsilon=args.epsilon,
            k=args.k,
            lst_protocol=["GRR", "LRT"],
        )
    elif args.cmd == "N_c_sweep":
        plot_N_c_sweep_2x2(
            csv_N=args.csv_N,
            csv_c=args.csv_c,
            outpath=args.out,
            epsilon_theoretical=float(args.epsilon),
            k=int(args.k),
            c_for_guideline=float(args.c_for_guideline),
            alpha=float(args.alpha),
            omega=float(args.omega),
            shift_list=(list(args.shifts) if args.shifts is not None else None),
        )


if __name__ == "__main__":
    _main_cli()

# python plot_functions.py lrt_compare \
#   -csv_list ./results/20251217/shift=0.1_c=1e-2.csv ./results/20251217/shift=0.1_c=1e-3.csv ./results/20251217/shift=2.0_c=1e-2.csv ./results/20251217/shift=2.0_c=1e-3.csv \
#   --titles "mean_shift=0.1, c=1e-2" "mean_shift=0.1, c=1e-3" "mean_shift=2.0, c=1e-2" "mean_shift=2.0, c=1e-3" \
#   --out results/fig_lrt_compare_2x2.pdf \
#   --k 2 \
#   --yscale log

# python plot_functions.py audit_2x2 \
#   -csv_list ./results/20251217/shift=0.1_c=1e-2.csv ./results/20251217/shift=0.1_c=1e-3.csv ./results/20251217/shift=2.0_c=1e-2.csv ./results/20251217/shift=2.0_c=1e-3.csv \
#   --titles "mean_shift=0.1, c=1e-2" "mean_shift=0.1, c=1e-3" "mean_shift=2.0, c=1e-2" "mean_shift=2.0, c=1e-3" \
#   --out results/fig_results_summary_audit_2x2.pdf \
#   --epsilon 1.0 \
#   --k 2

# python plot_functions.py lrt_compare \
#   -csv_list \
#     ./results/20260207/shift=0.1_c=0.01_decomp.csv \
#     ./results/20260207/shift=0.1_c=0.001_decomp.csv \
#     ./results/20260207/shift=2.0_c=0.01_decomp.csv \
#     ./results/20260207/shift=2.0_c=0.001_decomp.csv \
#   --titles "mean_shift=0.1, c=1e-2" "mean_shift=0.1, c=1e-3" "mean_shift=2.0, c=1e-2" "mean_shift=2.0, c=1e-3" \
#   --out results/fig_lrt_compare_2x2.pdf \
#   --k 2 \
#   --yscale log

# python plot_functions.py N_c_sweep \
#   --csv_N ./results/20260116/N_sweep_eps=1.0_c=0.01_k=2_seeds=5.csv \
#   --csv_c ./results/20260116/c_sweep_N=500000_eps=1.0_k=2_seeds=5.csv \
#   --out results/fig_N_c_sweep_2x2.pdf \
#   --shifts 0.1 0.5
