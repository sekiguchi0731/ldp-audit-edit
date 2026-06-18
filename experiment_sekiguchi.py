# general imports
import argparse
import logging
import os
from pathlib import Path
from typing import Any, Iterable, List, Sequence, TypedDict

import numpy as np
import pandas as pd
from tqdm import tqdm

# Our imports
from ldp_audit.base_auditor import LDPAuditor
from ldp_audit.simulation import (
    MixtureSpec,
    log_Rmax_complete_gaussian,
    log_Rmax_gaussian,
    set_B_from_quantile,
)

# ---------------------------------------------------------------------
# Logging / IO setup
# ---------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
os.makedirs('results', exist_ok=True)

# ---------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------
class ResultsDict(TypedDict):
    """結果CSVへ書き出す各列の型"""

    seed: List[int]
    protocol: List[str]
    k: List[int]  # カテゴリ数（ラベルの種類数）
    delta: List[float]
    epsilon: List[float]
    eps_emp: List[float]  # 監査で得た経験的 ε（スカラー）
    logRmax_eff: List[float]


# ---------------------------------------------------------------------
# Main audit results
# ---------------------------------------------------------------------
def run_main_experiments(
    nb_trials: int,
    alpha: float,
    lst_protocols: Sequence[str],
    lst_seed: Iterable[int],
    lst_k: Sequence[int],
    lst_eps: Sequence[float],
    delta: float,
    analysis: str,
    c: float = 1e-6,
    shift: float = 2,
) -> pd.DataFrame:
    """
    乱数シード・カテゴリ数 k・ε（および δ）・プロトコルの組み合わせを受け取り
    LDP 監査を実行し，経験的な ε（eps_emp）を集計して CSV に保存し，
    同じ内容を pd.DataFrame として返す関数

    Parameters
    ----------
    nb_trials : int
        監査内部で用いる試行回数（モンテカルロ反復やサンプル生成回数など）
        大きいほど推定精度は上がるが計算コストが増加
    alpha : float
        有意水準（監査の統計的判定で使用）.例: 1e-2 (= 0.01)
    lst_protocols : Sequence[str]
        評価対象の LDP プロトコル名（例: 'GRR', 'OUE', 'BLH' など）の列
        Sequence は list, tuple, str, ndarray などの，抽象的な「順序付きコレクション」
        Iterable とは異なり，インデックスアクセスや len() が可能
    lst_seed : Iterable[int]
        乱数シードの集まり（range や list）.再現性・分散評価のために複数指定可能
        Iterable は, for で回せるオブジェクトのこと
        list, tuple, set, dict, str, range, generator などが該当
    lst_k : Sequence[int]
        カテゴリ数 k の候補（例: [25, 50, 100, ...]）。
    lst_eps : Sequence[float]
        プライバシー予算 ε の候補（例: [0.25, 0.5, 1, 2, ...]）。
    delta : float
        近似 DP の δ。純粋 DP の場合は 0.0。
    analysis : str
        出力 CSV のファイル名識別子（例: 'main_pure_ldp_protocols'）。
    c : float
        FPR/TPR を下駄ばきする下限。小さくし過ぎると試行回数が膨らむので注意。

    Returns
    -------
    pd.DataFrame
        列: ['seed','protocol','k','delta','epsilon','eps_emp'] を持つ結果データフレーム。
        同内容を `results/ldp_audit_results_{analysis}.csv` にも保存。
    """
    # Initialize dictionary to save results
    results: dict[str, list] = {
        "seed": [],
        "protocol": [],
        "k": [],
        "delta": [],
        "epsilon": [],
        "eps_emp": [],
        "logRmax_eff": [],
        "eps_lower": [],
        "eps_upper": [],
    }

    # Initialize LDP-Auditor（初期値は後で set_params で上書きする）
    # --- ここでシミュレーションから η を生成し、監査器へ渡す ---
    spec = MixtureSpec(num_classes=2, d=2, sigma=1.0, mean_shift=shift)
    auditor: LDPAuditor = LDPAuditor(
        nb_trials=nb_trials,
        alpha=alpha,
        epsilon=lst_eps[0],
        delta=delta,
        k=lst_k[0],
        random_state=next(iter(lst_seed)) if hasattr(lst_seed, "__iter__") else 0,
        n_jobs=-1,
        rmax_alpha=0.01,
        c=c,
        spec=spec,
    )

    # Run the experiments
    for seed in lst_seed:    
        logging.info(f"Running experiments for seed: {seed}")
        for k in lst_k:
            for epsilon in lst_eps:
                # Update the auditor parameters
                auditor.set_params(epsilon=epsilon, k=k, random_state=seed)

                for protocol in tqdm(lst_protocols, desc=f'seed={seed}, k={k}, epsilon={epsilon}'):
                    eps_emp: float = auditor.run_audit(protocol)
                    logRmax_eff_seed: float | None = getattr(auditor, "logRmax_eff", None)
                    # 追加：LRT 以外は None のままでOK
                    eps_ci = getattr(auditor, "eps_ci", (None, None))
                    eps_lower_seed, eps_upper_seed = eps_ci
                    results['seed'].append(seed)
                    results['protocol'].append(protocol)
                    results['k'].append(k)
                    results['delta'].append(delta)
                    results['epsilon'].append(epsilon)
                    results['eps_emp'].append(eps_emp)
                    results['eps_lower'].append(eps_lower_seed)
                    results['eps_upper'].append(eps_upper_seed)
                    results['logRmax_eff'].append(logRmax_eff_seed)

    # Convert and save results in csv file
    df = pd.DataFrame(results)
    print('==== Results DataFrame ====')
    print('df : \n', df)
    output_path = Path(f"{analysis}.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    
    return df


def _format_float_for_name(value: float) -> str:
    return f"{float(value):g}".replace("-", "m").replace(".", "p")


def _parse_four_way_ratio(values: Sequence[str | float]) -> tuple[float, float, float, float]:
    if len(values) == 1 and isinstance(values[0], str) and "," in values[0]:
        parts: List[str] = [part.strip() for part in values[0].split(",") if part.strip()]
    else:
        parts = [str(value).strip() for value in values]
    if len(parts) != 4:
        raise ValueError(
            "--n_ratio must contain four values: train,val,threshold,eval. "
            "Example: --n_ratio 0.2,0.2,0.2,0.6"
        )
    ratios: tuple[float, ...] = tuple(float(part) for part in parts)
    if any(value < 0.0 for value in ratios):
        raise ValueError(f"--n_ratio entries must be nonnegative, got {ratios}.")
    total = float(sum(ratios))
    if total <= 0.0:
        raise ValueError("--n_ratio must have positive sum.")
    normalized: tuple[float, ...] = tuple(float(value / total) for value in ratios)
    if normalized[3] <= 0.0:
        raise ValueError("--n_ratio eval share, the fourth entry, must be positive.")
    return normalized  # type: ignore[return-value]


def _split_counts_from_eval_n(
    eval_n_per_class: int,
    split_ratio: tuple[float, float, float, float],
) -> tuple[int, int, int, int, int]:
    eval_n = int(eval_n_per_class)
    if eval_n <= 0:
        raise ValueError("eval_n_per_class must be positive.")
    total_per_class = int(np.ceil(eval_n / split_ratio[3]))
    train_n = int(np.ceil(total_per_class * split_ratio[0]))
    val_n = int(np.ceil(total_per_class * split_ratio[1]))
    threshold_n = int(np.ceil(total_per_class * split_ratio[2]))
    return train_n, val_n, threshold_n, eval_n, total_per_class


def _n_theory_from_target_eta(
    *,
    target_eta: float,
    psi_star: float,
    beta: float,
    constant: float = 1.0,
) -> float:
    eta = float(target_eta)
    psi = float(psi_star)
    beta_float = float(beta)
    constant_float = float(constant)
    if not (0.0 < eta < 1.0):
        raise ValueError(f"target_eta must be in (0, 1), got {target_eta}.")
    if psi <= 0.0 or not np.isfinite(psi):
        raise ValueError(f"psi_star must be positive and finite, got {psi_star}.")
    if not (0.0 < beta_float < 1.0):
        raise ValueError(f"beta must be in (0, 1), got {beta}.")
    if constant_float <= 0.0 or not np.isfinite(constant_float):
        raise ValueError(f"constant must be positive and finite, got {constant}.")
    return float((constant_float / ((eta**2) * (psi**2))) ** (1.0 / (1.0 - beta_float)))


def run_sample_complexity_experiments(
    *,
    n_values: Sequence[int] | None,
    beta_values: Sequence[float],
    shift_values: Sequence[float],
    epsilon_values: Sequence[float],
    seeds: Iterable[int],
    alpha: float,
    rmax_alpha: float,
    output_path: str | Path,
    sigma: float = 1.0,
    d: int = 2,
    lipschitz_c_floor: float | None = None,
    target_eta_values: Sequence[float] | None = None,
    n_eval_ratios: Sequence[float] | None = None,
    n_split_ratio: tuple[float, float, float, float] = (
        1.0 / 6.0,
        1.0 / 6.0,
        1.0 / 6.0,
        0.5,
    ),
    n_theory_constant: float = 1.0,
    max_grid_size: int | None = None,
) -> pd.DataFrame:
    """
    Validate the finite-grid CP-LCB decomp sample-complexity prediction on
    simulation data using the true Gaussian LR score.
    """
    rows: dict[str, list] = {
        "seed": [],
        "n_min": [],
        "n_theory": [],
        "n_theory_constant": [],
        "n_theory_ceil": [],
        "n_eval_ratio_requested": [],
        "n_eval_ratio": [],
        "target_eta": [],
        "beta": [],
        "c_n": [],
        "h_n": [],
        "L_log": [],
        "L_tail": [],
        "lipschitz_c_floor": [],
        "alpha": [],
        "alpha_cp": [],
        "rmax_alpha": [],
        "shift": [],
        "sigma": [],
        "d": [],
        "epsilon": [],
        "psi_star_decomp": [],
        "psi_star_decomp_total": [],
        "psi_star_complete": [],
        "logRmax_eff": [],
        "target_decomp": [],
        "eps_emp_x": [],
        "eps_lower_x": [],
        "eps_upper_x": [],
        "eps_emp": [],
        "eps_lower": [],
        "eps_upper": [],
        "gap_lower": [],
        "rel_gap_lower": [],
        "relative_error": [],
        "relative_error_total": [],
        "target_met": [],
        "best_direction": [],
        "tau": [],
        "tau_grid_size": [],
        "tau_grid_feasible_size": [],
        "total_cp_events": [],
        "ratio_train": [],
        "ratio_val": [],
        "ratio_threshold": [],
        "ratio_eval": [],
        "n_train_per_class": [],
        "n_val_per_class": [],
        "n_threshold_per_class": [],
        "n_eval_per_class": [],
        "n_total_per_class": [],
        "n_total_all_classes": [],
    }

    seed_list: List[int] = list(seeds)
    if target_eta_values is not None:
        eta_grid: list[float | None] = [float(value) for value in target_eta_values]
        eval_ratio_grid: list[float | None] = [
            float(value) for value in (n_eval_ratios if n_eval_ratios is not None else [0.25, 0.5, 1.0, 2.0, 4.0, 8.0])
        ]
        if any(value <= 0.0 for value in eval_ratio_grid if value is not None):
            raise ValueError(f"n_eval_ratios must be positive, got {eval_ratio_grid}.")
    else:
        eta_grid = [None]
        eval_ratio_grid = [None]
        if n_values is None:
            raise ValueError("Either target_eta_values or n_values must be provided.")

    n_value_grid: list[int] = [] if n_values is None else [int(value) for value in n_values]
    ratio_train, ratio_val, ratio_threshold, ratio_eval = n_split_ratio

    for shift in shift_values:
        spec = MixtureSpec(num_classes=2, d=int(d), sigma=float(sigma), mean_shift=float(shift))
        B_theory: float = set_B_from_quantile(spec, alpha=float(rmax_alpha))
        psi_star_decomp: float = float(log_Rmax_gaussian(spec, B=B_theory))
        for beta in beta_values:
            for target_eta in eta_grid:
                if target_eta is None:
                    derived_n_items: list[tuple[int, float, float, float | None]] = [
                        (int(n), float("nan"), float("nan"), None) for n in n_value_grid
                    ]
                else:
                    n_theory: float = _n_theory_from_target_eta(
                        target_eta=float(target_eta),
                        psi_star=psi_star_decomp,
                        beta=float(beta),
                        constant=float(n_theory_constant),
                    )
                    n_theory_ceil = float(np.ceil(n_theory))
                    derived_n_items = []
                    for eval_ratio in eval_ratio_grid:
                        assert eval_ratio is not None
                        n_eval: int = max(1, int(np.ceil(n_theory * float(eval_ratio))))
                        actual_ratio: float = float(n_eval / n_theory) if n_theory > 0.0 else float("nan")
                        derived_n_items.append((n_eval, n_theory, actual_ratio, float(eval_ratio)))

                for n_int, n_theory_value, actual_eval_ratio, requested_eval_ratio in derived_n_items:
                    train_n, val_n, threshold_n, eval_n, total_per_class = _split_counts_from_eval_n(
                        eval_n_per_class=n_int,
                        split_ratio=n_split_ratio,
                    )
                    for seed in seed_list:
                        auditor = LDPAuditor(
                            nb_trials=eval_n,
                            alpha=float(alpha),
                            epsilon=float(epsilon_values[0]),
                            delta=0.0,
                            k=2,
                            random_state=int(seed),
                            n_jobs=1,
                            rmax_alpha=float(rmax_alpha),
                            c=1e-12,
                            spec=spec,
                            dynamic_nb_trials=False,
                        )
                        for epsilon in epsilon_values:
                            auditor.set_params(
                                nb_trials=eval_n,
                                epsilon=float(epsilon),
                                random_state=int(seed),
                            )
                            psi_star_complete = float(
                                log_Rmax_complete_gaussian(
                                    spec,
                                    epsilon=float(epsilon),
                                    B=B_theory,
                                )
                            )
                            psi_star_decomp_total = float(psi_star_decomp + float(epsilon))
                            logging.info(
                                "finite-grid decomp sample-complexity: shift=%s n=%s "
                                "seed=%s epsilon=%s beta=%s target_eta=%s n_ratio=%s",
                                shift,
                                eval_n,
                                seed,
                                epsilon,
                                beta,
                                target_eta,
                                requested_eval_ratio,
                            )
                            res: dict[str, Any] = auditor.evaluate_decomp_finite_grid_cp(
                                beta=float(beta),
                                failure_delta=float(alpha),
                                lipschitz_c_floor=lipschitz_c_floor,
                                n_plus_eval=eval_n,
                                n_minus_eval=eval_n,
                                max_grid_size=max_grid_size,
                            )
                            target_decomp: float = psi_star_decomp_total
                            eps_lower_x = float(res["eps_lower_x"])
                            eps_lower = float(res["eps_lower"])
                            gap_lower: float = max(target_decomp - eps_lower, 0.0)
                            rel_gap_lower: float = (
                                gap_lower / target_decomp
                                if target_decomp > 0.0 and np.isfinite(gap_lower)
                                else np.nan
                            )
                            relative_error: float = (
                                max(psi_star_decomp - eps_lower_x, 0.0) / psi_star_decomp
                                if psi_star_decomp > 0.0 and np.isfinite(eps_lower_x)
                                else np.nan
                            )
                            relative_error_total = (
                                max(psi_star_decomp_total - eps_lower, 0.0) / psi_star_decomp_total
                                if psi_star_decomp_total > 0.0 and np.isfinite(eps_lower)
                                else np.nan
                            )
                            target_met = (
                                bool(relative_error <= float(target_eta))
                                if target_eta is not None and np.isfinite(relative_error)
                                else False
                            )

                            rows["seed"].append(int(seed))
                            rows["n_min"].append(int(res["n_min"]))
                            rows["n_theory"].append(float(n_theory_value))
                            rows["n_theory_constant"].append(float(n_theory_constant))
                            rows["n_theory_ceil"].append(
                                float(np.ceil(n_theory_value)) if np.isfinite(n_theory_value) else np.nan
                            )
                            rows["n_eval_ratio_requested"].append(
                                float(requested_eval_ratio) if requested_eval_ratio is not None else np.nan
                            )
                            rows["n_eval_ratio"].append(float(actual_eval_ratio))
                            rows["target_eta"].append(float(target_eta) if target_eta is not None else np.nan)
                            rows["beta"].append(float(beta))
                            rows["c_n"].append(float(res["c_n"]))
                            rows["h_n"].append(float(res["h_n"]))
                            rows["L_log"].append(float(res["L_log"]))
                            rows["L_tail"].append(float(res["L_tail"]))
                            rows["lipschitz_c_floor"].append(float(res["lipschitz_c_floor"]))
                            rows["alpha"].append(float(alpha))
                            rows["alpha_cp"].append(float(res["alpha_cp"]))
                            rows["rmax_alpha"].append(float(rmax_alpha))
                            rows["shift"].append(float(shift))
                            rows["sigma"].append(float(sigma))
                            rows["d"].append(int(d))
                            rows["epsilon"].append(float(epsilon))
                            rows["psi_star_decomp"].append(float(psi_star_decomp))
                            rows["psi_star_decomp_total"].append(float(psi_star_decomp_total))
                            rows["psi_star_complete"].append(float(psi_star_complete))
                            rows["logRmax_eff"].append(float(auditor.logRmax_eff))
                            rows["target_decomp"].append(target_decomp)
                            rows["eps_emp_x"].append(float(res["eps_emp_x"]))
                            rows["eps_lower_x"].append(eps_lower_x)
                            rows["eps_upper_x"].append(float(res["eps_upper_x"]))
                            rows["eps_emp"].append(float(res["eps_emp"]))
                            rows["eps_lower"].append(eps_lower)
                            rows["eps_upper"].append(float(res["eps_upper"]))
                            rows["gap_lower"].append(float(gap_lower))
                            rows["rel_gap_lower"].append(float(rel_gap_lower))
                            rows["relative_error"].append(float(relative_error))
                            rows["relative_error_total"].append(float(relative_error_total))
                            rows["target_met"].append(bool(target_met))
                            rows["best_direction"].append(str(res["best_direction"]))
                            rows["tau"].append(float(res["tau"]))
                            rows["tau_grid_size"].append(int(res["tau_grid_size"]))
                            rows["tau_grid_feasible_size"].append(int(res["tau_grid_feasible_size"]))
                            rows["total_cp_events"].append(int(res["total_cp_events"]))
                            rows["ratio_train"].append(float(ratio_train))
                            rows["ratio_val"].append(float(ratio_val))
                            rows["ratio_threshold"].append(float(ratio_threshold))
                            rows["ratio_eval"].append(float(ratio_eval))
                            rows["n_train_per_class"].append(int(train_n))
                            rows["n_val_per_class"].append(int(val_n))
                            rows["n_threshold_per_class"].append(int(threshold_n))
                            rows["n_eval_per_class"].append(int(eval_n))
                            rows["n_total_per_class"].append(int(total_per_class))
                            rows["n_total_all_classes"].append(int(2 * total_per_class))

    df = pd.DataFrame(rows)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print("==== Sample Complexity Results DataFrame ====")
    print(df)
    print(f"Saved sample-complexity results to {output_path}")
    return df


# ---------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_complexity", action="store_true")
    parser.add_argument("--shift_values", type=float, nargs="+", default=[0.1, 0.25, 0.5, 1.0])
    parser.add_argument("--epsilon_values", type=float, nargs="+", default=[0.25, 0.5, 0.75, 1, 2, 4, 6, 10])
    parser.add_argument("--seed_start", type=int, default=0)
    parser.add_argument("--seed_end", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--nb_trials", type=int, default=int(1e6))
    parser.add_argument("--c_values", type=float, nargs="+", default=[1e-2])
    parser.add_argument("--n_values", type=int, nargs="+", default=[1000, 2000, 5000])
    parser.add_argument("--beta_values", type=float, nargs="+", default=[0.25, 0.5, 0.75])
    parser.add_argument("--target_eta_values", type=float, nargs="+", default=None)
    parser.add_argument("--n_eval_ratios", type=float, nargs="+", default=[0.25, 0.5, 1.0, 2.0, 4.0, 8.0])
    parser.add_argument(
        "--n_ratio",
        nargs="+",
        default=["0.2,0.2,0.2,0.6"],
        help="Relative train,val,threshold,eval split. Example: --n_ratio 0.2,0.2,0.2,0.6",
    )
    parser.add_argument("--sim_sigma", type=float, default=1.0)
    parser.add_argument("--sim_d", type=int, default=2)
    parser.add_argument("--rmax_alpha", type=float, default=1e-6)
    parser.add_argument("--lipschitz_c_floor", type=float, default=None)
    parser.add_argument(
        "--n_theory_constant",
        type=float,
        default=1.0,
        help=(
            "Positive constant C in "
            "n_theory=(C/(target_eta^2 * psi_star^2))^(1/(1-beta))."
        ),
    )
    parser.add_argument("--max_grid_size", type=int, default=None)
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--skip_plots", action="store_true")
    args: argparse.Namespace = parser.parse_args()

    seeds = range(int(args.seed_start), int(args.seed_end))
    n_split_ratio: tuple[float, float, float, float] = _parse_four_way_ratio(args.n_ratio)

    if args.sample_complexity:
        if args.output_path is None:
            shift_name: str = "_".join(_format_float_for_name(x) for x in args.shift_values)
            beta_name: str = "_".join(_format_float_for_name(x) for x in args.beta_values)
            output_path: str = (
                f"results/sample_complexity/decomp_grid_shift={shift_name}"
                f"_beta={beta_name}_alpha={_format_float_for_name(args.alpha)}.csv"
            )
        else:
            output_path = args.output_path
        run_sample_complexity_experiments(
            n_values=None if args.target_eta_values is not None else args.n_values,
            beta_values=args.beta_values,
            shift_values=args.shift_values,
            epsilon_values=args.epsilon_values,
            seeds=seeds,
            alpha=float(args.alpha),
            rmax_alpha=float(args.rmax_alpha),
            output_path=output_path,
            sigma=float(args.sim_sigma),
            d=int(args.sim_d),
            lipschitz_c_floor=args.lipschitz_c_floor,
            target_eta_values=args.target_eta_values,
            n_eval_ratios=args.n_eval_ratios,
            n_split_ratio=n_split_ratio,
            n_theory_constant=float(args.n_theory_constant),
            max_grid_size=args.max_grid_size,
        )
        raise SystemExit(0)

    for shift_temp in args.shift_values:
        for c_temp in args.c_values:
            print(f"===== Running experiments for shift={shift_temp}, c={c_temp} =====")
            ## General parameters
            lst_eps: list[float] = [float(x) for x in args.epsilon_values]
            lst_k: list[int] = [2]
            lst_seed: range = seeds
            nb_trials = int(args.nb_trials)
            shift: float = shift_temp
            alpha: float = float(args.alpha)
            c: float = c_temp

            ## pure LDP protocols
            pure_ldp_protocols: list[str] = [
                "GRR",
                "LRT",
                "LRT_decomp",
            ]
            delta: float = 0.0
            analysis_pure: str = f"results/20260326/shift={shift}_c={c}_decomp"
            df_pure: pd.DataFrame = run_main_experiments(
                nb_trials,
                alpha,
                pure_ldp_protocols,
                lst_seed,
                lst_k,
                lst_eps,
                delta,
                analysis_pure,
                c=c,
                shift=shift,
            )
            if not args.skip_plots:
                from plot_functions import (
                    plot_results_example_audit,
                    plot_results_pure_ldp_protocols,
                )

                # Figure 1: Example of audit results for pure LDP protocols
                plot_results_example_audit(
                    df_pure, pure_ldp_protocols, epsilon=0.5, k=2
                )  # Example of audit results -- Figure 1 in paper
                # Figure 2: Main results for pure LDP protocols
                plot_results_pure_ldp_protocols(
                    df_pure, analysis_pure, pure_ldp_protocols, lst_eps, lst_k
                )  # Main results -- Figure 2 in paper
