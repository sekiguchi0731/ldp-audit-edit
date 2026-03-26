# experiment_eta_model.py

from __future__ import annotations

import argparse

import json
import logging
from dataclasses import dataclass
from typing import Iterable, Literal, cast

import numpy as np
import pandas as pd
from tqdm import tqdm

from ldp_audit.base_auditor import LDPAuditor
from ldp_audit.eta_models import (
    EtaModelConfig,
    get_default_eta_model_configs,
    get_fixed_logreg_eta_model_configs,
    get_reduced_eta_model_configs,
)
from ldp_audit.simulation import MixtureSpec

# 既に追加したはずの関数（base_auditor.py 内のメソッド）
# - auditor.run_eta_model_comparison_4way(...)


# ============================================================
# Result schema (long format)
# ============================================================


@dataclass(frozen=True)
class EtaExperimentConfig:
    """eta-model comparison experiment config (simulation / real-data compatible)."""

    # audit params
    nb_trials: int
    alpha: float
    c: float
    epsilon_list: tuple[float, ...] = (1.0,)
    delta: float = 0.0
    k: int = 2
    y_alt: int = 1
    y_null: int = 0

    # selection/report config
    selection: Literal["eps_lower", "eps_emp", "tpr_at_fpr"] = "eps_lower"
    attack_for_selection: Literal["indirect_LRT_hat", "complete_LRT_hat"] = (
        "complete_LRT_hat"
    )
    # If True, evaluate both complete/indirect attacks with attack-specific tau,q.
    evaluate_both_reports: bool = True
    use_reduced_grid: bool = True
    hyperparameter: Literal["grid", "fixed"] = "grid"

    # simulation dataset config
    sim_d: int = 2
    sim_sigma: float = 1.0
    sim_mean_shift: float = 2.0
    sim_rmax_alpha: float = 1e-6
    sim_n_train: int = 4000
    sim_n_val: int = 2000

    # output
    analysis: str = "eta_model_comparison"


def _append_metric(
    rows: dict[str, list],
    *,
    seed: int,
    protocol: str,
    k: int,
    delta: float,
    epsilon: float,
    c: float,
    nb_trials: int,
    alpha: float,
    selection: str,
    attack_for_selection: str,
    attack_for_report: str,
    sim_d: int | None,
    sim_sigma: float | None,
    sim_mean_shift: float | None,
    sim_n_train: int | None,
    sim_n_val: int | None,
    metric: str,
    value: float | int | None,
    params_json: str | None,
) -> None:
    rows["seed"].append(seed)
    rows["protocol"].append(protocol)
    rows["k"].append(k)
    rows["delta"].append(delta)
    rows["epsilon"].append(epsilon)

    rows["c"].append(c)
    rows["nb_trials"].append(nb_trials)
    rows["alpha"].append(alpha)

    rows["selection"].append(selection)
    rows["attack_for_selection"].append(attack_for_selection)
    rows["attack_for_report"].append(attack_for_report)

    rows["sim_d"].append(sim_d)
    rows["sim_sigma"].append(sim_sigma)
    rows["sim_mean_shift"].append(sim_mean_shift)
    rows["sim_n_train"].append(sim_n_train)
    rows["sim_n_val"].append(sim_n_val)

    rows["metric"].append(metric)
    rows["value"].append(value)
    rows["params_json"].append(params_json)


def run_eta_model_experiments(
    *,
    cfg: EtaExperimentConfig,
    lst_seed: Iterable[int],
    # real-data mode (optional). If provided, simulation generation is skipped.
    X_train: np.ndarray | None = None,
    y_train: np.ndarray | None = None,
    X_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
) -> pd.DataFrame:
    """
    Run eta-model (logreg/svm/rf/mlp) comparison and save results as CSV.

    Output DataFrame is long-format:
        - protocol: ETA_logreg / ETA_svm_rbf / ETA_rf / ETA_mlp
        - metric: val_score, test_eps_emp, test_eps_lower, test_eps_upper, test_tpr, test_fpr, tau, q, ...
        - value: scalar
        - params_json: best params per model (same for all metrics of that model)

    Notes:
        - In simulation mode, it internally generates attackers' auxiliary labeled data.
        - In real-data mode, pass X_train,y_train,X_val,y_val. (labels must be 0/1)
    """
    logging.info("=== run_eta_model_experiments ===")
    attack_for_selection_used = cast(
        Literal["indirect_LRT_hat", "complete_LRT_hat"], cfg.attack_for_selection
    )
    report_attacks: tuple[Literal["indirect_LRT_hat", "complete_LRT_hat"], ...] = (
        ("complete_LRT_hat", "indirect_LRT_hat")
        if cfg.evaluate_both_reports
        else (attack_for_selection_used,)
    )
    eps_list: list[float] = [float(eps) for eps in cfg.epsilon_list]
    if len(eps_list) == 0:
        raise ValueError("epsilon_list must be non-empty.")

    if cfg.hyperparameter == "fixed":
        eta_model_cfgs: list[EtaModelConfig] = get_fixed_logreg_eta_model_configs(c_value=0.1)
    else:
        eta_model_cfgs = (
            get_reduced_eta_model_configs()
            if cfg.use_reduced_grid
            else get_default_eta_model_configs()
        )

    logging.info(
        "analysis=%s selection=%s attack_sel=%s report_attacks=%s eps=%s reduced_grid=%s hyperparameter=%s",
        cfg.analysis,
        cfg.selection,
        attack_for_selection_used,
        report_attacks,
        eps_list,
        cfg.use_reduced_grid,
        cfg.hyperparameter,
    )

    # ---------- results container (long) ----------
    rows: dict[str, list] = {
        "seed": [],
        "protocol": [],
        "k": [],
        "delta": [],
        "epsilon": [],
        "c": [],
        "nb_trials": [],
        "alpha": [],
        "selection": [],
        "attack_for_selection": [],
        "attack_for_report": [],
        "sim_d": [],
        "sim_sigma": [],
        "sim_mean_shift": [],
        "sim_n_train": [],
        "sim_n_val": [],
        "metric": [],
        "value": [],
        "params_json": [],
    }

    # ---------- initialize auditor ----------
    # simulation spec is needed only when simulation mode; but LDPAuditor currently assumes spec exists.
    # For real-data mode, we still give a dummy spec with correct d, and you can later relax this if desired.
    if X_train is None:
        spec = MixtureSpec(
            num_classes=2,
            d=cfg.sim_d,
            sigma=cfg.sim_sigma,
            mean_shift=cfg.sim_mean_shift,
        )
        sim_meta: dict[str, float | None] = dict(
            sim_d=cfg.sim_d,
            sim_sigma=cfg.sim_sigma,
            sim_mean_shift=cfg.sim_mean_shift,
            sim_n_train=cfg.sim_n_train,
            sim_n_val=cfg.sim_n_val,
        )
    else:
        d_real = int(X_train.shape[1])
        # dummy spec (won't be used for train/val generation in real mode)
        spec = MixtureSpec(num_classes=2, d=d_real, sigma=1.0, mean_shift=1.0)
        sim_meta = dict(
            sim_d=None,
            sim_sigma=None,
            sim_mean_shift=None,
            sim_n_train=None,
            sim_n_val=None,
        )

    sim_d_row: int = int(sim_meta["sim_d"] if sim_meta["sim_d"] is not None else -1)
    sim_sigma_row: float | None = sim_meta["sim_sigma"]
    sim_mean_shift_row: float | None = sim_meta["sim_mean_shift"]
    sim_n_train_row: int = int(sim_meta["sim_n_train"] if sim_meta["sim_n_train"] is not None else -1)
    sim_n_val_row: int = int(sim_meta["sim_n_val"] if sim_meta["sim_n_val"] is not None else -1)

    def append_row(
        *,
        seed: int,
        protocol: str,
        epsilon: float,
        metric: str,
        value: float | int | None,
        params_json: str | None,
        attack_for_report: str,
    ) -> None:
        _append_metric(
            rows,
            seed=int(seed),
            protocol=protocol,
            k=cfg.k,
            delta=float(cfg.delta),
            epsilon=float(epsilon),
            c=float(cfg.c),
            nb_trials=int(cfg.nb_trials),
            alpha=float(cfg.alpha),
            selection=str(cfg.selection),
            attack_for_selection=str(attack_for_selection_used),
            attack_for_report=attack_for_report,
            sim_d=sim_d_row,
            sim_sigma=sim_sigma_row,
            sim_mean_shift=sim_mean_shift_row,
            sim_n_train=sim_n_train_row,
            sim_n_val=sim_n_val_row,
            metric=metric,
            value=value,
            params_json=params_json,
        )

    auditor = LDPAuditor(
        nb_trials=cfg.nb_trials,
        alpha=cfg.alpha,
        epsilon=eps_list[0],
        delta=cfg.delta,
        k=cfg.k,
        random_state=0,
        n_jobs=-1,
        rmax_alpha=cfg.sim_rmax_alpha,
        c=cfg.c,
        spec=spec,
        dynamic_nb_trials=False,
    )

    # In real-data mode, B/logRmax from spec is meaningless; but it won't break if you don't call those paths.
    # If you later refactor auditor to allow spec=None for real mode, you can simplify.

    # ---------- run seeds x eps ----------
    for seed in tqdm(list(lst_seed), desc="ETA model comparison (per seed, per epsilon)"):
        for epsilon in eps_list:
            logging.info(
                "Running eta-model comparison for seed=%s, epsilon=%s",
                seed,
                epsilon,
            )
            auditor.set_params(epsilon=float(epsilon), k=cfg.k, random_state=int(seed))

            out: dict = auditor.run_eta_model_comparison_4way(
                seed=int(seed),
                selection=cfg.selection,
                attack_for_selection=attack_for_selection_used,
                report_attacks=report_attacks,
                eta_model_cfgs=eta_model_cfgs,
                sim_n_train=cfg.sim_n_train if X_train is None else None,
                sim_n_val=cfg.sim_n_val if X_train is None else None,
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                y_alt=cfg.y_alt,
                y_null=cfg.y_null,
            )

            for model_name, info in out["results"].items():
                protocol: str = f"ETA_{model_name}"
                params_json: str | None = (
                    json.dumps(info["params"], sort_keys=True)
                    if info.get("params") is not None
                    else None
                )

                append_row(
                    seed=int(seed),
                    protocol=protocol,
                    epsilon=epsilon,
                    metric="val_score",
                    value=float(info["best_val_score"]),
                    params_json=params_json,
                    attack_for_report=str(report_attacks[0]),
                )
                tests_by_attack: dict[str, dict] = info.get("tests_by_attack", {})
                tau_q_by_attack: dict[str, dict[str, float]] = info.get("tau_q_by_attack", {})
                if tau_q_by_attack:
                    for attack_name, tau_q in tau_q_by_attack.items():
                        append_row(
                            seed=int(seed),
                            protocol=protocol,
                            epsilon=epsilon,
                            metric="tau",
                            value=float(tau_q["tau"]),
                            params_json=params_json,
                            attack_for_report=str(attack_name),
                        )
                        append_row(
                            seed=int(seed),
                            protocol=protocol,
                            epsilon=epsilon,
                            metric="q",
                            value=float(tau_q["q"]),
                            params_json=params_json,
                            attack_for_report=str(attack_name),
                        )
                else:
                    append_row(
                        seed=int(seed),
                        protocol=protocol,
                        epsilon=epsilon,
                        metric="tau",
                        value=float(info["tau"]),
                        params_json=params_json,
                        attack_for_report=str(report_attacks[0]),
                    )
                    append_row(
                        seed=int(seed),
                        protocol=protocol,
                        epsilon=epsilon,
                        metric="q",
                        value=float(info["q"]),
                        params_json=params_json,
                        attack_for_report=str(report_attacks[0]),
                    )
                if not tests_by_attack and "test" in info:
                    tests_by_attack = {str(report_attacks[0]): info["test"]}

                for attack_name, test in tests_by_attack.items():
                    append_row(
                        seed=int(seed),
                        protocol=protocol,
                        epsilon=epsilon,
                        metric="test_eps_emp",
                        value=float(test["eps_emp"]),
                        params_json=params_json,
                        attack_for_report=str(attack_name),
                    )
                    eps_lo, eps_hi = test["eps_ci"]
                    append_row(
                        seed=int(seed),
                        protocol=protocol,
                        epsilon=epsilon,
                        metric="test_eps_lower",
                        value=float(eps_lo),
                        params_json=params_json,
                        attack_for_report=str(attack_name),
                    )
                    append_row(
                        seed=int(seed),
                        protocol=protocol,
                        epsilon=epsilon,
                        metric="test_eps_upper",
                        value=float(eps_hi),
                        params_json=params_json,
                        attack_for_report=str(attack_name),
                    )
                    append_row(
                        seed=int(seed),
                        protocol=protocol,
                        epsilon=epsilon,
                        metric="test_tpr_hat",
                        value=float(test["tpr_hat"]),
                        params_json=params_json,
                        attack_for_report=str(attack_name),
                    )
                    append_row(
                        seed=int(seed),
                        protocol=protocol,
                        epsilon=epsilon,
                        metric="test_fpr_hat",
                        value=float(test["fpr_hat"]),
                        params_json=params_json,
                        attack_for_report=str(attack_name),
                    )
                    append_row(
                        seed=int(seed),
                        protocol=protocol,
                        epsilon=epsilon,
                        metric="test_TP",
                        value=int(test["TP"]),
                        params_json=params_json,
                        attack_for_report=str(attack_name),
                    )
                    append_row(
                        seed=int(seed),
                        protocol=protocol,
                        epsilon=epsilon,
                        metric="test_FP",
                        value=int(test["FP"]),
                        params_json=params_json,
                        attack_for_report=str(attack_name),
                    )

    df = pd.DataFrame(rows)

    out_csv: str = (
        f"results/eta_hat_{cfg.analysis}_"
        f"sel={attack_for_selection_used}.csv"
    )
    logging.info("Saving eta-model results to %s", out_csv)
    df.to_csv(out_csv, index=False)

    logging.info("Done. df.shape=%s", df.shape)
    return df


# ============================================================
# Entry point (similar style to experiment_sekiguchi.py)
# ============================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Run eta-model comparison experiment.")
    parser.add_argument(
        "--attack_for_selection",
        choices=["complete_LRT_hat", "indirect_LRT_hat"],
        default="complete_LRT_hat",
        help="Attack used for model/parameter selection.",
    )
    parser.add_argument(
        "--evaluate_both_reports",
        action="store_true",
        help="If set, evaluate both complete/indirect attacks with attack-specific tau,q.",
    )
    parser.add_argument("--nb_trials", type=int, default=int(2e5))
    parser.add_argument("--alpha", type=float, default=1e-2)
    parser.add_argument("--c", type=float, default=1e-2)
    parser.add_argument(
        "--epsilon_list",
        type=float,
        nargs="+",
        default=[0.25, 0.5, 0.75, 1.0, 2.0, 4.0, 6.0, 10.0],
    )
    parser.add_argument("--delta", type=float, default=0.0)
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--y_alt", type=int, default=1)
    parser.add_argument("--y_null", type=int, default=0)
    parser.add_argument(
        "--selection",
        choices=["eps_lower", "eps_emp", "tpr_at_fpr"],
        default="eps_lower",
    )
    parser.add_argument("--use_reduced_grid", action="store_true")
    parser.add_argument(
        "--hyperparameter",
        choices=["grid", "fixed"],
        default="grid",
        help="Hyperparameter strategy: grid search (all models) or fixed logreg C=0.1.",
    )
    parser.add_argument("--sim_d", type=int, default=2)
    parser.add_argument("--sim_sigma", type=float, default=1.0)
    parser.add_argument("--sim_mean_shift", type=float, default=0.1)
    parser.add_argument("--sim_rmax_alpha", type=float, default=1e-6)
    parser.add_argument("--sim_n_train", type=int, default=2000)
    parser.add_argument("--sim_n_val", type=int, default=2000)
    parser.add_argument(
        "--analysis",
        type=str,
        default="shift=0.1_c=1e-2_eps=0.25_10_complete_select_reduced",
    )
    parser.add_argument("--seed_start", type=int, default=0)
    parser.add_argument("--seed_end", type=int, default=5)

    args: argparse.Namespace = parser.parse_args()

    cfg = EtaExperimentConfig(
        nb_trials=args.nb_trials,
        alpha=args.alpha,
        c=args.c,
        epsilon_list=tuple(float(x) for x in args.epsilon_list),
        delta=args.delta,
        k=args.k,
        y_alt=args.y_alt,
        y_null=args.y_null,
        selection=cast(Literal["eps_lower", "eps_emp", "tpr_at_fpr"], args.selection),
        attack_for_selection=cast(
            Literal["indirect_LRT_hat", "complete_LRT_hat"],
            args.attack_for_selection,
        ),
        evaluate_both_reports=bool(args.evaluate_both_reports),
        use_reduced_grid=bool(args.use_reduced_grid),
        hyperparameter=cast(Literal["grid", "fixed"], args.hyperparameter),
        sim_d=args.sim_d,
        sim_sigma=args.sim_sigma,
        sim_mean_shift=args.sim_mean_shift,
        sim_rmax_alpha=args.sim_rmax_alpha,
        sim_n_train=args.sim_n_train,
        sim_n_val=args.sim_n_val,
        analysis=args.analysis,
    )

    lst_seed = range(args.seed_start, args.seed_end)

    df: pd.DataFrame = run_eta_model_experiments(cfg=cfg, lst_seed=lst_seed)
    print("==== Results DataFrame (head) ====")
    print(df.head(20))

# python experiment_eta_model.py \
#   --attack_for_selection complete_LRT_hat \
#   --evaluate_both_reports \
#   --use_reduced_grid

# python experiment_eta_model.py \
#   --attack_for_selection indirect_LRT_hat \
#   --use_reduced_grid

# python experiment_eta_model.py \
#   --attack_for_selection complete_LRT_hat \
#   --use_reduced_grid

# time python experiment_eta_model.py \
#   --attack_for_selection complete_LRT_hat \
#   --evaluate_both_reports \
#   --nb_trials  10000 \
#   --sim_n_train 1000 \
#   --sim_n_val 1000 \
#   --sim_d 20 \
#   --sim_mean_shift 0.1 \
#   --alpha 0.01 \
#   --c 0.01 \
#   --hyperparameter fixed

