# experiment_eta_model.py

from __future__ import annotations

import argparse
import ast
import json
import logging
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable, Literal, Sequence, Callable, cast

import numpy as np
from numpy.random import Generator
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from ldp_audit.base_auditor import LDPAuditor
from ldp_audit.eta_models import (
    EtaModelConfig,
    fit_and_select_eta_model,
    get_default_eta_model_configs,
    get_fixed_logreg_eta_model_configs,
    get_reduced_eta_model_configs,
    predict_eta,
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
    n_total: int | None = None
    ratio_train: float | None = None
    ratio_val: float | None = None
    ratio_final: float | None = None
    real_data_name: str | None = None
    collect_score_distribution: bool = False

    # output
    analysis: str = "eta_model_comparison"
    output_root: str = "./results"


def _count_csv_rows(csv_path: str, chunksize: int = 1 << 20) -> int:
    total: int = 0
    with open(csv_path, "rb") as f:
        while True:
            chunk: bytes = f.read(chunksize)
            if not chunk:
                break
            total += chunk.count(b"\n")
    return total


def _parse_n_total_arg(
    raw_value: str | None,
    *,
    real_data: bool,
    real_data_path: str,
) -> int | None:
    if raw_value is None:
        return None
    text: str = str(raw_value).strip().lower()
    if text == "all":
        if not real_data:
            raise ValueError("--N_total all is supported only with --real_data.")
        return _count_csv_rows(real_data_path)
    return int(raw_value)


def _validate_ratio_triplet(ratio: Sequence[float]) -> tuple[float, float, float]:
    if len(ratio) != 3:
        raise ValueError(f"N_ratio must contain exactly 3 values, got {ratio}.")
    r_train, r_val, r_final = (float(r) for r in ratio)
    if min(r_train, r_val, r_final) <= 0.0:
        raise ValueError(f"N_ratio values must be positive, got {ratio}.")
    if not np.isclose(r_train + r_val + r_final, 1.0, atol=1e-8):
        raise ValueError(f"N_ratio must sum to 1.0, got {ratio}.")
    return r_train, r_val, r_final


def _parse_n_ratio_args(
    raw_values: Sequence[str] | None,
) -> list[tuple[float, float, float]] | None:
    if raw_values is None or len(raw_values) == 0:
        return None

    text: str = " ".join(raw_values).strip()
    if text.startswith("[") or text.startswith("("):
        parsed = ast.literal_eval(text)
        if (
            isinstance(parsed, tuple)
            and len(parsed) == 3
            and all(isinstance(v, (int, float)) for v in parsed)
        ):
            return [_validate_ratio_triplet(tuple(float(v) for v in parsed))]
        if not isinstance(parsed, (list, tuple)):
            raise ValueError(f"Could not parse N_ratio from: {text}")
        return [
            _validate_ratio_triplet(tuple(float(v) for v in item)) for item in parsed
        ]

    ratios: list[tuple[float, float, float]] = []
    for token in raw_values:
        cleaned: str = token.strip().strip("[]()")
        parts: list[str] = [p.strip() for p in cleaned.split(",") if p.strip()]
        if len(parts) != 3:
            raise ValueError(
                f"Each N_ratio token must look like '0.1,0.1,0.8'. Got: {token}"
            )
        ratios.append(_validate_ratio_triplet(tuple(float(p) for p in parts)))
    return ratios


def _format_ratio_for_path(x: float) -> str:
    return f"{float(x):.3f}".rstrip("0").rstrip(".")


def _sanitize_for_path(text: str) -> str:
    safe_chars: set[str] = {"-", "_", ".", "="}
    return "".join(ch if ch.isalnum() or ch in safe_chars else "_" for ch in text)


def _save_score_distribution_csv(
    *,
    score_values: np.ndarray,
    class_values: np.ndarray,
    output_dir: Path,
    data_name: str,
    protocol: str,
) -> Path:
    score_values = np.asarray(score_values, dtype=np.float64)
    class_values = np.asarray(class_values, dtype=np.int64)
    if score_values.shape[0] != class_values.shape[0]:
        raise ValueError(
            "score_values and class_values must contain the same number of records. "
            f"Got {score_values.shape[0]} and {class_values.shape[0]}."
        )

    out_path: Path = output_dir / (
        f"{_sanitize_for_path(protocol)}_{_sanitize_for_path(data_name)}_distribution.csv"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "score": score_values,
            "class": class_values,
        }
    ).to_csv(out_path, index=False)
    return out_path


def _estimate_binary_class_prior(y: np.ndarray) -> tuple[float, float]:
    """Estimate (P[Y=0], P[Y=1]) from labels used to fit eta_hat."""
    y_arr: np.ndarray = np.asarray(y, dtype=np.int64)
    n0 = int(np.sum(y_arr == 0))
    n1 = int(np.sum(y_arr == 1))
    total: int = n0 + n1
    if n0 <= 0 or n1 <= 0 or total <= 0:
        raise ValueError(
            "eta prior correction requires both labels 0 and 1 in y_train. "
            f"Got n0={n0}, n1={n1}."
        )
    return (float(n0 / total), float(n1 / total))


def _save_indirect_score_distributions(
    *,
    eta_model_cfgs: Sequence[EtaModelConfig],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_final: np.ndarray,
    y_final: np.ndarray,
    output_dir: Path,
    data_name: str,
    selection: Literal["eps_lower", "eps_emp", "tpr_at_fpr"],
    auditor: LDPAuditor,
) -> None:
    score_fn: Callable[..., float] = auditor._select_score_fn_for_eta_model(
        X_val=X_val,
        y_val=y_val,
        selection=selection,
        rng_seed_for_val=0,
        attack_for_selection="indirect_LRT_hat",
    )

    # For each eta model family, fit+select best hyperparameters using the validation set,
    # then save the prior-corrected indirect score distribution on the final test set.
    for model_cfg in eta_model_cfgs:
        best: dict[str, object] = fit_and_select_eta_model(
            cfg=model_cfg,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            seed=0,
            score_fn=score_fn,
        )
        model: object = best["model"]
        if model is None:
            raise RuntimeError(f"Failed to fit model for protocol={model_cfg.name}.")
        score_csv_path: Path = _save_score_distribution_csv(
            score_values=auditor._eta_to_x_lr_prob(predict_eta(model, X_final)),  
            class_values=y_final,
            output_dir=output_dir,
            data_name=data_name,
            protocol=f"ETA_{model_cfg.name}",
        )
        logging.info("Saved score distribution to %s", score_csv_path)


def _resolve_count_plan(
    *,
    n_total: int,
    ratio: tuple[float, float, float],
    nb_trials_override: int | None,
    sim_n_train_override: int | None,
    sim_n_val_override: int | None,
) -> tuple[int, int, int]:
    ratio_train, ratio_val, ratio_final = ratio

    sim_n_train: int = (
        int(sim_n_train_override)
        if sim_n_train_override is not None
        else int(round(n_total * ratio_train))
    )
    sim_n_val: int = (
        int(sim_n_val_override)
        if sim_n_val_override is not None
        else int(round(n_total * ratio_val))
    )
    if nb_trials_override is not None:
        nb_trials = int(nb_trials_override)
    elif sim_n_train_override is None and sim_n_val_override is None:
        nb_trials = int(n_total - sim_n_train - sim_n_val)
    else:
        nb_trials = int(round(n_total * ratio_final))

    if min(sim_n_train, sim_n_val, nb_trials) <= 0:
        raise ValueError(
            "Resolved counts must be positive. "
            f"Got sim_n_train={sim_n_train}, sim_n_val={sim_n_val}, nb_trials={nb_trials}."
        )
    return sim_n_train, sim_n_val, nb_trials


def _build_ratio_cfgs(
    *,
    base_cfg: EtaExperimentConfig,
    n_total: int | None,
    ratios: list[tuple[float, float, float]] | None,
    nb_trials_override: int | None,
    sim_n_train_override: int | None,
    sim_n_val_override: int | None,
) -> list[EtaExperimentConfig]:
    if n_total is None:
        return [base_cfg]

    if ratios is None or len(ratios) == 0:
        if (
            nb_trials_override is not None
            and sim_n_train_override is not None
            and sim_n_val_override is not None
        ):
            return [replace(base_cfg, n_total=int(n_total))]
        raise ValueError(
            "When N_total is set, N_ratio must also be set unless all three counts are explicitly overridden."
        )

    cfgs: list[EtaExperimentConfig] = []
    for ratio in ratios:
        ratio_train, ratio_val, ratio_final = ratio
        sim_n_train, sim_n_val, nb_trials = _resolve_count_plan(
            n_total=int(n_total),
            ratio=ratio,
            nb_trials_override=nb_trials_override,
            sim_n_train_override=sim_n_train_override,
            sim_n_val_override=sim_n_val_override,
        )
        analysis_suffix: str = (
            f"_Ntotal={int(n_total)}"
            f"_r={_format_ratio_for_path(ratio_train)}-{_format_ratio_for_path(ratio_val)}-{_format_ratio_for_path(ratio_final)}"
        )
        cfgs.append(
            replace(
                base_cfg,
                nb_trials=nb_trials,
                sim_n_train=sim_n_train,
                sim_n_val=sim_n_val,
                n_total=int(n_total),
                ratio_train=ratio_train,
                ratio_val=ratio_val,
                ratio_final=ratio_final,
                analysis=f"{base_cfg.analysis}{analysis_suffix}",
            )
        )
    return cfgs


def _reservoir_sample_csv_rows(
    *,
    csv_path: str,
    n_rows: int,
    seed: int,
    chunksize: int = 100_000,
) -> np.ndarray:
    if n_rows <= 0:
        raise ValueError(f"n_rows must be positive, got {n_rows}.")

    rng: Generator = np.random.default_rng(seed)
    reservoir: np.ndarray | None = None
    seen: int = 0

    for chunk in pd.read_csv(csv_path, header=None, chunksize=chunksize, dtype=np.float32):
        arr = chunk.to_numpy(dtype=np.float32, copy=False)
        if reservoir is None:
            reservoir = np.empty((n_rows, arr.shape[1]), dtype=np.float32)
        for row in arr:
            seen += 1
            if seen <= n_rows:
                reservoir[seen - 1] = row
            else:
                j = int(rng.integers(seen))
                if j < n_rows:
                    reservoir[j] = row

    if reservoir is None or seen < n_rows:
        raise ValueError(f"Requested n_rows={n_rows}, but csv only had {seen} rows.")
    return reservoir


def _reservoir_sample_csv_rows_raw(
    *,
    csv_path: str,
    n_rows: int,
    seed: int,
    chunksize: int = 100_000,
) -> np.ndarray:
    if n_rows <= 0:
        raise ValueError(f"n_rows must be positive, got {n_rows}.")

    rng: Generator = np.random.default_rng(seed)
    reservoir: np.ndarray | None = None
    seen: int = 0

    for chunk in pd.read_csv(
        csv_path,
        header=None,
        chunksize=chunksize,
        dtype=str,
        keep_default_na=False,
        na_filter=False,
    ):
        arr: np.ndarray = chunk.to_numpy(dtype=object, copy=False)
        if reservoir is None:
            reservoir = np.empty((n_rows, arr.shape[1]), dtype=object)
        for row in arr:
            seen += 1
            if seen <= n_rows:
                reservoir[seen - 1] = row
            else:
                j: int = int(rng.integers(seen))
                if j < n_rows:
                    reservoir[j] = row

    if reservoir is None or seen < n_rows:
        raise ValueError(f"Requested n_rows={n_rows}, but csv only had {seen} rows.")
    return reservoir


def _infer_real_csv_feature_kinds(
    *,
    csv_path: str,
    sample_rows: int = 50_000,
) -> list[str]:
    sample_df: pd.DataFrame = pd.read_csv(
        csv_path,
        header=None,
        nrows=sample_rows,
        dtype=str,
        keep_default_na=False,
        na_filter=False,
    )
    if sample_df.shape[1] < 2:
        raise ValueError(
            "real_data csv must contain at least 2 columns (label + features). "
            f"Got shape={sample_df.shape}."
        )

    kinds: list[str] = []
    for col_idx in range(1, sample_df.shape[1]):
        col: pd.Series[str] = sample_df.iloc[:, col_idx].astype(str).str.strip()
        numeric: pd.Series   = pd.to_numeric(col, errors="coerce")
        kinds.append("numeric" if bool(numeric.notna().all()) else "categorical")
    return kinds


def _hash_categorical_series(series: pd.Series) -> np.ndarray:
    tokens: pd.Series[str] = series.astype(str)
    hashed: np.ndarray = pd.util.hash_pandas_object(tokens, index=False).to_numpy(dtype=np.uint64)
    scaled = (hashed.astype(np.float64) / np.float64(np.iinfo(np.uint64).max)) * 2.0 - 1.0
    return scaled.astype(np.float32, copy=False)


def _encode_real_csv_frame(
    *,
    frame: pd.DataFrame,
    feature_kinds: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    if frame.shape[1] != len(feature_kinds) + 1:
        raise ValueError(
            "Feature kind schema does not match csv width. "
            f"frame.shape={frame.shape}, len(feature_kinds)={len(feature_kinds)}."
        )

    y: np.ndarray = pd.to_numeric(frame.iloc[:, 0], errors="raise").to_numpy(dtype=np.int64, copy=False)
    X: np.ndarray = np.empty((len(frame), len(feature_kinds)), dtype=np.float32)

    for feature_idx, kind in enumerate(feature_kinds, start=1):
        col: pd.Series = frame.iloc[:, feature_idx]
        if kind == "numeric":
            X[:, feature_idx - 1] = pd.to_numeric(col, errors="raise").to_numpy(
                dtype=np.float32,
                copy=False,
            )
        elif kind == "categorical":
            X[:, feature_idx - 1] = _hash_categorical_series(col)
        else:
            raise ValueError(f"Unknown feature kind: {kind}")
    return X, y


def _load_mixed_real_csv_all_rows(
    *,
    csv_path: str,
    total_rows: int,
    feature_kinds: Sequence[str],
    chunksize: int = 100_000,
) -> tuple[np.ndarray, np.ndarray]:
    X: np.ndarray = np.empty((total_rows, len(feature_kinds)), dtype=np.float32)
    y: np.ndarray = np.empty(total_rows, dtype=np.int64)

    start = 0
    for chunk in pd.read_csv(
        csv_path,
        header=None,
        chunksize=chunksize,
        dtype=str,
        keep_default_na=False,
        na_filter=False,
    ):
        X_chunk, y_chunk = _encode_real_csv_frame(frame=chunk, feature_kinds=feature_kinds)
        end: int = start + len(chunk)
        X[start:end] = X_chunk
        y[start:end] = y_chunk
        start: int = end

    if start != total_rows:
        raise ValueError(f"Expected {total_rows} rows, but loaded {start}.")
    return X, y


def load_susy_real_data_split(
    *,
    csv_path: str,
    n_train: int,
    n_val: int,
    n_final: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_total = int(n_train + n_val + n_final)
    total_rows: int = _count_csv_rows(csv_path)
    logging.info(f"[load_susy] total_rows={total_rows}, requested={n_total}")
    feature_kinds: list[str] = _infer_real_csv_feature_kinds(csv_path=csv_path)
    numeric_only: bool = all(kind == "numeric" for kind in feature_kinds)
    logging.info(
        "[load_susy] detected %s feature columns (%s numeric, %s categorical)",
        len(feature_kinds),
        sum(kind == "numeric" for kind in feature_kinds),
        sum(kind == "categorical" for kind in feature_kinds),
    )

    if numeric_only:
        if n_total == total_rows:
            sampled: np.ndarray = pd.read_csv(csv_path, header=None, dtype=np.float32).to_numpy(
                dtype=np.float32,
                copy=False,
            )
        else:
            sampled = _reservoir_sample_csv_rows(csv_path=csv_path, n_rows=n_total, seed=seed)
        y: np.ndarray = sampled[:, 0].astype(np.int64)
        X: np.ndarray = sampled[:, 1:].astype(np.float32)
    else:
        if n_total == total_rows:
            X, y = _load_mixed_real_csv_all_rows(
                csv_path=csv_path,
                total_rows=total_rows,
                feature_kinds=feature_kinds,
            )
        else:
            sampled_raw: np.ndarray = _reservoir_sample_csv_rows_raw(
                csv_path=csv_path,
                n_rows=n_total,
                seed=seed,
            )
            sampled_frame: pd.DataFrame = pd.DataFrame(sampled_raw)
            X, y = _encode_real_csv_frame(frame=sampled_frame, feature_kinds=feature_kinds)

    X_train, X_rest, y_train, y_rest = train_test_split(
        X,
        y,
        train_size=int(n_train),
        stratify=y,
        random_state=int(seed),
    )
    X_val, X_final, y_val, y_final = train_test_split(
        X_rest,
        y_rest,
        train_size=int(n_val),
        stratify=y_rest,
        random_state=int(seed) + 1,
    )
    return X_train, y_train, X_val, y_val, X_final, y_final


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
    n_total: int | None,
    ratio_train: float | None,
    ratio_val: float | None,
    ratio_final: float | None,
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
    rows["n_total"].append(n_total)
    rows["ratio_train"].append(ratio_train)
    rows["ratio_val"].append(ratio_val)
    rows["ratio_final"].append(ratio_final)

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
    X_final: np.ndarray | None = None,
    y_final: np.ndarray | None = None,
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
        - In real-data mode, pass X_train,y_train,X_val,y_val,X_final,y_final. (labels must be 0/1)
    """
    logging.info("=== run_eta_model_experiments ===")
    report_attacks: tuple[Literal["indirect_LRT_hat", "complete_LRT_hat"], ...] = (
        ("complete_LRT_hat", "indirect_LRT_hat")
        if cfg.evaluate_both_reports
        else ("complete_LRT_hat",)
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
        "analysis=%s selection=%s report_attacks=%s eps=%s reduced_grid=%s hyperparameter=%s",
        cfg.analysis,
        cfg.selection,
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
        "n_total": [],
        "ratio_train": [],
        "ratio_val": [],
        "ratio_final": [],
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
        eta_class_prior: tuple[float, float] = (0.5, 0.5)
        sim_meta: dict[str, float | None] = dict(
            sim_d=cfg.sim_d,
            sim_sigma=cfg.sim_sigma,
            sim_mean_shift=cfg.sim_mean_shift,
            sim_n_train=cfg.sim_n_train,
            sim_n_val=cfg.sim_n_val,
        )
    else:
        if X_val is None or y_val is None or X_final is None or y_final is None:
            raise ValueError("Real-data mode requires X_val/y_val/X_final/y_final.")
        d_real = int(X_train.shape[1])
        # dummy spec (won't be used for train/val generation in real mode)
        spec = MixtureSpec(num_classes=2, d=d_real, sigma=1.0, mean_shift=1.0)
        eta_class_prior = _estimate_binary_class_prior(y_train) if y_train is not None else (0.5, 0.5)
        logging.info(
            "Using eta prior correction from y_train: P[Y=0]=%.6g, P[Y=1]=%.6g, log(pi0/pi1)=%.6g",
            eta_class_prior[0],
            eta_class_prior[1],
            float(np.log(eta_class_prior[0]) - np.log(eta_class_prior[1])),
        )
        sim_meta = dict(
            sim_d=d_real,
            sim_sigma=None,
            sim_mean_shift=None,
            sim_n_train=X_train.shape[0],
            sim_n_val=X_val.shape[0],
        )

    sim_d_row: int = int(sim_meta["sim_d"] if sim_meta["sim_d"] is not None else -1)
    sim_sigma_row: float | None = sim_meta["sim_sigma"]
    sim_mean_shift_row: float | None = sim_meta["sim_mean_shift"]
    sim_n_train_row: int = int(sim_meta["sim_n_train"] if sim_meta["sim_n_train"] is not None else -1)
    sim_n_val_row: int = int(sim_meta["sim_n_val"] if sim_meta["sim_n_val"] is not None else -1)
    output_root: Path = Path(cfg.output_root)
    dist_output_dir: Path = output_root / "score_distributions"
    score_dist_data_name: str = cfg.real_data_name or "real"

    def append_row(
        *,
        seed: int,
        protocol: str,
        epsilon: float,
        attack_for_selection: str,
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
            attack_for_selection=attack_for_selection,
            attack_for_report=attack_for_report,
            sim_d=sim_d_row,
            sim_sigma=sim_sigma_row,
            sim_mean_shift=sim_mean_shift_row,
            sim_n_train=sim_n_train_row,
            sim_n_val=sim_n_val_row,
            n_total=cfg.n_total,
            ratio_train=cfg.ratio_train,
            ratio_val=cfg.ratio_val,
            ratio_final=cfg.ratio_final,
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
        sim_hat=(X_train is None),
        real_val_X=X_val,
        real_val_y=y_val,
        real_final_X=X_final,
        real_final_y=y_final,
        eta_class_prior=eta_class_prior,
    )

    # In real-data mode, B/logRmax from spec is meaningless; but it won't break if you don't call those paths.
    # If you later refactor auditor to allow spec=None for real mode, you can simplify.

    if (
        cfg.collect_score_distribution
        and X_train is not None
        and y_train is not None
        and X_val is not None
        and y_val is not None
        and X_final is not None
        and y_final is not None
    ):
        _save_indirect_score_distributions(
            eta_model_cfgs=eta_model_cfgs,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            X_final=X_final,
            y_final=y_final,
            output_dir=dist_output_dir,
            data_name=score_dist_data_name,
            selection=cfg.selection,
            auditor=auditor,
        )

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
            prior_info: dict[str, float] = cast(dict[str, float], out.get("eta_class_prior", {}))

            for model_name, info in out["results"].items():
                protocol: str = f"ETA_{model_name}"
                best_by_attack: dict[str, dict] = info.get("best_by_attack", {})
                params_json_by_attack: dict[str, str | None] = {}
                if best_by_attack:
                    for attack_name, best in best_by_attack.items():
                        params_json: str | None = (
                            json.dumps(best["params"], sort_keys=True)
                            if best.get("params") is not None
                            else None
                        )
                        params_json_by_attack[str(attack_name)] = params_json
                        append_row(
                            seed=int(seed),
                            protocol=protocol,
                            epsilon=epsilon,
                            attack_for_selection=str(attack_name),
                            metric="val_score",
                            value=float(best["best_val_score"]),
                            params_json=params_json,
                            attack_for_report=str(attack_name),
                        )
                else:
                    params_json = (
                        json.dumps(info["params"], sort_keys=True)
                        if info.get("params") is not None
                        else None
                    )
                    params_json_by_attack[str(report_attacks[0])] = params_json
                    append_row(
                        seed=int(seed),
                        protocol=protocol,
                        epsilon=epsilon,
                        attack_for_selection=str(report_attacks[0]),
                        metric="val_score",
                        value=float(info["best_val_score"]),
                        params_json=params_json,
                        attack_for_report=str(report_attacks[0]),
                    )
                tests_by_attack: dict[str, dict] = info.get("tests_by_attack", {})
                tau_q_by_attack: dict[str, dict[str, float]] = info.get("tau_q_by_attack", {})
                if tau_q_by_attack:
                    for attack_name, tau_q in tau_q_by_attack.items():
                        for metric_name, prior_key in (
                            ("eta_prior_y0", "pi0"),
                            ("eta_prior_y1", "pi1"),
                            ("eta_log_prior_correction", "log_pi0_over_pi1"),
                        ):
                            append_row(
                                seed=int(seed),
                                protocol=protocol,
                                epsilon=epsilon,
                                attack_for_selection=str(attack_name),
                                metric=metric_name,
                                value=float(prior_info[prior_key]) if prior_key in prior_info else None,
                                params_json=params_json_by_attack.get(str(attack_name)),
                                attack_for_report=str(attack_name),
                            )
                        append_row(
                            seed=int(seed),
                            protocol=protocol,
                            epsilon=epsilon,
                            attack_for_selection=str(attack_name),
                            metric="tau",
                            value=float(tau_q["tau"]),
                            params_json=params_json_by_attack.get(str(attack_name)),
                            attack_for_report=str(attack_name),
                        )
                        append_row(
                            seed=int(seed),
                            protocol=protocol,
                            epsilon=epsilon,
                            attack_for_selection=str(attack_name),
                            metric="q",
                            value=float(tau_q["q"]),
                            params_json=params_json_by_attack.get(str(attack_name)),
                            attack_for_report=str(attack_name),
                        )
                else:
                    append_row(
                        seed=int(seed),
                        protocol=protocol,
                        epsilon=epsilon,
                        attack_for_selection=str(report_attacks[0]),
                        metric="tau",
                        value=float(info["tau"]),
                        params_json=params_json_by_attack.get(str(report_attacks[0])),
                        attack_for_report=str(report_attacks[0]),
                    )
                    append_row(
                        seed=int(seed),
                        protocol=protocol,
                        epsilon=epsilon,
                        attack_for_selection=str(report_attacks[0]),
                        metric="q",
                        value=float(info["q"]),
                        params_json=params_json_by_attack.get(str(report_attacks[0])),
                        attack_for_report=str(report_attacks[0]),
                    )
                if not tests_by_attack and "test" in info:
                    tests_by_attack = {str(report_attacks[0]): info["test"]}

                for attack_name, test in tests_by_attack.items():
                    append_row(
                        seed=int(seed),
                        protocol=protocol,
                        epsilon=epsilon,
                        attack_for_selection=str(attack_name),
                        metric="test_eps_emp",
                        value=float(test["eps_emp"]),
                        params_json=params_json_by_attack.get(str(attack_name)),
                        attack_for_report=str(attack_name),
                    )
                    eps_lo, eps_hi = test["eps_ci"]
                    append_row(
                        seed=int(seed),
                        protocol=protocol,
                        epsilon=epsilon,
                        attack_for_selection=str(attack_name),
                        metric="test_eps_lower",
                        value=float(eps_lo),
                        params_json=params_json_by_attack.get(str(attack_name)),
                        attack_for_report=str(attack_name),
                    )
                    append_row(
                        seed=int(seed),
                        protocol=protocol,
                        epsilon=epsilon,
                        attack_for_selection=str(attack_name),
                        metric="test_eps_upper",
                        value=float(eps_hi),
                        params_json=params_json_by_attack.get(str(attack_name)),
                        attack_for_report=str(attack_name),
                    )
                    append_row(
                        seed=int(seed),
                        protocol=protocol,
                        epsilon=epsilon,
                        attack_for_selection=str(attack_name),
                        metric="test_tpr_hat",
                        value=float(test["tpr_hat"]),
                        params_json=params_json_by_attack.get(str(attack_name)),
                        attack_for_report=str(attack_name),
                    )
                    append_row(
                        seed=int(seed),
                        protocol=protocol,
                        epsilon=epsilon,
                        attack_for_selection=str(attack_name),
                        metric="test_fpr_hat",
                        value=float(test["fpr_hat"]),
                        params_json=params_json_by_attack.get(str(attack_name)),
                        attack_for_report=str(attack_name),
                    )
                    append_row(
                        seed=int(seed),
                        protocol=protocol,
                        epsilon=epsilon,
                        attack_for_selection=str(attack_name),
                        metric="test_TP",
                        value=int(test["TP"]),
                        params_json=params_json_by_attack.get(str(attack_name)),
                        attack_for_report=str(attack_name),
                    )
                    append_row(
                        seed=int(seed),
                        protocol=protocol,
                        epsilon=epsilon,
                        attack_for_selection=str(attack_name),
                        metric="test_FP",
                        value=int(test["FP"]),
                        params_json=params_json_by_attack.get(str(attack_name)),
                        attack_for_report=str(attack_name),
                    )

    df = pd.DataFrame(rows)

    if X_train is None:
        out_csv: str = (
            f"{output_root}/eta_hat_shift={cfg.sim_mean_shift}_c={cfg.c}"
            f"_f={cfg.nb_trials}_t={cfg.sim_n_train}_v={cfg.sim_n_val}_d={cfg.sim_d}.csv"
        )
    else:
        data_name: str = cfg.real_data_name or "real"
        out_csv = (
            f"{output_root}/eta_hat_{data_name}_c={cfg.c}"
            f"_f={cfg.nb_trials}_t={X_train.shape[0]}_v={X_val.shape[0]}_d={X_train.shape[1]}.csv" # type: ignore
        )
    logging.info("Saving eta-model results to %s", out_csv)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
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
        "--evaluate_both_reports",
        action="store_true",
        help="If set, evaluate both complete/indirect attacks with attack-specific tau,q.",
    )
    parser.add_argument("--nb_trials", type=int, default=None)
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
    parser.add_argument("--sim_n_train", type=int, default=None)
    parser.add_argument("--sim_n_val", type=int, default=None)
    parser.add_argument("--real_data", action="store_true")
    parser.add_argument(
        "--real_data_path",
        type=str,
        default="./data/SUSY/SUSY.csv",
    )
    parser.add_argument("--real_data_name", type=str, default="SUSY")
    parser.add_argument("--real_data_seed", type=int, default=0)
    parser.add_argument(
        "--score_dist",
        action="store_true",
        help=(
            "If set, save indirect eta_hat(x) distributions as score/class CSVs. "
            "This is only enabled when --real_data is also set."
        ),
    )
    parser.add_argument("--N_total", type=str, default=None)
    parser.add_argument(
        "--N_ratio",
        nargs="*",
        default=None,
        help=(
            "Ratio settings for (train, val, final). "
            "Examples: --N_ratio 0.1,0.1,0.8 0.15,0.15,0.7 "
            "or --N_ratio '[(0.1,0.1,0.8),(0.15,0.15,0.7)]'"
        ),
    )
    parser.add_argument(
        "--analysis",
        type=str,
        default="shift=0.1_c=1e-2_eps=0.25_10_f=10000_t=1000_v=1000",
    )
    parser.add_argument("--seed_start", type=int, default=0)
    parser.add_argument("--seed_end", type=int, default=5)
    parser.add_argument(
        "--output_root",
        type=str,
        default="./results",
        help="Root directory to save outputs.",
    )

    args: argparse.Namespace = parser.parse_args()

    if (
        args.N_total is not None
        and str(args.N_total).strip().lower() == "all"
        and any(v is not None for v in (args.nb_trials, args.sim_n_train, args.sim_n_val))
    ):
        raise ValueError(
            "When --N_total all is used, do not also specify --nb_trials/--sim_n_train/--sim_n_val."
        )

    n_total_value: int | None = _parse_n_total_arg(
        args.N_total,
        real_data=bool(args.real_data),
        real_data_path=str(args.real_data_path),
    )
    n_ratio_list: list[tuple[float, float, float]] | None = _parse_n_ratio_args(args.N_ratio)
    collect_score_distribution: bool = bool(args.score_dist and args.real_data)
    if args.score_dist and not args.real_data:
        logging.info("Ignoring --score_dist because score distributions are saved only with --real_data.")
    base_cfg = EtaExperimentConfig(
        nb_trials=int(args.nb_trials) if args.nb_trials is not None else int(2e5),
        alpha=args.alpha,
        c=args.c,
        epsilon_list=tuple(float(x) for x in args.epsilon_list),
        delta=args.delta,
        k=args.k,
        y_alt=args.y_alt,
        y_null=args.y_null,
        selection=cast(Literal["eps_lower", "eps_emp", "tpr_at_fpr"], args.selection),
        evaluate_both_reports=bool(args.evaluate_both_reports),
        use_reduced_grid=bool(args.use_reduced_grid),
        hyperparameter=cast(Literal["grid", "fixed"], args.hyperparameter),
        sim_d=args.sim_d,
        sim_sigma=args.sim_sigma,
        sim_mean_shift=args.sim_mean_shift,
        sim_rmax_alpha=args.sim_rmax_alpha,
        sim_n_train=int(args.sim_n_train) if args.sim_n_train is not None else 2000,
        sim_n_val=int(args.sim_n_val) if args.sim_n_val is not None else 2000,
        real_data_name=args.real_data_name if args.real_data else None,
        collect_score_distribution=collect_score_distribution,
        analysis=args.analysis,
        output_root=args.output_root,
    )

    lst_seed = range(args.seed_start, args.seed_end)
    cfgs: list[EtaExperimentConfig] = _build_ratio_cfgs(
        base_cfg=base_cfg,
        n_total=n_total_value,
        ratios=n_ratio_list,
        nb_trials_override=args.nb_trials,
        sim_n_train_override=args.sim_n_train,
        sim_n_val_override=args.sim_n_val,
    )

    for cfg in cfgs:
        if cfg.n_total is not None:
            effective_total: int = cfg.nb_trials + cfg.sim_n_train + cfg.sim_n_val
            if effective_total != int(cfg.n_total):
                logging.warning(
                    "Effective total (%s) does not match N_total (%s) because explicit overrides were applied.",
                    effective_total,
                    cfg.n_total,
                )
        if args.real_data:
            X_train, y_train, X_val, y_val, X_final, y_final = load_susy_real_data_split(
                csv_path=args.real_data_path,
                n_train=int(cfg.sim_n_train),
                n_val=int(cfg.sim_n_val),
                n_final=int(cfg.nb_trials),
                seed=int(args.real_data_seed),
            )
            df: pd.DataFrame = run_eta_model_experiments(
                cfg=cfg,
                lst_seed=lst_seed,
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                X_final=X_final,
                y_final=y_final,
            )
        else:
            df = run_eta_model_experiments(cfg=cfg, lst_seed=lst_seed)
        print("==== Results DataFrame (head) ====")
        print(df.head(20))

# python experiment_eta_model.py \
#   --evaluate_both_reports \
#   --use_reduced_grid

# time python experiment_eta_model.py \
#   --evaluate_both_reports \
#   --nb_trials  10000 \
#   --sim_n_train 1000 \
#   --sim_n_val 1000 \
#   --sim_d 20 \
#   --sim_mean_shift 0.1 \
#   --alpha 0.01 \
#   --c 0.01 \
#   --hyperparameter fixed

# time python experiment_eta_model.py \
#   --evaluate_both_reports \
#   --N_total 30000 \
#   --N_ratio 0.1,0.1,0.8 0.15,0.15,0.7 0.2,0.2,0.6 0.25,0.15,0.6 0.15,0.25,0.6 0.3,0.1,0.6 0.1,0.3,0.6 \
#   --hyperparameter fixed \
#   --sim_d 20 \
#   --sim_mean_shift 0.5
#   --alpha 0.01 \
#   --c 0.01

# time python experiment_eta_model.py \
#   --real_data \
#   --real_data_path ./data/SUSY/SUSY.csv \
#   --real_data_name SUSY \
#   --evaluate_both_reports \
#   --hyperparameter fixed \
#   --selection eps_lower \
#   --N_total 3000 \
#   --N_ratio 0.2,0.2,0.6 \
#   --seed_start 0 \
#   --seed_end 1

# time python experiment_eta_model.py \
#   --real_data \
#   --real_data_path ./data/SUSY/SUSY.csv \
#   --real_data_name SUSY \
#   --output_root ./results/20260401 \
#   --evaluate_both_reports \
#   --hyperparameter fixed \
#   --N_total all \
#   --N_ratio 0.2,0.2,0.6 \
#   --seed_start 0 \
#   --seed_end 9 \
#   --score_dist

# python experiment_eta_model.py \
#   --real_data \
#   --real_data_path ./data/criteo_search/criteo_search_numeric.csv \
#   --real_data_name CriteoSearch \
#   --output_root ./results/20260401 \
#   --evaluate_both_reports \
#   --hyperparameter fixed \
#   --N_total 3597294 \
#   --N_ratio 0.2,0.2,0.6 \
#   --seed_start 0 \
#   --seed_end 3
