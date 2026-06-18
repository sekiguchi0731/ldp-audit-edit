# experiment_eta_model.py

from __future__ import annotations

import argparse
import ast
import json
import logging
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Literal, Sequence, Callable, cast

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
    get_torch_dnn_eta_model_configs,
    get_libffm_eta_model_configs,
    get_reduced_eta_model_configs,
    get_torch_mlp_eta_model_configs,
    predict_eta,
    resolve_torch_device,
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
    tau_selection: Literal["cN", "theory"] = "cN"
    use_reduced_grid: bool = True
    hyperparameter: Literal["grid", "fixed"] = "grid"
    eta_models: tuple[str, ...] = ("logreg", "svm_rbf", "rf", "mlp")
    use_gpu: bool = False
    torch_device: str = "auto"
    ffm_train_path: str = "ffm-train"
    ffm_predict_path: str = "ffm-predict"
    ffm_threads: int = 1

    # simulation dataset config
    sim_d: int = 2
    sim_sigma: float = 1.0
    sim_mean_shift: float = 2.0
    sim_rmax_alpha: float = 1e-6
    sim_n_train: int = 4000
    sim_n_val: int = 2000
    sim_n_threshold: int = 2000
    n_total: int | None = None
    ratio_train: float | None = None
    ratio_val: float | None = None
    ratio_threshold: float | None = None
    ratio_final: float | None = None
    real_data_name: str | None = None
    criteo_search_drop_product_price: bool = False
    collect_score_distribution: bool = False

    # output
    analysis: str = "eta_model_comparison"
    output_root: str = "./results"


FeatureMatrix = np.ndarray | pd.DataFrame


def _resolve_eta_model_configs(cfg: EtaExperimentConfig) -> list[EtaModelConfig]:
    requested: list[str] = [str(name).strip() for name in cfg.eta_models if str(name).strip()]
    if len(requested) == 0:
        raise ValueError("eta_models must be non-empty.")

    if "all" in requested:
        requested = ["logreg", "svm_rbf", "rf", "mlp", "ffm"]
    if (
        cfg.hyperparameter == "fixed"
        and requested == ["logreg", "svm_rbf", "rf", "mlp"]
    ):
        requested = ["logreg"]

    needs_torch: bool = bool(
        cfg.use_gpu or "torch_mlp" in requested or "torch_dnn" in requested
    )
    resolved_torch_device: str = resolve_torch_device(cfg.torch_device) if needs_torch else "cpu"
    if cfg.use_gpu:
        if resolved_torch_device != "cpu":
            requested = ["torch_mlp" if name == "mlp" else name for name in requested]
            logging.info(
                "--use_gpu enabled: replacing sklearn mlp with torch_mlp on device=%s",
                resolved_torch_device,
            )
        else:
            logging.info(
                "--use_gpu enabled, but no CUDA/MPS backend is available; keeping CPU sklearn mlp."
            )

    if cfg.hyperparameter == "fixed":
        base_cfgs: list[EtaModelConfig] = get_fixed_logreg_eta_model_configs(c_value=0.1)
    else:
        base_cfgs = (
            get_reduced_eta_model_configs()
            if cfg.use_reduced_grid
            else get_default_eta_model_configs()
        )

    by_name: dict[str, EtaModelConfig] = {model_cfg.name: model_cfg for model_cfg in base_cfgs}
    if "torch_mlp" in requested:
        by_name.update(
            {
                model_cfg.name: model_cfg
                for model_cfg in get_torch_mlp_eta_model_configs(
                    device=cfg.torch_device,
                    reduced=cfg.use_reduced_grid,
                )
            }
        )
    if "torch_dnn" in requested:
        by_name.update(
            {
                model_cfg.name: model_cfg
                for model_cfg in get_torch_dnn_eta_model_configs(
                    device=cfg.torch_device,
                    reduced=cfg.use_reduced_grid,
                )
            }
        )
    if "ffm" in requested:
        by_name.update(
            {
                model_cfg.name: model_cfg
                for model_cfg in get_libffm_eta_model_configs(
                    ffm_train_path=cfg.ffm_train_path,
                    ffm_predict_path=cfg.ffm_predict_path,
                    threads=cfg.ffm_threads,
                    reduced=cfg.use_reduced_grid,
                )
            }
        )

    unknown: list[str] = [name for name in requested if name not in by_name]
    if unknown:
        if cfg.hyperparameter == "fixed":
            raise ValueError(
                "--hyperparameter fixed only defines logreg C=0.1. "
                f"Unknown or unavailable eta_models under this setting: {unknown}."
            )
        raise ValueError(f"Unknown eta_models: {unknown}.")

    selected: list[EtaModelConfig] = [by_name[name] for name in requested]
    return selected


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


def _validate_ratio_values(ratio: Sequence[float]) -> tuple[float, float, float, float]:
    if len(ratio) == 3:
        r_train, r_val, r_final = (float(r) for r in ratio)
        if min(r_train, r_val, r_final) <= 0.0:
            raise ValueError(f"N_ratio values must be positive, got {ratio}.")
        if not np.isclose(r_train + r_val + r_final, 1.0, atol=1e-8):
            raise ValueError(f"N_ratio must sum to 1.0, got {ratio}.")
        return r_train, r_val / 2.0, r_val / 2.0, r_final
    if len(ratio) != 4:
        raise ValueError(f"N_ratio must contain 4 values (train, model_val, threshold, final), got {ratio}.")
    r_train, r_val, r_threshold, r_final = (float(r) for r in ratio)
    if min(r_train, r_val, r_threshold, r_final) <= 0.0:
        raise ValueError(f"N_ratio values must be positive, got {ratio}.")
    if not np.isclose(r_train + r_val + r_threshold + r_final, 1.0, atol=1e-8):
        raise ValueError(f"N_ratio must sum to 1.0, got {ratio}.")
    return r_train, r_val, r_threshold, r_final


def _parse_n_ratio_args(
    raw_values: Sequence[str] | None,
) -> list[tuple[float, float, float, float]] | None:
    if raw_values is None or len(raw_values) == 0:
        return None

    text: str = " ".join(raw_values).strip()
    if text.startswith("[") or text.startswith("("):
        parsed = ast.literal_eval(text)
        if (
            isinstance(parsed, tuple)
            and len(parsed) in (3, 4)
            and all(isinstance(v, (int, float)) for v in parsed)
        ):
            return [_validate_ratio_values(tuple(float(v) for v in parsed))]
        if not isinstance(parsed, (list, tuple)):
            raise ValueError(f"Could not parse N_ratio from: {text}")
        return [
            _validate_ratio_values(tuple(float(v) for v in item)) for item in parsed
        ]

    ratios: list[tuple[float, float, float, float]] = []
    for token in raw_values:
        cleaned: str = token.strip().strip("[]()")
        parts: list[str] = [p.strip() for p in cleaned.split(",") if p.strip()]
        if len(parts) not in (3, 4):
            raise ValueError(
                "Each N_ratio token must look like '0.2,0.2,0.2,0.4' "
                f"for (train, model_val, threshold, final). Got: {token}"
            )
        ratios.append(_validate_ratio_values(tuple(float(p) for p in parts)))
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
    X_train: FeatureMatrix,
    y_train: np.ndarray,
    X_val: FeatureMatrix,
    y_val: np.ndarray,
    X_final: FeatureMatrix,
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
    ratio: tuple[float, float, float, float],
    nb_trials_override: int | None,
    sim_n_train_override: int | None,
    sim_n_val_override: int | None,
    sim_n_threshold_override: int | None,
) -> tuple[int, int, int, int]:
    ratio_train, ratio_val, ratio_threshold, ratio_final = ratio

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
    sim_n_threshold: int = (
        int(sim_n_threshold_override)
        if sim_n_threshold_override is not None
        else int(round(n_total * ratio_threshold))
    )
    if nb_trials_override is not None:
        nb_trials = int(nb_trials_override)
    elif (
        sim_n_train_override is None
        and sim_n_val_override is None
        and sim_n_threshold_override is None
    ):
        nb_trials = int(n_total - sim_n_train - sim_n_val - sim_n_threshold)
    else:
        nb_trials = int(round(n_total * ratio_final))

    if min(sim_n_train, sim_n_val, sim_n_threshold, nb_trials) <= 0:
        raise ValueError(
            "Resolved counts must be positive. "
            f"Got sim_n_train={sim_n_train}, sim_n_val={sim_n_val}, "
            f"sim_n_threshold={sim_n_threshold}, nb_trials={nb_trials}."
        )
    return sim_n_train, sim_n_val, sim_n_threshold, nb_trials


def _build_ratio_cfgs(
    *,
    base_cfg: EtaExperimentConfig,
    n_total: int | None,
    ratios: list[tuple[float, float, float, float]] | None,
    nb_trials_override: int | None,
    sim_n_train_override: int | None,
    sim_n_val_override: int | None,
    sim_n_threshold_override: int | None,
) -> list[EtaExperimentConfig]:
    if n_total is None:
        return [base_cfg]

    if ratios is None or len(ratios) == 0:
        if (
            nb_trials_override is not None
            and sim_n_train_override is not None
            and sim_n_val_override is not None
            and sim_n_threshold_override is not None
        ):
            return [replace(base_cfg, n_total=int(n_total))]
        raise ValueError(
            "When N_total is set, N_ratio must also be set unless all four counts are explicitly overridden."
        )

    cfgs: list[EtaExperimentConfig] = []
    for ratio in ratios:
        ratio_train, ratio_val, ratio_threshold, ratio_final = ratio
        sim_n_train, sim_n_val, sim_n_threshold, nb_trials = _resolve_count_plan(
            n_total=int(n_total),
            ratio=ratio,
            nb_trials_override=nb_trials_override,
            sim_n_train_override=sim_n_train_override,
            sim_n_val_override=sim_n_val_override,
            sim_n_threshold_override=sim_n_threshold_override,
        )
        analysis_suffix: str = (
            f"_Ntotal={int(n_total)}"
            f"_r={_format_ratio_for_path(ratio_train)}-{_format_ratio_for_path(ratio_val)}"
            f"-{_format_ratio_for_path(ratio_threshold)}-{_format_ratio_for_path(ratio_final)}"
        )
        cfgs.append(
            replace(
                base_cfg,
                nb_trials=nb_trials,
                sim_n_train=sim_n_train,
                sim_n_val=sim_n_val,
                sim_n_threshold=sim_n_threshold,
                n_total=int(n_total),
                ratio_train=ratio_train,
                ratio_val=ratio_val,
                ratio_threshold=ratio_threshold,
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
    real_data_name: str | None = None,
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

    num_features: int = sample_df.shape[1] - 1
    if real_data_name is not None and str(real_data_name).strip().lower() == "criteosearch":
        if num_features != 20:
            raise ValueError(
                "CriteoSearch schema expects exactly 20 feature columns after the label. "
                f"Got {num_features}."
            )
        return ["numeric", "numeric", "numeric"] + ["categorical"] * 17
    if real_data_name is not None and str(real_data_name).strip().lower() == "avazu":
        if num_features != 22:
            raise ValueError(
                "Avazu schema expects exactly 22 feature columns after the label. "
                f"Got {num_features}."
            )
        return ["categorical"] * 22

    kinds: list[str] = []
    for col_idx in range(1, sample_df.shape[1]):
        col: pd.Series[str] = sample_df.iloc[:, col_idx].astype(str).str.strip()
        numeric: pd.Series   = pd.to_numeric(col, errors="coerce")
        kinds.append("numeric" if bool(numeric.notna().all()) else "categorical")
    return kinds


def _get_real_csv_feature_names(
    *,
    real_data_name: str | None,
    num_features: int,
) -> list[str]:
    if real_data_name is not None and str(real_data_name).strip().lower() == "criteosearch":
        expected_names: list[str] = [
            "click_timestamp",
            "nb_clicks_1week",
            "product_price",
            "product_age_group",
            "device_type",
            "audience_id",
            "product_gender",
            "product_brand",
            "product_category1",
            "product_category2",
            "product_category3",
            "product_category4",
            "product_category5",
            "product_category6",
            "product_category7",
            "product_country",
            "product_id",
            "product_title",
            "partner_id",
            "user_id",
        ]
        if num_features != len(expected_names):
            raise ValueError(
                "CriteoSearch feature naming expects exactly 20 feature columns after the label. "
                f"Got {num_features}."
            )
        return expected_names
    if real_data_name is not None and str(real_data_name).strip().lower() == "avazu":
        expected_names = [
            "hour",
            "C1",
            "banner_pos",
            "site_id",
            "site_domain",
            "site_category",
            "app_id",
            "app_domain",
            "app_category",
            "device_id",
            "device_ip",
            "device_model",
            "device_type",
            "device_conn_type",
            "C14",
            "C15",
            "C16",
            "C17",
            "C18",
            "C19",
            "C20",
            "C21",
        ]
        if num_features != len(expected_names):
            raise ValueError(
                "Avazu feature naming expects exactly 22 feature columns after the label. "
                f"Got {num_features}."
            )
        return expected_names
    return [f"f{feature_idx}" for feature_idx in range(num_features)]


def _coerce_real_csv_frame(
    *,
    frame: pd.DataFrame,
    feature_kinds: Sequence[str],
    feature_names: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    if frame.shape[1] != len(feature_kinds) + 1:
        raise ValueError(
            "Feature kind schema does not match csv width. "
            f"frame.shape={frame.shape}, len(feature_kinds)={len(feature_kinds)}."
        )
    if feature_names is None:
        feature_name_list: list[str] = [f"f{feature_idx}" for feature_idx in range(len(feature_kinds))]
    else:
        feature_name_list = list(feature_names)
        if len(feature_name_list) != len(feature_kinds):
            raise ValueError(
                "Feature names do not match feature kinds. "
                f"len(feature_names)={len(feature_name_list)}, len(feature_kinds)={len(feature_kinds)}."
            )

    y: np.ndarray = pd.to_numeric(frame.iloc[:, 0], errors="raise").to_numpy(dtype=np.int64, copy=False)
    columns: dict[str, pd.Series] = {}

    for feature_idx, kind in enumerate(feature_kinds, start=1):
        col_name: str = feature_name_list[feature_idx - 1]
        col: pd.Series = frame.iloc[:, feature_idx].astype(str).str.strip()
        if kind == "numeric":
            columns[col_name] = pd.to_numeric(col, errors="raise").astype(np.float32)
        elif kind == "categorical":
            columns[col_name] = col.astype("category")
        else:
            raise ValueError(f"Unknown feature kind: {kind}")
    return pd.DataFrame(columns), y


def _is_criteo_search_dataset(real_data_name: str | None) -> bool:
    return real_data_name is not None and str(real_data_name).strip().lower() == "criteosearch"


def _normalize_categorical_missing(series: pd.Series) -> pd.Series:
    s: pd.Series = series.astype(str).str.strip()
    return s.mask(s.isin(["", "-1", "nan", "None"]), "__MISSING__")


def _add_log_numeric_feature(
    frame: pd.DataFrame,
    *,
    source_col: str,
    missing_if_nonpositive: bool = False,
) -> None:
    values: pd.Series = pd.to_numeric(frame[source_col], errors="coerce")
    missing: pd.Series = values.isna() | (values < 0)
    if missing_if_nonpositive:
        missing = missing | (values <= 0)
    clean: pd.Series = values.mask(missing, 0.0).clip(lower=0.0)
    frame[f"{source_col}_missing"] = missing.astype(np.int8)
    frame[f"{source_col}_log1p"] = np.log1p(clean).astype(np.float32)


def _preprocess_criteo_search_features(X: pd.DataFrame) -> pd.DataFrame:
    """
    Feature engineering for Criteo Sponsored Search conversion data.

    The raw dataset is not the Kaggle CTR table: it has a Unix click timestamp,
    sparse price/click-count fields, many hashed categorical ids, and a
    product_title field that is already a whitespace-separated token list.
    """
    required_cols: set[str] = {
        "click_timestamp",
        "nb_clicks_1week",
        "product_title",
    }
    missing_cols: set[str] = required_cols.difference(str(col) for col in X.columns)
    if missing_cols:
        raise ValueError(f"CriteoSearch preprocessing missing columns: {sorted(missing_cols)}")

    out: pd.DataFrame = X.copy(deep=False)

    timestamp: pd.Series = pd.to_numeric(out["click_timestamp"], errors="coerce")
    valid_timestamp: pd.Series = timestamp.notna() & (timestamp > 0)
    dt: pd.Series = pd.to_datetime(timestamp.where(valid_timestamp), unit="s", utc=True, errors="coerce")
    day_index: pd.Series = np.floor((timestamp - float(timestamp[valid_timestamp].min())) / 86400.0)    # type: ignore
    day_index = day_index.where(valid_timestamp, -1)

    out["click_hour"] = dt.dt.hour.fillna(-1).astype(np.int16).astype("category")
    out["click_dayofweek"] = dt.dt.dayofweek.fillna(-1).astype(np.int16).astype("category")
    out["click_day_index"] = day_index.fillna(-1).astype(np.int16).astype("category")
    out["click_is_weekend"] = dt.dt.dayofweek.isin([5, 6]).fillna(False).astype(np.int8)

    _add_log_numeric_feature(out, source_col="nb_clicks_1week")
    if "product_price" in out.columns:
        _add_log_numeric_feature(out, source_col="product_price", missing_if_nonpositive=True)

    categorical_cols: list[str] = out.select_dtypes(exclude=np.number).columns.tolist()
    for col in categorical_cols:
        normalized: pd.Series = _normalize_categorical_missing(out[col])
        if col == "product_title":
            normalized = normalized.str.replace(r"\s+", " ", regex=True)
            out[col] = normalized.astype(str)
        else:
            out[col] = normalized.astype("category")

    return out.drop(columns=["click_timestamp", "nb_clicks_1week", "product_price"], errors="ignore")


def _preprocess_real_features(
    X: FeatureMatrix,
    *,
    real_data_name: str | None,
    criteo_search_drop_product_price: bool = False,
) -> FeatureMatrix:
    if _is_criteo_search_dataset(real_data_name):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("CriteoSearch preprocessing expects a pandas DataFrame.")
        if criteo_search_drop_product_price:
            X = X.drop(columns=["product_price"])
        return _preprocess_criteo_search_features(X)
    return X


def _load_mixed_real_csv_all_rows(
    *,
    csv_path: str,
    total_rows: int,
    feature_kinds: Sequence[str],
    feature_names: Sequence[str] | None = None,
    chunksize: int = 100_000,
) -> tuple[pd.DataFrame, np.ndarray]:
    X_parts: list[pd.DataFrame] = []
    y_parts: list[np.ndarray] = []

    for chunk in pd.read_csv(
        csv_path,
        header=None,
        chunksize=chunksize,
        dtype=str,
        keep_default_na=False,
        na_filter=False,
    ):
        X_chunk, y_chunk = _coerce_real_csv_frame(
            frame=chunk,
            feature_kinds=feature_kinds,
            feature_names=feature_names,
        )
        X_parts.append(X_chunk)
        y_parts.append(y_chunk)

    loaded_rows: int = sum(len(part) for part in X_parts)
    if loaded_rows != total_rows:
        raise ValueError(f"Expected {total_rows} rows, but loaded {loaded_rows}.")
    X: pd.DataFrame = pd.concat(X_parts, axis=0, ignore_index=True)
    feature_name_list: list[str]
    if feature_names is None:
        feature_name_list = [f"f{feature_idx}" for feature_idx in range(len(feature_kinds))]
    else:
        feature_name_list = list(feature_names)
    for feature_idx, kind in enumerate(feature_kinds):
        if kind == "categorical":
            X[feature_name_list[feature_idx]] = X[feature_name_list[feature_idx]].astype("category")
    return X, np.concatenate(y_parts, axis=0)


def load_susy_real_data_split(
    *,
    csv_path: str,
    n_train: int,
    n_val: int,
    n_threshold: int,
    n_final: int,
    seed: int,
    real_data_name: str | None = None,
    criteo_search_drop_product_price: bool = False,
) -> tuple[
    FeatureMatrix,
    np.ndarray,
    FeatureMatrix,
    np.ndarray,
    FeatureMatrix,
    np.ndarray,
    FeatureMatrix,
    np.ndarray,
]:
    n_total = int(n_train + n_val + n_threshold + n_final)
    total_rows: int = _count_csv_rows(csv_path)
    logging.info(f"[load_susy] total_rows={total_rows}, requested={n_total}")
    feature_kinds: list[str] = _infer_real_csv_feature_kinds(
        csv_path=csv_path,
        real_data_name=real_data_name,
    )
    feature_names: list[str] = _get_real_csv_feature_names(
        real_data_name=real_data_name,
        num_features=len(feature_kinds),
    )
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
        X: FeatureMatrix = sampled[:, 1:].astype(np.float32)
    else:
        if n_total == total_rows:
            X, y = _load_mixed_real_csv_all_rows(
                csv_path=csv_path,
                total_rows=total_rows,
                feature_kinds=feature_kinds,
                feature_names=feature_names,
            )   # type: ignore
        else:
            sampled_raw: np.ndarray = _reservoir_sample_csv_rows_raw(
                csv_path=csv_path,
                n_rows=n_total,
                seed=seed,
            )
            sampled_frame: pd.DataFrame = pd.DataFrame(sampled_raw)
            X, y = _coerce_real_csv_frame(
                frame=sampled_frame,
                feature_kinds=feature_kinds,
                feature_names=feature_names,
            ) # type: ignore

    X = _preprocess_real_features(
        X,
        real_data_name=real_data_name,
        criteo_search_drop_product_price=criteo_search_drop_product_price,
    )

    X_train, X_rest, y_train, y_rest = train_test_split(
        X,
        y,
        train_size=int(n_train),
        stratify=y,
        random_state=int(seed),
    )
    X_val, X_rest2, y_val, y_rest2 = train_test_split(
        X_rest,
        y_rest,
        train_size=int(n_val),
        stratify=y_rest,
        random_state=int(seed) + 1,
    )
    X_threshold, X_final, y_threshold, y_final = train_test_split(
        X_rest2,
        y_rest2,
        train_size=int(n_threshold),
        stratify=y_rest2,
        random_state=int(seed) + 2,
    )
    return X_train, y_train, X_val, y_val, X_threshold, y_threshold, X_final, y_final


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
    tau_selection: str,
    attack_for_selection: str,
    attack_for_report: str,
    sim_d: int | None,
    sim_sigma: float | None,
    sim_mean_shift: float | None,
    sim_n_train: int | None,
    sim_n_val: int | None,
    sim_n_threshold: int | None,
    n_total: int | None,
    ratio_train: float | None,
    ratio_val: float | None,
    ratio_threshold: float | None,
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
    rows["tau_selection"].append(tau_selection)
    rows["attack_for_selection"].append(attack_for_selection)
    rows["attack_for_report"].append(attack_for_report)

    rows["sim_d"].append(sim_d)
    rows["sim_sigma"].append(sim_sigma)
    rows["sim_mean_shift"].append(sim_mean_shift)
    rows["sim_n_train"].append(sim_n_train)
    rows["sim_n_val"].append(sim_n_val)
    rows["sim_n_threshold"].append(sim_n_threshold)
    rows["n_total"].append(n_total)
    rows["ratio_train"].append(ratio_train)
    rows["ratio_val"].append(ratio_val)
    rows["ratio_threshold"].append(ratio_threshold)
    rows["ratio_final"].append(ratio_final)

    rows["metric"].append(metric)
    rows["value"].append(value)
    rows["params_json"].append(params_json)


def run_eta_model_experiments(
    *,
    cfg: EtaExperimentConfig,
    lst_seed: Iterable[int],
    # real-data mode (optional). If provided, simulation generation is skipped.
    X_train: FeatureMatrix | None = None,
    y_train: np.ndarray | None = None,
    X_val: FeatureMatrix | None = None,
    y_val: np.ndarray | None = None,
    X_threshold: FeatureMatrix | None = None,
    y_threshold: np.ndarray | None = None,
    X_final: FeatureMatrix | None = None,
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
        - In real-data mode, pass X_train/y_train, X_val/y_val, X_threshold/y_threshold, X_final/y_final.
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

    eta_model_cfgs: list[EtaModelConfig] = _resolve_eta_model_configs(cfg)

    logging.info(
        "analysis=%s selection=%s tau_selection=%s report_attacks=%s eps=%s reduced_grid=%s hyperparameter=%s eta_models=%s use_gpu=%s torch_device=%s",
        cfg.analysis,
        cfg.selection,
        cfg.tau_selection,
        report_attacks,
        eps_list,
        cfg.use_reduced_grid,
        cfg.hyperparameter,
        [model_cfg.name for model_cfg in eta_model_cfgs],
        cfg.use_gpu,
        cfg.torch_device,
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
        "tau_selection": [],
        "attack_for_selection": [],
        "attack_for_report": [],
        "sim_d": [],
        "sim_sigma": [],
        "sim_mean_shift": [],
        "sim_n_train": [],
        "sim_n_val": [],
        "sim_n_threshold": [],
        "n_total": [],
        "ratio_train": [],
        "ratio_val": [],
        "ratio_threshold": [],
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
            sim_n_threshold=cfg.sim_n_threshold,
        )
    else:
        if (
            X_val is None
            or y_val is None
            or X_threshold is None
            or y_threshold is None
            or X_final is None
            or y_final is None
        ):
            raise ValueError("Real-data mode requires X_val/y_val/X_threshold/y_threshold/X_final/y_final.")
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
            sim_n_threshold=X_threshold.shape[0],
        )

    sim_d_row: int = int(sim_meta["sim_d"] if sim_meta["sim_d"] is not None else -1)
    sim_sigma_row: float | None = sim_meta["sim_sigma"]
    sim_mean_shift_row: float | None = sim_meta["sim_mean_shift"]
    sim_n_train_row: int = int(sim_meta["sim_n_train"] if sim_meta["sim_n_train"] is not None else -1)
    sim_n_val_row: int = int(sim_meta["sim_n_val"] if sim_meta["sim_n_val"] is not None else -1)
    sim_n_threshold_row: int = int(
        sim_meta["sim_n_threshold"] if sim_meta["sim_n_threshold"] is not None else -1
    )
    output_root: Path = Path(cfg.output_root)
    dist_output_dir: Path = output_root / "score_distributions"
    score_dist_data_name: str = cfg.real_data_name or "real"
    tau_selection_suffix: str = "" if cfg.tau_selection == "cN" else f"_tau={cfg.tau_selection}"

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
            tau_selection=str(cfg.tau_selection),
            attack_for_selection=attack_for_selection,
            attack_for_report=attack_for_report,
            sim_d=sim_d_row,
            sim_sigma=sim_sigma_row,
            sim_mean_shift=sim_mean_shift_row,
            sim_n_train=sim_n_train_row,
            sim_n_val=sim_n_val_row,
            sim_n_threshold=sim_n_threshold_row,
            n_total=cfg.n_total,
            ratio_train=cfg.ratio_train,
            ratio_val=cfg.ratio_val,
            ratio_threshold=cfg.ratio_threshold,
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
        real_threshold_X=X_threshold,
        real_threshold_y=y_threshold,
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
        indirect_model_cache: dict[str, dict[str, dict[str, object]]] = {}
        for epsilon in eps_list:
            logging.info(
                "Running eta-model comparison for seed=%s, epsilon=%s",
                seed,
                epsilon,
            )
            auditor.set_params(epsilon=float(epsilon), k=cfg.k, random_state=int(seed))

            prefit_cache: dict[str, dict[str, dict[str, object]]] | None = None
            if "indirect_LRT_hat" in report_attacks and "indirect_LRT_hat" in indirect_model_cache:
                prefit_cache = {"indirect_LRT_hat": indirect_model_cache["indirect_LRT_hat"]}

            out: dict = auditor.run_eta_model_comparison_4way(
                seed=int(seed),
                selection=cfg.selection,
                tau_selection=cfg.tau_selection,
                report_attacks=report_attacks,
                eta_model_cfgs=eta_model_cfgs,
                sim_n_train=cfg.sim_n_train if X_train is None else None,
                sim_n_val=cfg.sim_n_val if X_train is None else None,
                sim_n_threshold=cfg.sim_n_threshold if X_train is None else None,
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                X_threshold=X_threshold,
                y_threshold=y_threshold,
                prefit_model_best_by_attack=prefit_cache,  # type: ignore[arg-type]
                y_alt=cfg.y_alt,
                y_null=cfg.y_null,
            )
            model_best_by_attack = cast(
                dict[str, dict[str, dict[str, object]]],
                out.get("model_best_by_attack", {}),
            )
            if "indirect_LRT_hat" in report_attacks and "indirect_LRT_hat" not in indirect_model_cache:
                indirect_best: dict[str, dict[str, object]] | None = model_best_by_attack.get("indirect_LRT_hat")
                if indirect_best is not None:
                    indirect_model_cache["indirect_LRT_hat"] = indirect_best
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
                            ("eta_prior_y0", "prior_0"),
                            ("eta_prior_y1", "prior_1"),
                            ("eta_log_prior_correction", "log_prior_0_over_prior_1"),
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
                        for metric_key in (
                            "tau_grid_size",
                            "tau_grid_feasible_size",
                            "tau_cp_one_sided_alpha",
                            "tau_selected_index",
                        ):
                            if metric_key not in tau_q:
                                continue
                            append_row(
                                seed=int(seed),
                                protocol=protocol,
                                epsilon=epsilon,
                                attack_for_selection=str(attack_name),
                                metric=metric_key,
                                value=float(tau_q[metric_key]),
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
                    if "indirect" in str(attack_name).lower():
                        append_row(
                            seed=int(seed),
                            protocol=protocol,
                            epsilon=epsilon,
                            attack_for_selection=str(attack_name),
                            metric="test_eps_emp_plus_rr",
                            value=float(test["eps_emp"]) + float(epsilon),
                            params_json=params_json_by_attack.get(str(attack_name)),
                            attack_for_report=str(attack_name),
                        )
                        append_row(
                            seed=int(seed),
                            protocol=protocol,
                            epsilon=epsilon,
                            attack_for_selection=str(attack_name),
                            metric="test_eps_lower_plus_rr",
                            value=float(eps_lo) + float(epsilon),
                            params_json=params_json_by_attack.get(str(attack_name)),
                            attack_for_report=str(attack_name),
                        )
                        append_row(
                            seed=int(seed),
                            protocol=protocol,
                            epsilon=epsilon,
                            attack_for_selection=str(attack_name),
                            metric="test_eps_upper_plus_rr",
                            value=float(eps_hi) + float(epsilon),
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
        for per_model_best in indirect_model_cache.values():
            for best in per_model_best.values():
                cleanup_model: Any | None = getattr(best.get("model"), "cleanup", None)
                if callable(cleanup_model):
                    cleanup_model()

    df = pd.DataFrame(rows)

    if X_train is None:
        out_csv: str = (
            f"{output_root}/eta_hat_shift={cfg.sim_mean_shift}_c={cfg.c}"
            f"_f={cfg.nb_trials}_t={cfg.sim_n_train}_v={cfg.sim_n_val}"
            f"_th={cfg.sim_n_threshold}_d={cfg.sim_d}{tau_selection_suffix}.csv"
        )
    else:
        data_name: str = cfg.real_data_name or "real"
        if X_val is None or X_threshold is None:
            raise ValueError("Real-data mode requires X_val and X_threshold when saving results.")
        out_csv = (
            f"{output_root}/eta_hat_{data_name}_c={cfg.c}"
            f"_f={cfg.nb_trials}_t={X_train.shape[0]}_v={X_val.shape[0]}"
            f"_th={X_threshold.shape[0]}_d={X_train.shape[1]}{tau_selection_suffix}.csv"
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
    parser.add_argument(
        "--tau_selection",
        choices=["cN", "theory"],
        default="cN",
        help=(
            "Threshold selection strategy: cN keeps the current DP-Sniper "
            "ceil(cN) null order statistic; theory builds a finite threshold "
            "grid from threshold null scores and maximizes Bonferroni-adjusted "
            "CP-LCB on the final split."
        ),
    )
    parser.add_argument("--use_reduced_grid", action="store_true")
    parser.add_argument(
        "--hyperparameter",
        choices=["grid", "fixed"],
        default="grid",
        help="Hyperparameter strategy: grid search (all models) or fixed logreg C=0.1.",
    )
    parser.add_argument(
        "--eta_models",
        nargs="+",
        choices=[
            "logreg",
            "svm_rbf",
            "rf",
            "mlp",
            "ffm",
            "torch_mlp",
            "torch_dnn",
            "all",
        ],
        default=["logreg", "svm_rbf", "rf", "mlp"],
        help=(
            "Eta model families to run. Add 'ffm' to use libffm, add "
            "'torch_mlp' or 'torch_dnn' to use optional PyTorch neural nets, "
            "or use 'all' for non-torch baseline families."
        ),
    )
    parser.add_argument(
        "--use_gpu",
        action="store_true",
        help=(
            "If a CUDA/MPS backend is available, replace requested sklearn mlp "
            "with torch_mlp so the MLP can run on GPU. Other sklearn models stay CPU."
        ),
    )
    parser.add_argument(
        "--torch_device",
        type=str,
        default="auto",
        help="Device for torch_mlp. Use auto, cpu, cuda, cuda:N, or mps.",
    )
    parser.add_argument(
        "--ffm_train_path",
        type=str,
        default="ffm-train",
        help="Path to libffm ffm-train executable, or a command available on PATH.",
    )
    parser.add_argument(
        "--ffm_predict_path",
        type=str,
        default="ffm-predict",
        help="Path to libffm ffm-predict executable, or a command available on PATH.",
    )
    parser.add_argument(
        "--ffm_threads",
        type=int,
        default=1,
        help="Number of threads passed to ffm-train -s. Use 1 for deterministic libffm runs.",
    )
    parser.add_argument("--sim_d", type=int, default=2)
    parser.add_argument("--sim_sigma", type=float, default=1.0)
    parser.add_argument("--sim_mean_shift", type=float, default=0.1)
    parser.add_argument("--sim_rmax_alpha", type=float, default=1e-6)
    parser.add_argument("--sim_n_train", type=int, default=None)
    parser.add_argument("--sim_n_val", type=int, default=None)
    parser.add_argument("--sim_n_threshold", type=int, default=None)
    parser.add_argument("--real_data", action="store_true")
    parser.add_argument(
        "--real_data_path",
        type=str,
        default="./data/SUSY/SUSY.csv",
    )
    parser.add_argument("--real_data_name", type=str, default="SUSY")
    parser.add_argument(
        "--criteo_search_drop_product_price",
        action="store_true",
        help=(
            "If set with --real_data_name CriteoSearch, drop product_price before "
            "CriteoSearch-specific feature engineering."
        ),
    )
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
            "Ratio settings for (train, model_val, threshold, final). "
            "Example: --N_ratio 0.2,0.2,0.2,0.4 "
            "or --N_ratio '[(0.2,0.2,0.2,0.4),(0.1,0.2,0.2,0.5)]'. "
            "Legacy 3-value ratios are accepted by splitting the old validation share "
            "into model_val and threshold."
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
        and any(v is not None for v in (args.nb_trials, args.sim_n_train, args.sim_n_val, args.sim_n_threshold))
    ):
        raise ValueError(
            "When --N_total all is used, do not also specify --nb_trials/--sim_n_train/--sim_n_val/--sim_n_threshold."
        )

    n_total_value: int | None = _parse_n_total_arg(
        args.N_total,
        real_data=bool(args.real_data),
        real_data_path=str(args.real_data_path),
    )
    n_ratio_list: list[tuple[float, float, float, float]] | None = _parse_n_ratio_args(args.N_ratio)
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
        tau_selection=cast(Literal["cN", "theory"], args.tau_selection),
        evaluate_both_reports=bool(args.evaluate_both_reports),
        use_reduced_grid=bool(args.use_reduced_grid),
        hyperparameter=cast(Literal["grid", "fixed"], args.hyperparameter),
        eta_models=tuple(str(name) for name in args.eta_models),
        use_gpu=bool(args.use_gpu),
        torch_device=str(args.torch_device),
        ffm_train_path=str(args.ffm_train_path),
        ffm_predict_path=str(args.ffm_predict_path),
        ffm_threads=int(args.ffm_threads),
        sim_d=args.sim_d,
        sim_sigma=args.sim_sigma,
        sim_mean_shift=args.sim_mean_shift,
        sim_rmax_alpha=args.sim_rmax_alpha,
        sim_n_train=int(args.sim_n_train) if args.sim_n_train is not None else 2000,
        sim_n_val=int(args.sim_n_val) if args.sim_n_val is not None else 2000,
        sim_n_threshold=(
            int(args.sim_n_threshold)
            if args.sim_n_threshold is not None
            else int(args.sim_n_val)
            if args.sim_n_val is not None
            else 2000
        ),
        real_data_name=args.real_data_name if args.real_data else None,
        criteo_search_drop_product_price=bool(args.criteo_search_drop_product_price),
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
        sim_n_threshold_override=args.sim_n_threshold,
    )

    for cfg in cfgs:
        if cfg.n_total is not None:
            effective_total: int = cfg.nb_trials + cfg.sim_n_train + cfg.sim_n_val + cfg.sim_n_threshold
            if effective_total != int(cfg.n_total):
                logging.warning(
                    "Effective total (%s) does not match N_total (%s) because explicit overrides were applied.",
                    effective_total,
                    cfg.n_total,
                )
        if args.real_data:
            X_train, y_train, X_val, y_val, X_threshold, y_threshold, X_final, y_final = load_susy_real_data_split(
                csv_path=args.real_data_path,
                n_train=int(cfg.sim_n_train),
                n_val=int(cfg.sim_n_val),
                n_threshold=int(cfg.sim_n_threshold),
                n_final=int(cfg.nb_trials),
                seed=int(args.real_data_seed),
                real_data_name=args.real_data_name,
                criteo_search_drop_product_price=bool(args.criteo_search_drop_product_price),
            )
            df: pd.DataFrame = run_eta_model_experiments(
                cfg=cfg,
                lst_seed=lst_seed,
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                X_threshold=X_threshold,
                y_threshold=y_threshold,
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

# time python experiment_eta_model.py \
#   --real_data \
#   --real_data_path ./data/criteo_search/criteo_search_numeric.csv \
#   --real_data_name CriteoSearch \
#   --output_root ./results/20260401 \
#   --evaluate_both_reports \
#   --hyperparameter fixed \
#   --N_total 3597294 \
#   --N_ratio 0.2,0.2,0.6 \
#   --seed_start 0 \
#   --seed_end 3 \
#   --score_dist

# python experiment_eta_model.py \
#   --real_data \
#   --real_data_path ./data/avazu/avazu_train_label_first.csv \
#   --real_data_name Avazu \
#   --output_root ./results/20260401 \
#   --evaluate_both_reports \
#   --hyperparameter fixed \
#   --N_total 100000 \
#   --N_ratio 0.2,0.2,0.6 \
#   --seed_start 0 \
#   --seed_end 1 \
#   --score_dist

# python experiment_eta_model.py \
#   --real_data \
#   --real_data_path ./data/avazu/avazu_train_label_first.csv \
#   --real_data_name Avazu \
#   --output_root ./results/20260401 \
#   --evaluate_both_reports \
#   --eta_models ffm \
#   --use_reduced_grid \
#   --N_total 100000 \
#   --N_ratio 0.2,0.2,0.6 \
#   --seed_start 0 \
#   --seed_end 1

# python experiment_eta_model.py \
#   --real_data \
#   --real_data_path ./data/criteo_search/criteo_search_numeric.csv \
#   --real_data_name CriteoSearch \
#   --output_root ./results/20260401/criteo_smoke \
#   --evaluate_both_reports \
#   --eta_models ffm \
#   --use_reduced_grid \
#   --N_total 10000 \
#   --N_ratio 0.2,0.2,0.6 \
#   --seed_start 0 \
#   --seed_end 1

# nohup time python experiment_eta_model.py \
#   --real_data \
#   --real_data_path ./data/avazu/avazu_train_label_first.csv \
#   --real_data_name Avazu \
#   --output_root ./results/20260401/avazu_smoke \
#   --evaluate_both_reports \
#   --eta_models ffm \
#   --use_reduced_grid \
#   --N_total 40428967 \
#   --N_ratio 0.2,0.2,0.6 \
#   --seed_start 0 \
#   --seed_end 9 \
#   --score_dist &

# nohup time python experiment_eta_model.py \
#   --real_data \
#   --real_data_path ./data/criteo_search/criteo_search_numeric.csv \
#   --real_data_name CriteoSearch \
#   --criteo_search_drop_product_price \
#   --output_root ./results/20260616/criteo_search_preprocessed_no_price_smoke \
#   --evaluate_both_reports \
#   --use_reduced_grid \
#   --eta_models torch_mlp \
#   --selection eps_lower \
#   --N_total 15995634 \
#   --N_ratio 0.2,0.2,0.2,0.4 \
#   --seed_start 0 \
#   --seed_end 4 \
#   --tau_selection cN \
#   --use_gpu \
#   --torch_device auto \
#   --score_dist &

# CUDA_VISIBLE_DEVICES=0 \
# OMP_NUM_THREADS=4 \
# MKL_NUM_THREADS=4 \
# OPENBLAS_NUM_THREADS=4 \
# NUMEXPR_NUM_THREADS=4 \
# taskset -c 16-19 \
# nohup python experiment_eta_model.py \
#   --real_data \
#   --real_data_path ./data/criteo_search/criteo_search_numeric.csv \
#   --real_data_name CriteoSearch \
#   --eta_models torch_mlp \
#   --torch_device cuda \
#   --use_reduced_grid \
#   --N_total 100000 \
#   --N_ratio 0.2,0.2,0.2,0.4 \
#   --output_root ./results/20260616/criteo_search_preprocessed_smoke_torch_mlp \
#   --evaluate_both_reports \
#   --selection eps_lower \
#   --tau_selection cN \
#   --score_dist \
#   --seed_start 0 \
#   --seed_end 4 \
#   > torch_mlp_criteo.log 2>&1 &

# echo $!