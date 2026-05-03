# eta_models.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC


def clip_eta(e: np.ndarray, eps: float = 1e-15) -> np.ndarray:
    """Numerical safety for eta in (0,1)."""
    e = np.asarray(e, dtype=float)
    return np.clip(e, eps, 1.0 - eps)


@dataclass(frozen=True)
class EtaModelConfig:
    """
    1 つの eta モデルファミリに関する設定情報
    """
    name: str
    param_grid: Sequence[dict[str, Any]]
    needs_scaling: bool
    builder: Callable[[dict[str, Any], int], Any]  # (params, seed) -> estimator


def _build_logreg(params: dict[str, Any], seed: int) -> LogisticRegression:
    print("Building Logistic Regression with params:", params)
    # L2 only; saga handles both dense numeric data and sparse one-hot features.
    return LogisticRegression(
        penalty="l2",
        solver="saga",
        max_iter=5000,
        random_state=seed,
        **params,
    )


def _build_svm_rbf(params: dict[str, Any], seed: int) -> SVC:
    print("Building SVM with params:", params)
    # probability=True enables predict_proba via Platt scaling (slower but convenient)
    return SVC(
        kernel="rbf",
        probability=True,
        random_state=seed,
        **params,
    )


def _build_rf(params: dict[str, Any], seed: int) -> RandomForestClassifier:
    print("Building Random Forest with params:", params)
    return RandomForestClassifier(
        random_state=seed,
        n_jobs=-1,
        **params,
    )


def _build_mlp(params: dict[str, Any], seed: int) -> MLPClassifier:
    print("Building MLP with params:", params)
    return MLPClassifier(
        random_state=seed,
        max_iter=5000,
        early_stopping=True,
        n_iter_no_change=20,
        **params,
    )


def _build_real_data_preprocessor(
    X_sample: pd.DataFrame,
    *,
    needs_scaling: bool,
) -> ColumnTransformer:
    del needs_scaling
    numeric_cols: list[str] = X_sample.select_dtypes(include=np.number).columns.tolist()
    categorical_cols: list[str] = X_sample.select_dtypes(exclude=np.number).columns.tolist()
    transformers: list[tuple[str, Any, Any]] = []

    if len(numeric_cols) > 0:
        transformers.append(("num", "passthrough", numeric_cols))

    text_title_col: str | None = None
    if "product_title" in X_sample.columns and "product_title" in categorical_cols:
        text_title_col = "product_title"
        categorical_cols = [col for col in categorical_cols if col != text_title_col]

    if len(categorical_cols) > 0:
        transformers.append(
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=True),
                categorical_cols,
            )
        )

    if text_title_col is not None:
        transformers.append(
            (
                "title",
                CountVectorizer(binary=True, token_pattern=r"[^ ]+"),
                text_title_col,
            )
        )

    return ColumnTransformer(
        transformers=transformers,
        sparse_threshold=1.0,
    )   # type: ignore -- ColumnTransformer typing is weird and doesn't recognize sparse_threshold argument


def make_eta_pipeline(
    cfg: EtaModelConfig,
    params: dict[str, Any],
    seed: int,
    X_sample: np.ndarray | pd.DataFrame | None = None,
) -> Pipeline:
    """
    Build an sklearn Pipeline for eta(x)=P(Y=1|X=x) estimation.

    Depending on the model configuration, this function optionally applies
    feature standardization before the classifier and returns a unified
    Pipeline with ``fit``/``predict_proba`` interface.
    """
    est: LogisticRegression | SVC | RandomForestClassifier | MLPClassifier = cfg.builder(params, seed)
    if isinstance(X_sample, pd.DataFrame):
        return Pipeline(
            [
                (
                    "preprocess",
                    _build_real_data_preprocessor(X_sample, needs_scaling=cfg.needs_scaling),
                ),
                ("clf", est),
            ]
        )
    if cfg.needs_scaling:
        return Pipeline([("scaler", StandardScaler()), ("clf", est)])
    return Pipeline([("clf", est)])


def get_default_eta_model_configs() -> list[EtaModelConfig]:
    """
    4-model configs (logreg/svm-rbf/rf/mlp) with compact, paper-friendly grids.
    These work for both simulation (d=2) and real data (e.g., d=10, N=4000).
    """
    logreg_grid: list[dict[str, float]] = [{"C": C} for C in [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]]

    svm_grid: list[dict[str, float]] = [
        {"C": C, "gamma": g}
        for C in [1e-2, 1e-1, 1.0, 10.0, 100.0, 1e3]
        for g in [1e-4, 1e-3, 1e-2, 1e-1, 1.0]
    ]

    rf_grid: list[dict[str, Any]] = [
        {
            "n_estimators": 500,
            "max_depth": d,
            "min_samples_leaf": leaf,
            "max_features": mf,
        }
        for d in [None, 5, 10, 20]
        for leaf in [1, 5, 10, 20]
        for mf in ["sqrt", 0.5, 1.0]
    ]

    mlp_grid: list[dict[str, Any]] = [
        {"hidden_layer_sizes": hs, "alpha": a, "learning_rate_init": lr}
        for hs in [(32,), (64,), (128,), (64, 64)]
        for a in [1e-5, 1e-4, 1e-3]  # weight decay
        for lr in [1e-4, 1e-3, 1e-2]
    ]

    return [
        EtaModelConfig("logreg", logreg_grid, True, _build_logreg),
        EtaModelConfig("svm_rbf", svm_grid, True, _build_svm_rbf),
        EtaModelConfig("rf", rf_grid, False, _build_rf),
        EtaModelConfig("mlp", mlp_grid, True, _build_mlp),
    ]


def get_reduced_eta_model_configs() -> list[EtaModelConfig]:
    """
    Reduced search space for faster eta-model tuning.

    This grid is narrowed around parameter regions that were selected in
    prior runs (results/eta_model_results_*.csv) while still keeping
    coverage for different epsilon regimes.
    """
    logreg_grid: list[dict[str, float]] = [{"C": C} for C in [1e-4, 1e-3, 1e-2, 1e-1]]

    svm_grid: list[dict[str, float]] = [
        {"C": C, "gamma": g}
        for C in [1e-2, 1e-1, 1.0, 100.0, 1e3]
        for g in [1e-4, 1e-3, 1e-1, 1.0]
    ]

    rf_grid: list[dict[str, Any]] = [
        {
            "n_estimators": 500,
            "max_depth": d,
            "min_samples_leaf": leaf,
            "max_features": mf,
        }
        for d in [None, 5, 10]
        for leaf in [1, 10]
        for mf in ["sqrt", 1.0]
    ]

    mlp_grid: list[dict[str, Any]] = [
        {"hidden_layer_sizes": hs, "alpha": a, "learning_rate_init": lr}
        for hs in [(128,), (64, 64)]
        for a in [1e-5, 1e-4, 1e-3]
        for lr in [1e-3, 1e-2]
    ]

    return [
        EtaModelConfig("logreg", logreg_grid, True, _build_logreg),
        EtaModelConfig("svm_rbf", svm_grid, True, _build_svm_rbf),
        EtaModelConfig("rf", rf_grid, False, _build_rf),
        EtaModelConfig("mlp", mlp_grid, True, _build_mlp),
    ]


def get_fixed_logreg_eta_model_configs(c_value: float = 0.1) -> list[EtaModelConfig]:
    """
    Fixed hyperparameter setting for logistic regression only.
    """
    logreg_grid: list[dict[str, float]] = [{"C": float(c_value)}]
    return [EtaModelConfig("logreg", logreg_grid, True, _build_logreg)]


def _check_binary_labels(y: np.ndarray) -> None:
    y = np.asarray(y)
    uniq: set[int] = set(np.unique(y).tolist())
    if not uniq.issubset({0, 1}):
        raise ValueError(f"y must be binary in {{0,1}}. got unique={sorted(uniq)}")


def fit_and_select_eta_model(
    *,
    cfg: EtaModelConfig,
    X_train: np.ndarray | pd.DataFrame,
    y_train: np.ndarray,
    X_val: np.ndarray | pd.DataFrame,
    y_val: np.ndarray,
    seed: int,
    score_fn: Callable[[Pipeline], float],
) -> dict[str, Any]:
    """
    Fit+select best hyperparameters for ONE model family.
    score_fn evaluates a fitted model on the validation data, returning a scalar to maximize.
    """
    _check_binary_labels(y_train)
    _check_binary_labels(y_val)

    best: dict[str, Any] = {
        "score": -np.inf,
        "params": None,
        "model": None,
        "cfg_name": cfg.name,
    }

    for params in cfg.param_grid:
        model: Pipeline = make_eta_pipeline(cfg, params, seed=seed, X_sample=X_train)
        model.fit(X_train, y_train)
        s = float(score_fn(model))
        if s > best["score"]:
            best = {
                "score": s,
                "params": dict(params),
                "model": model,
                "cfg_name": cfg.name,
            }

    if best["model"] is None:
        raise RuntimeError(f"Failed to select model for cfg={cfg.name}")
    return best


def predict_eta(model: Any, X: np.ndarray | pd.DataFrame) -> np.ndarray:
    """
    Return eta_hat(x)=P(Y=1|X=x) for binary y.
    Requires model to implement predict_proba.
    """
    proba: np.ndarray = model.predict_proba(X)
    eta: np.ndarray = proba[:, 1]
    return clip_eta(eta)
