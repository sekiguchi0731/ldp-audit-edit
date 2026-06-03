# eta_models.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any, Callable, Iterator, Sequence

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
    direct_input: bool = False


class LibFFMClassifier:
    """
    sklearn-like wrapper around the libffm command-line tools.

    The wrapper owns the conversion from DataFrame/ndarray rows to libffm's
    `<label> <field>:<feature>:<value> ...` format so callers can continue
    passing the same real-data splits used by the sklearn models.
    """

    def __init__(
        self,
        *,
        ffm_train_path: str = "ffm-train",
        ffm_predict_path: str = "ffm-predict",
        lambda_: float = 2e-5,
        k: int = 4,
        iterations: int = 15,
        eta: float = 0.2,
        threads: int = 1,
        auto_stop: bool = True,
        no_norm: bool = False,
        quiet: bool = True,
        keep_tmp: bool = False,
        seed: int | None = None,
    ) -> None:
        self.ffm_train_path: str = ffm_train_path
        self.ffm_predict_path: str = ffm_predict_path
        self.lambda_ = float(lambda_)
        self.k = int(k)
        self.iterations = int(iterations)
        self.eta = float(eta)
        self.threads = int(threads)
        self.auto_stop = bool(auto_stop)
        self.no_norm = bool(no_norm)
        self.quiet = bool(quiet)
        self.keep_tmp = bool(keep_tmp)
        self.seed: int | None = seed

        self._tmpdir: tempfile.TemporaryDirectory[str] | None = None
        self._model_path: Path | None = None
        self._feature_to_id: dict[tuple[str, str], int] = {}
        self._numeric_feature_to_id: dict[str, int] = {}
        self._field_to_id: dict[str, int] = {}
        self._feature_names: list[str] | None = None
        self._is_fitted: bool = False
        self._predict_counter: int = 0

    def fit(self, X: np.ndarray | pd.DataFrame, y: np.ndarray) -> "LibFFMClassifier":
        return self.fit_with_validation(X, y, None, None)

    def fit_with_validation(
        self,
        X: np.ndarray | pd.DataFrame,
        y: np.ndarray,
        X_val: np.ndarray | pd.DataFrame | None,
        y_val: np.ndarray | None,
    ) -> "LibFFMClassifier":
        train_exe: str | None = shutil.which(self.ffm_train_path) if not Path(self.ffm_train_path).exists() else self.ffm_train_path
        predict_exe: str | None = (
            shutil.which(self.ffm_predict_path)
            if not Path(self.ffm_predict_path).exists()
            else self.ffm_predict_path
        )
        if train_exe is None:
            raise FileNotFoundError(
                "ffm-train was not found. Install/build libffm and pass --ffm_train_path "
                "or put ffm-train on PATH."
            )
        if predict_exe is None:
            raise FileNotFoundError(
                "ffm-predict was not found. Install/build libffm and pass --ffm_predict_path "
                "or put ffm-predict on PATH."
            )
        self.ffm_train_path = str(train_exe)
        self.ffm_predict_path = str(predict_exe)

        self._reset_feature_state(X)
        self._tmpdir = tempfile.TemporaryDirectory(prefix="ldp_audit_libffm_")
        tmp_path = Path(self._tmpdir.name)
        train_path: Path = tmp_path / "train.ffm"
        model_path: Path = tmp_path / "model.ffm"
        self._model_path = model_path
        self._write_ffm_file(train_path, X, y, fit=True)

        cmd: list[str] = [
            self.ffm_train_path,
            "-l",
            str(self.lambda_),
            "-k",
            str(self.k),
            "-t",
            str(self.iterations),
            "-r",
            str(self.eta),
            "-s",
            str(self.threads),
        ]
        if self.quiet:
            cmd.append("--quiet")
        if self.no_norm:
            cmd.append("--no-norm")

        if X_val is not None and y_val is not None:
            val_path: Path = tmp_path / "validation.ffm"
            self._write_ffm_file(val_path, X_val, y_val, fit=False)
            cmd.extend(["-p", str(val_path)])
            if self.auto_stop:
                cmd.append("--auto-stop")

        cmd.extend([str(train_path), str(model_path)])
        subprocess.run(cmd, check=True, capture_output=self.quiet, text=True, cwd=tmp_path)
        self._is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        if not self._is_fitted or self._tmpdir is None or self._model_path is None:
            raise RuntimeError("LibFFMClassifier must be fitted before predict_proba.")

        tmp_path = Path(self._tmpdir.name)
        self._predict_counter += 1
        test_path: Path = tmp_path / f"predict_{self._predict_counter}.ffm"
        output_path: Path = tmp_path / f"predict_{self._predict_counter}.out"
        dummy_y: np.ndarray = np.zeros(self._num_rows(X), dtype=np.int64)
        self._write_ffm_file(test_path, X, dummy_y, fit=False)

        subprocess.run(
            [self.ffm_predict_path, str(test_path), str(self._model_path), str(output_path)],
            check=True,
            capture_output=self.quiet,
            text=True,
            cwd=tmp_path,
        )
        p1: np.ndarray = np.loadtxt(output_path, dtype=np.float64)
        p1 = np.atleast_1d(p1)
        p1 = clip_eta(p1)
        return np.column_stack([1.0 - p1, p1])

    def cleanup(self) -> None:
        if self.keep_tmp:
            return
        if self._tmpdir is not None:
            self._tmpdir.cleanup()
            self._tmpdir = None

    def __del__(self) -> None:
        try:
            self.cleanup()
        except Exception:
            pass

    def _reset_feature_state(self, X: np.ndarray | pd.DataFrame) -> None:
        self._feature_to_id = {}
        self._numeric_feature_to_id = {}
        self._field_to_id = {}
        self._feature_names = self._get_feature_names(X)

    def _get_feature_names(self, X: np.ndarray | pd.DataFrame) -> list[str]:
        if isinstance(X, pd.DataFrame):
            return [str(col) for col in X.columns]
        if X.ndim != 2:
            raise ValueError(f"X must be 2-dimensional, got shape={X.shape}.")
        return [f"f{idx}" for idx in range(X.shape[1])]

    def _num_rows(self, X: np.ndarray | pd.DataFrame) -> int:
        return int(X.shape[0])

    def _field_id(self, field_name: str) -> int:
        if field_name not in self._field_to_id:
            self._field_to_id[field_name] = len(self._field_to_id)
        return self._field_to_id[field_name]

    def _next_feature_id(self) -> int:
        return len(self._numeric_feature_to_id) + len(self._feature_to_id)

    def _numeric_feature_id(self, field_name: str) -> int:
        if field_name not in self._numeric_feature_to_id:
            self._numeric_feature_to_id[field_name] = self._next_feature_id()
        return self._numeric_feature_to_id[field_name]

    def _categorical_feature_id(self, field_name: str, value: Any, *, fit: bool) -> int | None:
        key: tuple[str, str] = (field_name, str(value))
        if key not in self._feature_to_id:
            if not fit:
                return None
            self._feature_to_id[key] = self._next_feature_id()
        return self._feature_to_id[key]

    def _categorical_tokens(self, value: Any) -> list[str]:
        text: str = str(value).strip()
        if text == "":
            return []
        # CriteoSearch product_title is stored as a whitespace-separated hashed
        # token list.  Expanding any whitespace-separated categorical value keeps
        # that field multi-hot for libFFM instead of treating the whole title as
        # one rare category.
        return [token for token in text.split() if token]

    def _iter_rows(self, X: np.ndarray | pd.DataFrame) -> Iterator[list[Any]]:
        feature_names: list[str] = self._feature_names if self._feature_names is not None else self._get_feature_names(X)
        if isinstance(X, pd.DataFrame):
            if [str(col) for col in X.columns] != feature_names:
                raise ValueError("Prediction DataFrame columns differ from the training columns.")
            for row in X.itertuples(index=False, name=None):
                yield list(row)
            return

        if X.ndim != 2:
            raise ValueError(f"X must be 2-dimensional, got shape={X.shape}.")
        if X.shape[1] != len(feature_names):
            raise ValueError(
                f"X has {X.shape[1]} columns, but the fitted FFM model expects {len(feature_names)}."
            )
        for row_idx in range(X.shape[0]):
            yield X[row_idx, :].tolist()

    def _format_value(self, value: Any) -> str:
        return f"{float(value):.12g}"

    def _is_numeric_value(self, value: Any) -> bool:
        if pd.isna(value):
            return False
        return isinstance(value, (int, float, np.integer, np.floating))

    def _write_ffm_file(
        self,
        path: Path,
        X: np.ndarray | pd.DataFrame,
        y: np.ndarray,
        *,
        fit: bool,
    ) -> None:
        y_arr: np.ndarray = np.asarray(y)
        if y_arr.shape[0] != self._num_rows(X):
            raise ValueError(f"X/y length mismatch: X has {self._num_rows(X)} rows, y has {y_arr.shape[0]}.")
        feature_names: list[str] = self._feature_names if self._feature_names is not None else self._get_feature_names(X)

        with path.open("w", encoding="utf-8") as f:
            for label, row_values in zip(y_arr, self._iter_rows(X)):
                parts: list[str] = [str(int(label))]
                for field_name, raw_value in zip(feature_names, row_values):
                    if pd.isna(raw_value):
                        continue
                    field_id: int = self._field_id(field_name)
                    if self._is_numeric_value(raw_value):
                        value: float = float(raw_value)
                        if value == 0.0:
                            continue
                        feature_id: int | None = self._numeric_feature_id(field_name)
                        parts.append(f"{field_id}:{feature_id}:{self._format_value(value)}")
                    else:
                        for token in self._categorical_tokens(raw_value):
                            feature_id = self._categorical_feature_id(field_name, token, fit=fit)
                            if feature_id is not None:
                                parts.append(f"{field_id}:{feature_id}:1")
                f.write(" ".join(parts))
                f.write("\n")


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


def _build_libffm(params: dict[str, Any], seed: int) -> LibFFMClassifier:
    print("Building libffm FFM with params:", params)
    return LibFFMClassifier(seed=seed, **params)


def _build_real_data_preprocessor(
    X_sample: pd.DataFrame,
    *,
    needs_scaling: bool,
) -> ColumnTransformer:
    numeric_cols: list[str] = X_sample.select_dtypes(include=np.number).columns.tolist()
    categorical_cols: list[str] = X_sample.select_dtypes(exclude=np.number).columns.tolist()
    transformers: list[tuple[str, Any, Any]] = []

    if len(numeric_cols) > 0:
        numeric_transformer: Any = StandardScaler() if needs_scaling else "passthrough"
        transformers.append(("num", numeric_transformer, numeric_cols))

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
) -> Any:
    """
    Build an sklearn Pipeline for eta(x)=P(Y=1|X=x) estimation.

    Depending on the model configuration, this function optionally applies
    feature standardization before the classifier and returns a unified
    Pipeline with ``fit``/``predict_proba`` interface.
    """
    est: Any = cfg.builder(params, seed)
    if cfg.direct_input:
        return est
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


def get_libffm_eta_model_configs(
    *,
    ffm_train_path: str = "ffm-train",
    ffm_predict_path: str = "ffm-predict",
    threads: int = 1,
    reduced: bool = True,
) -> list[EtaModelConfig]:
    """
    FFM configs backed by libffm's ffm-train/ffm-predict CLI.

    The grid intentionally stays compact because each candidate launches an
    external training process and Avazu/Criteo feature spaces are large.
    """
    if reduced:
        ffm_grid: list[dict[str, Any]] = [
            {
                "lambda_": lambda_,
                "k": 4,
                "iterations": 30,
                "eta": eta,
                "threads": int(threads),
                "auto_stop": True,
                "ffm_train_path": ffm_train_path,
                "ffm_predict_path": ffm_predict_path,
            }
            for lambda_ in [2e-5, 1e-4]
            for eta in [0.05, 0.2]
        ]
    else:
        ffm_grid = [
            {
                "lambda_": lambda_,
                "k": k,
                "iterations": iterations,
                "eta": eta,
                "threads": int(threads),
                "auto_stop": True,
                "ffm_train_path": ffm_train_path,
                "ffm_predict_path": ffm_predict_path,
            }
            for lambda_ in [2e-5, 1e-4, 1e-3]
            for k in [4, 8]
            for iterations in [15, 30]
            for eta in [0.05, 0.2]
        ]
    return [EtaModelConfig("ffm", ffm_grid, False, _build_libffm, direct_input=True)]


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
    score_fn: Callable[[Any], float],
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
        model: Any = make_eta_pipeline(cfg, params, seed=seed, X_sample=X_train)
        fit_with_validation: Any | None = getattr(model, "fit_with_validation", None)
        if callable(fit_with_validation):
            fit_with_validation(X_train, y_train, X_val, y_val)
        else:
            model.fit(X_train, y_train)
        s = float(score_fn(model))
        if s > best["score"]:
            previous_best_model: Any | None = best.get("model")
            cleanup_previous_best: Any | None = getattr(previous_best_model, "cleanup", None)
            if callable(cleanup_previous_best):
                cleanup_previous_best()
            best = {
                "score": s,
                "params": dict(params),
                "model": model,
                "cfg_name": cfg.name,
            }
        else:
            cleanup_model = getattr(model, "cleanup", None)
            if callable(cleanup_model):
                cleanup_model()

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
