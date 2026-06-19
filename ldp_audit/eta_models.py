# eta_models.py
from __future__ import annotations

import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

import numpy as np
import pandas as pd
import torch
from scipy import sparse
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC


def clip_eta(e: np.ndarray, eps: float = 1e-15) -> np.ndarray:
    """Numerical safety for eta in (0,1)."""
    e = np.asarray(e, dtype=float)
    return np.clip(e, eps, 1.0 - eps)


def resolve_torch_device(device: str = "auto") -> str:
    """Resolve a requested PyTorch device string without importing torch globally."""
    requested: str = str(device).strip().lower()
    if requested not in {"auto", "cpu", "cuda", "mps"} and not requested.startswith(
        "cuda:"
    ):
        raise ValueError(
            f"Unknown torch device: {device!r}. Use auto/cpu/cuda/mps/cuda:N."
        )

    try:
        import torch
    except Exception as exc:
        if requested == "auto":
            return "cpu"
        raise RuntimeError(
            f"PyTorch is required for torch_device={requested!r}, but importing torch failed."
        ) from exc

    if requested == "cpu":
        return "cpu"
    if requested == "cuda":
        if torch.cuda.is_available():
            return "cuda"
        raise RuntimeError(
            "torch_device='cuda' was requested, but CUDA is not available."
        )
    if requested.startswith("cuda:"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"torch_device={requested!r} was requested, but CUDA is not available."
            )
        try:
            cuda_index = int(requested.split(":", 1)[1])
        except ValueError as exc:
            raise ValueError(
                f"Invalid CUDA device string: {device!r}. Use cuda:N."
            ) from exc
        if cuda_index < 0 or cuda_index >= torch.cuda.device_count():
            raise RuntimeError(
                f"torch_device={requested!r} was requested, but torch sees "
                f"{torch.cuda.device_count()} CUDA device(s)."
            )
        return requested
    if requested == "mps":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        raise RuntimeError(
            "torch_device='mps' was requested, but MPS is not available."
        )

    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


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
    reuse_fitted_model: bool = False


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


class TorchMLPClassifier(ClassifierMixin, BaseEstimator):
    """Small sklearn-like binary MLP classifier backed by PyTorch."""

    def __init__(
        self,
        *,
        hidden_layer_sizes: tuple[int, ...] = (64,),
        alpha: float = 1e-4,
        learning_rate_init: float = 1e-3,
        batch_size: int = 512,
        max_epochs: int = 500,
        patience: int = 20,
        device: str = "auto",
        model_name: str = "Torch MLP",
        activation: str = "relu",
        dropout_rate: float = 0.0,
        optimizer_name: str = "adamw",
        momentum_start: float = 0.9,
        momentum_end: float = 0.99,
        momentum_ramp_epochs: int = 200,
        learning_rate_decay: float = 1.0,
        min_learning_rate: float = 1e-6,
        min_epochs: int = 0,
        relative_improvement: float = 0.0,
        input_scaling: str = "none",
        first_layer_init_std: float | None = None,
        hidden_layer_init_std: float | None = None,
        output_layer_init_std: float | None = None,
        validation_batch_size: int = 16384,
        seed: int | None = None,
    ) -> None:
        # Keep constructor parameters unchanged so sklearn.clone/get_params can
        # treat this class as a regular estimator. Convert values only where
        # they are consumed during fitting.
        self.hidden_layer_sizes: tuple[int, ...] = hidden_layer_sizes
        self.alpha: float = alpha
        self.learning_rate_init: float = learning_rate_init
        self.batch_size: int = batch_size
        self.max_epochs: int = max_epochs
        self.patience: int = patience
        self.device: str = device
        self.model_name: str = model_name
        self.activation: str = activation
        self.dropout_rate: float = dropout_rate
        self.optimizer_name: str = optimizer_name
        self.momentum_start: float = momentum_start
        self.momentum_end: float = momentum_end
        self.momentum_ramp_epochs: int = momentum_ramp_epochs
        self.learning_rate_decay: float = learning_rate_decay
        self.min_learning_rate: float = min_learning_rate
        self.min_epochs: int = min_epochs
        self.relative_improvement: float = relative_improvement
        self.input_scaling: str = input_scaling
        self.first_layer_init_std: float | None = first_layer_init_std
        self.hidden_layer_init_std: float | None = hidden_layer_init_std
        self.output_layer_init_std: float | None = output_layer_init_std
        self.validation_batch_size: int = validation_batch_size
        self.seed: int | None = seed

        self.device_: str | None = None
        self._model: Any | None = None
        self._input_dim: int | None = None
        self._is_fitted: bool = False
        self._input_center: np.ndarray | None = None
        self._input_scale: np.ndarray | None = None
        self._positive_mean_mask: np.ndarray | None = None

    def fit(self, X: np.ndarray | pd.DataFrame, y: np.ndarray) -> "TorchMLPClassifier":
        return self.fit_with_validation(X, y, None, None)

    def fit_with_validation(
        self,
        X: np.ndarray | pd.DataFrame,
        y: np.ndarray,
        X_val: np.ndarray | pd.DataFrame | None,
        y_val: np.ndarray | None,
    ) -> "TorchMLPClassifier":
        try:
            import torch
            from torch import nn
        except Exception as exc:
            raise RuntimeError(
                "PyTorch is required to use TorchMLPClassifier."
            ) from exc

        if self.seed is not None:
            torch.manual_seed(int(self.seed))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(self.seed))

        X_np_raw: np.ndarray = self._as_float32_matrix(X)
        self._fit_input_scaler(X_np_raw)
        X_np: np.ndarray = self._transform_input(X_np_raw)
        y_np: np.ndarray = np.asarray(y, dtype=np.float32).reshape(-1, 1)
        if X_np.shape[0] != y_np.shape[0]:
            raise ValueError(
                f"X/y length mismatch: X has {X_np.shape[0]} rows, y has {y_np.shape[0]}."
            )

        X_val_np: np.ndarray | None = (
            self._transform_input(self._as_float32_matrix(X_val))
            if X_val is not None
            else None
        )
        y_val_np: np.ndarray | None = (
            np.asarray(y_val, dtype=np.float32).reshape(-1, 1)
            if y_val is not None
            else None
        )
        if (
            X_val_np is not None
            and y_val_np is not None
            and X_val_np.shape[0] != y_val_np.shape[0]
        ):
            raise ValueError(
                f"X_val/y_val length mismatch: X_val has {X_val_np.shape[0]} rows, "
                f"y_val has {y_val_np.shape[0]}."
            )

        self.device_ = resolve_torch_device(self.device)
        device = torch.device(self.device_)
        self._input_dim = int(X_np.shape[1])
        model: nn.Module = self._build_model(self._input_dim, nn).to(device)
        self._model = model
        print(
            f"Building {self.model_name} with params:",
            {
                "hidden_layer_sizes": self.hidden_layer_sizes,
                "alpha": self.alpha,
                "learning_rate_init": self.learning_rate_init,
                "activation": self.activation,
                "dropout_rate": self.dropout_rate,
                "optimizer_name": self.optimizer_name,
                "input_scaling": self.input_scaling,
                "device": self.device_,
            },
        )

        train_x_t: torch.Tensor = torch.from_numpy(X_np).to(device)
        train_y_t: torch.Tensor = torch.from_numpy(y_np).to(device)
        batch_size: int = max(1, min(self.batch_size, X_np.shape[0]))
        optimizer_name: str = str(self.optimizer_name).strip().lower()
        if optimizer_name == "adamw":
            optimizer = torch.optim.AdamW(
                self._model.parameters(),
                lr=self.learning_rate_init,
                weight_decay=self.alpha,
            )
        elif optimizer_name == "sgd":
            optimizer = torch.optim.SGD(
                self._model.parameters(),
                lr=self.learning_rate_init,
                momentum=self.momentum_start,
                weight_decay=self.alpha,
            )
        else:
            raise ValueError(
                f"Unknown optimizer_name={self.optimizer_name!r}; use 'adamw' or 'sgd'."
            )

        def loss_fn(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
            signed_targets: torch.Tensor = targets.mul(2.0).sub(1.0)
            return torch.nn.functional.softplus(-signed_targets * logits).mean()

        val_x_t: torch.Tensor | None = (
            torch.from_numpy(X_val_np).to(device) if X_val_np is not None else None
        )
        val_y_t: torch.Tensor | None = (
            torch.from_numpy(y_val_np).to(device) if y_val_np is not None else None
        )
        best_state: dict[str, Any] | None = None
        best_val_loss: float = np.inf
        epochs_without_improvement: int = 0

        for epoch in range(max(1, self.max_epochs)):
            if optimizer_name == "sgd":
                ramp_epochs: int = max(1, int(self.momentum_ramp_epochs))
                ramp_fraction: float = min(float(epoch) / float(ramp_epochs), 1.0)
                momentum: float = float(
                    self.momentum_start
                    + ramp_fraction * (self.momentum_end - self.momentum_start)
                )
                for group in optimizer.param_groups:
                    group["momentum"] = momentum

            self._model.train()
            order: torch.Tensor = torch.randperm(train_x_t.shape[0], device=device)
            for start in range(0, train_x_t.shape[0], batch_size):
                batch_idx: torch.Tensor = order[start : start + batch_size]
                xb: torch.Tensor = train_x_t.index_select(0, batch_idx)
                yb: torch.Tensor = train_y_t.index_select(0, batch_idx)
                optimizer.zero_grad(set_to_none=True)
                logits: torch.Tensor = self._model(xb)
                loss: torch.Tensor = loss_fn(logits, yb)
                loss.backward()
                optimizer.step()
                if self.learning_rate_decay > 1.0:
                    for group in optimizer.param_groups:
                        group["lr"] = max(
                            float(self.min_learning_rate),
                            float(group["lr"]) / float(self.learning_rate_decay),
                        )

            if val_x_t is None or val_y_t is None:
                continue

            self._model.eval()
            with torch.no_grad():
                val_loss: float = self._mean_loss_in_batches(
                    val_x_t,
                    val_y_t,
                    loss_fn=loss_fn,
                )
            previous_best: float = best_val_loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {
                    k: v.detach().cpu().clone()
                    for k, v in self._model.state_dict().items()
                }

            improvement_threshold: float = max(
                1e-7,
                abs(previous_best) * float(self.relative_improvement)
                if np.isfinite(previous_best)
                else 1e-7,
            )
            improved_enough: bool = (
                not np.isfinite(previous_best)
                or val_loss < previous_best - improvement_threshold
            )
            if improved_enough:
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if (
                    epoch + 1 >= max(1, int(self.min_epochs))
                    and epochs_without_improvement >= self.patience
                ):
                    break

        if best_state is not None:
            self._model.load_state_dict(best_state)
            self._model.to(device)
        self._is_fitted = True
        return self

    def __sklearn_is_fitted__(self) -> bool:
        return self._is_fitted

    def predict_proba(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        if not self._is_fitted or self._model is None or self.device_ is None:
            raise RuntimeError(
                "TorchMLPClassifier must be fitted before predict_proba."
            )
        import torch

        X_np: np.ndarray = self._transform_input(self._as_float32_matrix(X))
        device: torch.device = torch.device(self.device_)
        self._model.eval()
        probs: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, X_np.shape[0], max(1, self.batch_size)):
                xb: torch.Tensor = torch.from_numpy(
                    X_np[start : start + self.batch_size]
                ).to(device)
                p1: np.ndarray = (
                    torch.sigmoid(self._model(xb)).detach().cpu().numpy().reshape(-1)
                )
                probs.append(p1)
        p1_all: np.ndarray = clip_eta(
            np.concatenate(probs) if probs else np.empty(0, dtype=np.float64)
        )
        return np.column_stack([1.0 - p1_all, p1_all])

    def cleanup(self) -> None:
        self._model = None

    def _fit_input_scaler(self, X: np.ndarray) -> None:
        scaling: str = str(self.input_scaling).strip().lower()
        if scaling == "none":
            self._input_center = None
            self._input_scale = None
            self._positive_mean_mask = None
            return
        if scaling != "paper":
            raise ValueError(
                f"Unknown input_scaling={self.input_scaling!r}; use 'none' or 'paper'."
            )

        positive_mean_mask: np.ndarray = np.all(X > 0.0, axis=0)
        center: np.ndarray = np.mean(X, axis=0, dtype=np.float64).astype(np.float32) #type: ignore
        scale: np.ndarray = np.std(X, axis=0, dtype=np.float64).astype(np.float32)
        scale = np.where(scale > 0.0, scale, 1.0).astype(np.float32)
        positive_means: np.ndarray = np.where(np.abs(center) > 0.0, center, 1.0)
        center = center.copy()
        scale = scale.copy()
        center[positive_mean_mask] = 0.0
        scale[positive_mean_mask] = positive_means[positive_mean_mask]

        self._input_center = center
        self._input_scale = scale
        self._positive_mean_mask = positive_mean_mask

    def _transform_input(self, X: np.ndarray) -> np.ndarray:
        if self._input_center is None or self._input_scale is None:
            return X
        return np.ascontiguousarray(
            (X - self._input_center) / self._input_scale,
            dtype=np.float32,
        )

    def _mean_loss_in_batches(
        self,
        X: Any,
        y: Any,
        *,
        loss_fn: Callable[[Any, Any], Any],
    ) -> float:
        if self._model is None:
            raise RuntimeError("Model is not initialized.")
        batch_size: int = max(1, int(self.validation_batch_size))
        total_loss: float = 0.0
        total_count: int = 0
        for start in range(0, int(X.shape[0]), batch_size):
            xb = X[start : start + batch_size]
            yb = y[start : start + batch_size]
            batch_count: int = int(xb.shape[0])
            total_loss += float(
                loss_fn(self._model(xb), yb).detach().cpu().item()
            ) * batch_count
            total_count += batch_count
        return total_loss / max(total_count, 1)

    @staticmethod
    def _as_float32_matrix(X: Any) -> np.ndarray:
        if X is None:
            raise ValueError("X must not be None.")
        if sparse.issparse(X):
            X = X.toarray()
        elif isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        arr: np.ndarray = np.asarray(X, dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError(f"X must be 2-dimensional, got shape={arr.shape}.")
        return np.ascontiguousarray(arr)

    def _build_model(self, input_dim: int, nn: Any) -> Any:
        layers: list[Any] = []
        prev = int(input_dim)
        activation_name: str = str(self.activation).strip().lower()
        for layer_idx, hidden_dim in enumerate(self.hidden_layer_sizes):
            linear = nn.Linear(prev, int(hidden_dim))
            self._initialize_linear_layer(
                linear,
                std=(
                    self.first_layer_init_std
                    if layer_idx == 0
                    else self.hidden_layer_init_std
                ),
                nn=nn,
            )
            layers.append(linear)
            if activation_name == "relu":
                layers.append(nn.ReLU())
            elif activation_name == "tanh":
                layers.append(nn.Tanh())
            else:
                raise ValueError(
                    f"Unknown activation={self.activation!r}; use 'relu' or 'tanh'."
                )
            if (
                self.dropout_rate > 0.0
                and layer_idx == len(self.hidden_layer_sizes) - 1
            ):
                layers.append(nn.Dropout(p=float(self.dropout_rate)))
            prev = int(hidden_dim)
        output = nn.Linear(prev, 1)
        self._initialize_linear_layer(
            output,
            std=self.output_layer_init_std,
            nn=nn,
        )
        layers.append(output)
        return nn.Sequential(*layers)

    @staticmethod
    def _initialize_linear_layer(linear: Any, *, std: float | None, nn: Any) -> None:
        if std is None:
            return
        nn.init.normal_(linear.weight, mean=0.0, std=float(std))
        nn.init.zeros_(linear.bias)


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


def _build_torch_mlp(params: dict[str, Any], seed: int) -> TorchMLPClassifier:
    return TorchMLPClassifier(seed=seed, **params)


def _build_torch_dnn(params: dict[str, Any], seed: int) -> TorchMLPClassifier:
    return TorchMLPClassifier(seed=seed, model_name="Torch DNN", **params)


def _build_torch_dnn_paper(
    params: dict[str, Any],
    seed: int,
) -> TorchMLPClassifier:
    return TorchMLPClassifier(
        seed=seed,
        model_name="Torch DNN (Baldi et al. 2014)",
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


def get_torch_mlp_eta_model_configs(
    *,
    device: str = "auto",
    reduced: bool = True,
) -> list[EtaModelConfig]:
    """
    PyTorch MLP configs.  With device="auto", CUDA/MPS is used when available
    and CPU is used otherwise.
    """
    if reduced:
        grid: list[dict[str, Any]] = [
            {
                "hidden_layer_sizes": hs,
                "alpha": a,
                "learning_rate_init": lr,
                "device": device,
                "max_epochs": 500,
                "patience": 20,
                "batch_size": 512,
            }
            for hs in [(64,), (128,)]
            for a in [1e-5, 1e-4, 1e-3]
            for lr in [1e-3, 1e-2]
        ]
    else:
        grid = [
            {
                "hidden_layer_sizes": hs,
                "alpha": a,
                "learning_rate_init": lr,
                "device": device,
                "max_epochs": 500,
                "patience": 20,
                "batch_size": 512,
            }
            for hs in [(32,), (64,), (128,)]
            for a in [1e-5, 1e-4, 1e-3]
            for lr in [1e-4, 1e-3, 1e-2]
        ]
    return [EtaModelConfig("torch_mlp", grid, True, _build_torch_mlp)]


def get_torch_dnn_eta_model_configs(
    *,
    device: str = "auto",
    reduced: bool = True,
) -> list[EtaModelConfig]:
    """
    Deeper PyTorch MLP configs intended for numeric real-data benchmarks such as
    SUSY.  The architecture candidates have at least two hidden layers, so this
    family is reported separately from the shallow torch_mlp baseline.
    """
    if reduced:
        grid: list[dict[str, Any]] = [
            {
                "hidden_layer_sizes": hs,
                "alpha": a,
                "learning_rate_init": lr,
                "device": device,
                "max_epochs": 500,
                "patience": 20,
                "batch_size": 512,
            }
            for hs in [(128, 128), (256, 128, 64)]
            for a in [1e-5, 1e-4, 1e-3]
            for lr in [1e-3, 1e-2]
        ]
    else:
        grid = [
            {
                "hidden_layer_sizes": hs,
                "alpha": a,
                "learning_rate_init": lr,
                "device": device,
                "max_epochs": 500,
                "patience": 20,
                "batch_size": 512,
            }
            for hs in [
                (128, 128),
                (256, 128, 64),
                (300, 300, 300, 300, 300),
            ]
            for a in [1e-5, 1e-4, 1e-3]
            for lr in [1e-4, 1e-3, 1e-2]
        ]
    return [EtaModelConfig("torch_dnn", grid, True, _build_torch_dnn)]


def get_torch_dnn_paper_eta_model_configs(
    *,
    device: str = "auto",
) -> list[EtaModelConfig]:
    """
    Fixed SUSY DNN preset based on Baldi, Sadowski, and Whiteson (2014).

    The paper's best SUSY model used five 300-unit tanh hidden layers and
    50% dropout on the top hidden layer. Training used minibatch SGD with
    momentum ramped from 0.9 to 0.99 over 200 epochs, per-update learning-rate
    decay, and up to roughly 1000 epochs.

    Input scaling follows the paper using training-set statistics only:
    general features are standardized, while features strictly greater than
    zero are divided by their mean. The original paper used statistics from
    the full train/test data; avoiding that leakage is the intentional
    difference here.
    """
    paper_config: dict[str, Any] = {
        "hidden_layer_sizes": (300, 300, 300, 300, 300),
        "alpha": 1e-5,
        "learning_rate_init": 0.05,
        "device": device,
        "max_epochs": 1000,
        "patience": 40,
        "batch_size": 100,
        "activation": "tanh",
        "dropout_rate": 0.5,
        "optimizer_name": "sgd",
        "momentum_start": 0.9,
        "momentum_end": 0.99,
        "momentum_ramp_epochs": 200,
        "learning_rate_decay": 1.0000003,
        "min_learning_rate": 1e-6,
        "min_epochs": 200,
        "relative_improvement": 1e-5,
        "input_scaling": "paper",
        "first_layer_init_std": 0.1,
        "hidden_layer_init_std": 0.05,
        "output_layer_init_std": 0.001,
        "validation_batch_size": 16384,
    }
    return [
        EtaModelConfig(
            "torch_dnn_paper",
            [paper_config],
            False,
            _build_torch_dnn_paper,
            direct_input=True,
            reuse_fitted_model=True,
        )
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
