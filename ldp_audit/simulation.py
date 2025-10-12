# ldp_audit/simulation.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from numpy._typing._array_like import NDArray
from scipy.stats import multivariate_normal , chi2


@dataclass
class MixtureSpec:
    num_classes: int = 2
    d: int = 100
    sigma: float = 1.0  # 各クラス同一の等方/対角共分散: sigma^2 I
    mean_shift: float = 2.0  # クラス j の j 次元目を +mean_shift だけずらす


def _mus(spec: MixtureSpec) -> list[np.ndarray]:
    mus: list[np.ndarray] = []
    for j in range(spec.num_classes):
        mu: np.ndarray = np.zeros(spec.d, dtype=float)
        mu[j % spec.d] = spec.mean_shift
        mus.append(mu)
    return mus


def log_Rmax_effective_gaussian(spec: MixtureSpec, alpha: float = 0.01) -> float:
    """
    連続モデルの解析式で ln R_max^(alpha) を返す（事前は不要）。
    alpha は高確率領域 1-alpha の外を切り捨てる有効上限。
    """
    assert spec.num_classes >= 2, "binary 以上で使用"
    mus = _mus(spec)
    sigma2: float = spec.sigma**2

    # 球 K_{1-alpha} = {||x|| <= R} を構成
    Ry: float = float(spec.sigma * np.sqrt(chi2.ppf(1 - alpha, spec.d)))
    R: float = float(max(np.linalg.norm(mu) + Ry for mu in mus))

    # 全ペア (j,k) で r_max を評価（num_classes>2 に対応）
    max_val = 0.0
    for j in range(spec.num_classes):
        for k in range(spec.num_classes):
            if j == k:
                continue
            a = (mus[j] - mus[k]) / sigma2  # shape (d,)
            b: float = -0.5 * (mus[j] @ mus[j] - mus[k] @ mus[k]) / sigma2
            val: float = abs(b) + np.linalg.norm(a) * R  # sup_{||x||<=R} |a^T x + b|
            max_val: float = max(max_val, val)

    return float(max_val)

def _make_gaussian_components(spec: MixtureSpec) -> list[multivariate_normal]: # type: ignore
    """クラスごとの多変量正規分布（同一共分散）を作る。"""
    if spec.num_classes < 2:
        raise ValueError("num_classes must be >= 2")
    if spec.d < spec.num_classes:
        # d>=num_classes が自然
        # ただし足りない場合は循環させて配置
        # expample: d=2, num_classes=3 -> mu_0=(s,0), mu_1=(0,s), mu_2=(s,0)
        pass

    cov = np.eye(spec.d) * (spec.sigma**2)
    comps: list[multivariate_normal] = []  # type: ignore
    for j in range(spec.num_classes):
        mu: np.ndarray = np.zeros(spec.d, dtype=float)
        mu[j % spec.d] = spec.mean_shift  # d < num_classes の場合も動くよう mod
        comps.append(multivariate_normal(mean=mu, cov=cov))  # type: ignore
    return comps


def generate_mixture_probs(
    N: int, spec: MixtureSpec, seed: int | None = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    N 個のサンプル X を生成し，各クラスの事後確率 P(Y=c|X=x) を返す。

    Returns
    -------
    X : (N, d)
    probs : (N, C)  各行はクラス事後確率の正規化ベクトル（事前一様を前提）
    """
    rng: np.random.Generator = np.random.default_rng(seed)
    comps: list[multivariate_normal] = _make_gaussian_components(spec) # type: ignore

    # 事前 P(Y=c) は一様なので，各クラスから N/C 個ずつサンプリングして混ぜる
    # （一様サンプリングでも同等だが、分かりやすさ優先）
    per: int = N // spec.num_classes
    X_parts: list[np.ndarray] = [
        comps[j].rvs(size=per, random_state=rng) for j in range(spec.num_classes)
    ]
    X: np.ndarray = np.vstack(X_parts)
    rng.shuffle(X, axis=0)

    # 事後確率：probs[n, j] ∝ N(x_n | μ_j, Σ) で正規化
    # logpdf: P(X=x_n | Y=j) の行列 (N,C)
    logpdf: np.ndarray = np.stack(
        [comps[j].logpdf(X) for j in range(spec.num_classes)], axis=1
    )  # (N,C)
    # 数値安定化のため log-sum-exp
    m: np.ndarray = logpdf.max(axis=1, keepdims=True)
    probs: np.ndarray = np.exp(logpdf - m)
    # P(Y=j | X=x_n) = P(X=x_n | Y=j)P(Y=j) / Σ_j' P(X=x_n | Y=j')P(Y=j')
    # ただし P(Y=j) は一様なので分母に影響しない
    probs /= probs.sum(axis=1, keepdims=True)

    return X, probs


def simulate_eta_split(
    N_total: int,
    spec: MixtureSpec,
    val_ratio: float = 0.5,
    seed: int | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    η(x)=P(Y=1|X=x) を val/test に分割して返す（LRT-CF 監査用）。

    Returns
    -------
    eta_val : (N_val,)
    eta_test: (N_test,)
    """
    if not (0.0 < val_ratio < 1.0):
        raise ValueError("val_ratio must be in (0,1).")
    # 生成
    _, probs = generate_mixture_probs(N_total, spec, seed=seed)
    eta_all: NDArray[np.float64] = probs[:, 1]  # クラス "1" の事後確率

    N_val = int(N_total * val_ratio)
    eta_val: NDArray[np.float64] = eta_all[:N_val].astype(float)
    eta_test: NDArray[np.float64] = eta_all[N_val:].astype(float)
    return eta_val, eta_test
