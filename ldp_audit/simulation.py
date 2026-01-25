# ldp_audit/simulation.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
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
    per = int(np.ceil(N / spec.num_classes))
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

def sample_x_given_y_truncated(
    n: int,
    spec: MixtureSpec | None,
    y: int,
    rng: np.random.Generator,
    B: float,
    *,
    batch_size: int | None = None,
    max_rounds: int = 10_000,
) -> np.ndarray:
    """
    Truncated sampling:  X ~ N(mu_y, sigma^2 I) conditioned on ||X||_2 <= B.

    いわゆる「棄却サンプリング」で、球外に出たサンプルは捨てて
    必要数 n 個が集まるまで繰り返します。

    Parameters
    ----------
    n : int
        生成したいサンプル数
    spec : MixtureSpec
        mixture の仕様（ここではクラス条件付き Gaussian）
    y : int
        クラスラベル（0..num_classes-1）
    rng : np.random.Generator
        乱数生成器
    B : float
        L2 ボール半径（||X||_2 <= B の制約）
    batch_size : int | None
        1回にまとめて生成する候補数。None の場合は自動設定。
    max_rounds : int
        無限ループ防止。受理率が極端に小さいときに例外を投げます。

    Returns
    -------
    X_acc : (n, d) ndarray
        棄却サンプリングで得たサンプル
    """
    if spec is None:
        raise ValueError("spec must be provided")
    if n <= 0:
        raise ValueError("n must be positive")
    if B <= 0.0:
        raise ValueError("B must be positive")
    if not (0 <= y < spec.num_classes):
        raise ValueError(f"y must be in [0, {spec.num_classes - 1}]")

    comps: list[multivariate_normal] = _make_gaussian_components(spec)  # type: ignore
    comp = comps[y]

    # バッチサイズ自動設定：とりあえず n の数倍を投げる（受理率が高い前提）
    if batch_size is None:
        # n が小さいときは最低 1024、n が大きいときは 4n くらい
        batch_size = max(1024, 4 * n)

    accepted: list[np.ndarray] = []
    total_acc = 0

    for _round in range(max_rounds):
        # 候補を生成
        X_cand = np.asarray(comp.rvs(size=batch_size, random_state=rng))
        # 受理判定
        norms = np.linalg.norm(X_cand, axis=1)
        mask = norms <= B
        if np.any(mask):
            X_ok = X_cand[mask]
            accepted.append(X_ok)
            total_acc += X_ok.shape[0]
            if total_acc >= n:
                X_all = np.vstack(accepted)
                return X_all[:n]

        # まだ足りないとき：残り必要数に応じて batch を少し調整（任意）
        remaining = n - total_acc
        if remaining > 0 and remaining < batch_size // 4:
            batch_size = max(1024, 4 * remaining)

    raise RuntimeError(
        f"sample_x_given_y_truncated: failed to collect n={n} samples "
        f"within max_rounds={max_rounds}. "
        f"Acceptance rate may be too small for B={B} (d={spec.d}, sigma={spec.sigma})."
    )


def sample_x_given_y(
    n: int, spec: MixtureSpec | None, y: int, rng: np.random.Generator
) -> np.ndarray:
    if spec is None:
        raise ValueError("spec must be provided")
    comps: list[multivariate_normal] = _make_gaussian_components(spec)  # type: ignore
    return np.asarray(comps[y].rvs(size=n, random_state=rng))


def posterior_probs_from_x(X: np.ndarray, spec: MixtureSpec | None) -> np.ndarray:
    if spec is None:
        raise ValueError("spec must be provided")
    comps: list[multivariate_normal] = _make_gaussian_components(spec)  # type: ignore
    logpdf: np.ndarray = np.stack([comps[j].logpdf(X) for j in range(spec.num_classes)], axis=1)
    m: np.ndarray = logpdf.max(axis=1, keepdims=True)
    probs: np.ndarray = np.exp(logpdf - m)
    probs /= probs.sum(axis=1, keepdims=True)
    return probs

def set_B_from_quantile(spec: MixtureSpec | None = None, alpha: float = 1e-6) -> float:
    """
    クラス条件付き Gaussian の (1-alpha) 分位球をすべて包含する
    L2 ボール半径 B を設定する

    この B は「モデル仕様」として固定され，
    logRmax はこの B に対して厳密に評価される
    """
    if spec is None:
        raise ValueError("spec must be set before computing B")

    mus: list[np.ndarray] = _mus(spec)  # list[np.ndarray]
    sigma: float = spec.sigma
    d: int = spec.d

    # χ^2_d の (1-alpha) 分位
    radius = sigma * np.sqrt(chi2.ppf(1.0 - alpha, d))

    B: float = float(max(np.linalg.norm(mu) + radius for mu in mus))

    return B


def log_Rmax_gaussian(spec: MixtureSpec | None, B: float = 0.01) -> float:
    """
    L2 ボール半径 self.B に対する
    厳密な log R_max を計算してセットする。

    前提：
    - 同一共分散 Gaussian mixture
    - X は必ず L2 投影される
    """
    if spec is None:
        raise ValueError("spec must be set")
    if B <= 0.0:
        raise ValueError("B must be positive")

    assert spec.num_classes >= 2, "binary 以上で使用"
    
    mus: list[np.ndarray] = _mus(spec)
    sigma2: float = spec.sigma**2

    # 球 K_{1-alpha} = {||x|| <= B} を構成
    # 全ペア (j,k) で r_max を評価（num_classes>2 に対応）
    max_val = 0.0
    for j in range(spec.num_classes):
        for k in range(spec.num_classes):
            if j == k:
                continue
            a = (mus[j] - mus[k]) / sigma2  # shape (d,)
            b: float = -0.5 * (mus[j] @ mus[j] - mus[k] @ mus[k]) / sigma2
            val: float = abs(b) + np.linalg.norm(a) * B  # sup_{||x||<=B} |a^T x + b|
            max_val: float = max(max_val, val)

    return float(max_val)