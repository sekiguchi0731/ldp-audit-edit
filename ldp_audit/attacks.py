# general imports
import numpy as np
from numba import jit, prange
import xxhash

# --- GRR(k=2) のチャネルパラメータ ---
@jit(nopython=True)
def _grr_pq_binary(epsilon: float) -> tuple[float, float]:
    exp_eps: float = np.exp(epsilon)
    p: float = exp_eps / (exp_eps + 1.0)  # P( \tilde y = y | y )
    q: float = 1.0 - p  # P( \tilde y != y | y )
    return p, q


# --- 単一点の LRT 対数スコア ---
@jit(nopython=True)
def attack_lrt_score(ytilde: int, eta: float, epsilon: float) -> float:
    """
    Λ(x,\tilde y) = log( P(\tilde y|Y=1)*eta / (P(\tilde y|Y=0)*(1-eta)) )
    ** log を取ることで，積は和に，割り算は差に変換されるため，数値安定性が向上する **
    - \tilde y: 0/1
    - eta     : P(Y=1|X=x) in (0,1)
    - epsilon : RR の ε (>0)
    """
    p, q = _grr_pq_binary(epsilon)
    if eta <= 1e-15:
        eta = 1e-15
    if eta >= 1.0 - 1e-15:
        eta = 1.0 - 1e-15

    if ytilde == 1:
        return np.log(p * eta) - np.log(q * (1.0 - eta))
    else:
        return np.log(q * eta) - np.log(p * (1.0 - eta))


# --- ベクトル版スコア ---
@jit(nopython=True, parallel=True)
def attack_lrt_scores(
    ytilde_arr: np.ndarray, eta_arr: np.ndarray, epsilon: float
) -> np.ndarray:
    n: int = ytilde_arr.shape[0]
    out: np.ndarray = np.empty(n, dtype=np.float64)
    for i in prange(n):
        out[i] = attack_lrt_score(int(ytilde_arr[i]), float(eta_arr[i]), float(epsilon))
    return out


# --- しきい値 τ を与えて 0/1 予測（score >= tau -> 1） ---
@jit(nopython=True, parallel=True)
def attack_lrt_predict(
    ytilde_arr: np.ndarray, eta_arr: np.ndarray, epsilon: float, tau: float
) -> np.ndarray:
    n: int = ytilde_arr.shape[0]
    yhat: np.ndarray = np.empty(n, dtype=np.int64)
    for i in prange(n):
        s: float = attack_lrt_score(int(ytilde_arr[i]), float(eta_arr[i]), float(epsilon))
        yhat[i] = 1 if s >= tau else 0
    return yhat


def choose_tau_max_tpr_over_fpr(
    scores: np.ndarray,
    y_true: np.ndarray,
    alpha: float = 1e-2,
) -> tuple[float, float, float]:
    """
    validation の (scores, y_true) から TPR/FPR を最大化する τ* を見つける。
    返り値は (tau_star, tpr_at_tau_star, fpr_at_tau_star)。

    実装メモ:
    - スコアを降順に並べ、各しきい値での (TPR, FPR) を1回の走査で計算。
    - 同一スコア値で重複計算しないよう unique 位置だけ評価。
    - 数値不安定を避けるため FPR は clip で下駄をはかせる（ゼロ割回避）。
    """
    if scores.ndim != 1 or y_true.ndim != 1 or scores.size != y_true.size:
        raise ValueError("scores と y_true は同長の1次元配列である必要があります。")
    y_true = y_true.astype(np.int64, copy=False)
    if not (np.any(y_true == 0) and np.any(y_true == 1)):
        raise ValueError("y_true は 0/1 の両方を含む必要があります。")

    # 降順ソート
    order: np.ndarray = np.argsort(scores)[::-1]
    s: np.ndarray = scores[order]
    y: np.ndarray = y_true[order]

    # 累積で TP/FP を積む（score >= tau を陽性とする規則）
    P = float(np.sum(y == 1))
    N = float(np.sum(y == 0))
    # cum_pos[i]: スコア上位 i 個のうち何個が Y=1 か
    # cum_neg[i]: スコア上位 i 個のうち何個が Y=0 か
    cum_pos: np.ndarray = np.cumsum(y == 1).astype(np.float64)
    cum_neg: np.ndarray = np.cumsum(y == 0).astype(np.float64)
    # tpr_all[i] = TPR at threshold s[i]
    # fpr_all[i] = FPR at threshold s[i]
    tpr_all = cum_pos / P
    fpr_all = cum_neg / N

    # 同じスコア値が連続している場合、それらの間では判定結果が変わらないため、スコア値が変化する地点だけ評価する
    unique_mask: np.ndarray = np.r_[True, s[1:] < s[:-1]]

    tpr_u: np.ndarray = tpr_all[unique_mask]
    fpr_u: np.ndarray = fpr_all[unique_mask]
    tau_u: np.ndarray = s[unique_mask]

    
    mask: np.ndarray = fpr_u >= alpha
    if not np.any(mask):
        # FPR >= alpha を満たす tau_u が一つもない場合, Youden's J で近似
        J = tpr_all - fpr_all
        idx = int(np.argmax(J))
        return float(s[idx]), float(tpr_all[idx]), float(fpr_all[idx])

    ratios_masked: np.ndarray = (tpr_u[mask]) / (fpr_u[mask])

    # 1次キー: TPR/FPR 比を最大化
    best_ratio: float = float(np.max(ratios_masked))
    tol = 1e-12
    cand_local: np.ndarray = np.where(np.isclose(ratios_masked, best_ratio, rtol=0, atol=tol))[0]

    # 局所→全体 添字に戻す
    idxs = np.nonzero(mask)[0]  # 全体添字の候補
    cand_global: np.ndarray = idxs[cand_local]  # 全体添字へマッピング

    # 2次キー: FPR を最小化
    fpr_cand: np.ndarray = fpr_u[cand_global]
    fpr_min: float = np.min(fpr_cand)
    cand_global = cand_global[
        np.where(np.isclose(fpr_cand, fpr_min, rtol=0, atol=tol))[0]
    ]

    # 3次キー: TPR を最大化
    tpr_cand: np.ndarray = tpr_u[cand_global]
    idx = int(cand_global[np.argmax(tpr_cand)])
    return float(tau_u[idx]), float(tpr_u[idx]), float(fpr_u[idx])



def compute_tpr_fpr(
    scores: np.ndarray, y_true: np.ndarray, tau: float
) -> tuple[float, float]:
    pos: np.ndarray = y_true == 1
    neg: np.ndarray = ~pos
    tpr: float = float(np.mean(scores[pos] >= tau)) if np.any(pos) else 0.0
    fpr: float = float(np.mean(scores[neg] >= tau)) if np.any(neg) else 0.0
    return tpr, fpr

@jit(nopython=True)
def attack_ss(ss):
    """
    Privacy attack to Subset Selection (SS) protocol.

    Parameters:
    ----------
    ss : array
        Obfuscated subset of values.

    Returns:
    -------
    int
        A random inference of the true value.
    """
                
    return np.random.choice(ss)

@jit(nopython=True)
def attack_ue(ue_val, k):
    """
    Privacy attack to Unary Encoding (UE) protocols.

    Parameters:
    ----------
    ue_val : array
        Obfuscated vector.
    k : int
        Domain size.

    Returns:
    -------
    int
        A random inference of the true value.
    """

    if np.sum(ue_val) == 0:
        return np.random.randint(k)
    else:
        return np.random.choice(np.where(ue_val == 1)[0])

@jit(nopython=True)
def attack_the(ue_val, k, thresh):
    """
    Privacy attack to Thresholding with Histogram Encoding (THE) protocol.

    Parameters:
    ----------
    ue_val : array
        Obfuscated vector.
    k : int
        Domain size.
    thresh : float
        Optimal threshold value.

    Returns:
    -------
    int
        A random inference of the true value.
    """

    ss_the = np.where(ue_val > thresh)[0]
    if len(ss_the) == 0:
        return np.random.randint(k)
    else:
        return np.random.choice(ss_the)

@jit(nopython=True)
def attack_she(y, k, epsilon):
    """
    Privacy attack to Summation with Histogram Encoding (THE) protocol.

    Parameters:
    ----------
    y : array
        Obfuscated vector.
    k : int
        Domain size.
    epsilon : float
        Theoretical privacy guarantees.

    Returns:
    -------
    int
        A random inference of the true value.
    """

    # v_likelihood = np.zeros(k)
    # for v in prange(k):
    #     x_v = np.zeros(k)
    #     x_v[v] = 1
    #     v_likelihood[v] = np.prod(np.exp(-np.abs(y - x_v) / (2/epsilon)))
    # posterior_v = v_likelihood / np.sum(v_likelihood)
    # m = max(posterior_v) 
    # return np.random.choice(np.where(posterior_v == m)[0])

    # Equivalent and much faster (see [1])
    # [1] Arcolezi, Héber H., and Sébastien Gambs. "Revisiting Locally Differentially Private Protocols: Towards Better Trade-offs in Privacy, Utility, and Attack Resistance." arXiv preprint arXiv:2503.01482 (2025).
    return np.argmax(y)

def attack_lh(val_seed, k, g):
    """
    Privacy attack to Local Hashing (LH) protocols.

    Parameters:
    ----------
    val_seed : tuple
        Obfuscated tuple (obfuscated value, seed as "hash function").
    k : int
        Domain size.
    g : int
        Hash domain size.

    Returns:
    -------
    int
        A random inference of the true value.
    """

    lh_val = val_seed[0]
    rnd_seed = val_seed[1]

    ss_lh = []
    for v in range(k):
        if lh_val == (xxhash.xxh32(str(v), seed=rnd_seed).intdigest() % g):
            ss_lh.append(v)

    if len(ss_lh) == 0:
        return np.random.randint(k)
    else:
        return np.random.choice(ss_lh)

@jit(nopython=True)
def attack_gm(y, k, sigma):
    """
    Privacy attack to Gaussian Mechanisms (GM).

    Parameters:
    ----------
    y : array
        Obfuscated vector.
    k : int
        Domain size.
    sigma : float
        Sigma used for drawing noise.

    Returns:
    -------
    int
        A random inference of the true value.
    """

    # Prior probability for each v is uniform
    prior_v = 1 / k

    # Compute the likelihood P_Y(y|v) for each possible v
    v_likelihood = np.zeros(k)
    for v in prange(k):
        v_encoded = np.zeros(k)
        v_encoded[v] = 1.0  # One-hot encoding

        # Compute the L2 squared distance
        l2_squared = np.sum((y - v_encoded) ** 2)

        # Compute the likelihood using the Gaussian probability density function
        v_likelihood[v] = np.exp(-l2_squared / (2 * sigma**2))

    # Normalize the likelihood by multiplying by the prior and summing across all v
    posterior_v = v_likelihood * prior_v
    posterior_v /= np.sum(posterior_v)  # Normalization

    # Select the v with the highest posterior probability (randomized "argmax")
    m = np.max(posterior_v)
    return np.random.choice(np.where(posterior_v == m)[0])