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

def grr_sample_binary(
    y: int, n: int, epsilon: float, rng: np.random.Generator
) -> np.ndarray:
    """ rng を外部から受け取るため実装 """
    p, q = _grr_pq_binary(float(epsilon))
    u: np.ndarray = rng.random(n)
    # correct with prob p, flip with prob q
    return np.where(u < p, y, 1 - y).astype(np.int64)


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


def dp_sniper_threshold_from_scores(
    scores_yprime: np.ndarray, c: float
) -> tuple[float, float]:
    """
    DP-Sniper Alg.1 line 5-8 相当:
    y' (==0) 側サンプルのスコアから (t#, q#) を作る。
    """
    if scores_yprime.ndim != 1 or scores_yprime.size < 1:
        raise ValueError("scores_yprime must be 1D non-empty.")
    c = float(c)
    if not (0.0 < c < 1.0):
        raise ValueError("c must be in (0,1).")

    s: np.ndarray = np.sort(scores_yprime)[::-1]  # 降順
    N: int = s.size

    # 0-indexed で ceil(cN)-1 番目（= 上位 ceil(cN) 個に入る境界）
    k = int(np.ceil(c * N))
    idx: int = min(max(k - 1, 0), N - 1)
    t = float(s[idx])

    # tie の数を数えて q# を作る（Alg.1 line 7）
    atol = 1e-12
    gt = int(np.sum(s > t + atol))                 # “明確に上”
    eq = int(np.sum(np.abs(s - t) <= atol))        # “同点帯”
    # q = (cN - #>t) / #=t
    q: float = (c * N - gt) / eq if eq > 0 else 0.0
    q = float(np.clip(q, 0.0, 1.0))
    return t, q



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