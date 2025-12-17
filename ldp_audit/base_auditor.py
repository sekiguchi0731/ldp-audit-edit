# General imports
import warnings
from collections import defaultdict
from typing import Any, Self

import numpy as np
import psutil

# Imports for parallelization
import ray
import xxhash

# Import LDP protocols (by default from multi-freq-ldpy package -- https://github.com/hharcolezi/multi-freq-ldpy)
from multi_freq_ldpy.pure_frequency_oracles.GRR import GRR_Client
from multi_freq_ldpy.pure_frequency_oracles.HE import HE_Client
from multi_freq_ldpy.pure_frequency_oracles.LH import LH_Client
from multi_freq_ldpy.pure_frequency_oracles.SS import SS_Client
from multi_freq_ldpy.pure_frequency_oracles.UE import UE_Client
from numba.core.errors import NumbaExperimentalFeatureWarning

# Import UE protocols from pure-ldp package (https://github.com/Samuel-Maddock/pure-LDP)
from pure_ldp.frequency_oracles.unary_encoding import UEClient
from scipy.optimize import OptimizeResult, minimize_scalar
from scipy.special import erfinv
from statsmodels.stats.proportion import proportion_confint

from .approximate_ldp import AGRR_Client, ALH_Client, ASUE_Client, GM_Client, find_scale
from .attacks import (
    attack_gm,
    attack_lh,
    attack_lrt_scores,
    dp_sniper_threshold_from_scores,
    attack_she,
    attack_ss,
    attack_the,
    attack_ue,
    grr_sample_binary,
)
from .simulation import (
    MixtureSpec,
    log_Rmax_gaussian,
    posterior_probs_from_x,
    sample_x_given_y,
    set_B_from_quantile,
)

# Our imports
from .utils import find_tresh, setting_seed

warnings.simplefilter("ignore")
class LDPAuditor:
    """
    The LDPAuditor class is designed to audit various Local Differential Privacy (LDP) protocols.

    Methods:
    -------
    run_audit(protocol_name: str) -> float
        Runs the audit for the specified LDP protocol and returns the estimated empirical epsilon value eps_emp.
    """

    def __init__(
        self,
        nb_trials: int = int(1e6),
        alpha: float = 1e-2,
        epsilon: float = 0.25,
        delta: float = 0.0,
        k: int = 2,
        random_state: int = 42,
        n_jobs: int = -1,
        rmax_alpha: float = 1e-6,
        spec: MixtureSpec | None = None,
        c: float = 1e-6,
    ) -> None:
        """
        Initializes the LDPAuditor with the specified parameters.

        Parameters:
        ----------
        nb_trials : int
            The number of trials for the audit (default is 1e6).
        alpha : float
            The significance level for the Clopper-Pearson confidence intervals (default is 1e-2).
        epsilon : float
            The theoretical privacy budget (default is 0.25).
        delta : float
            The privacy parameter for approximate LDP protocols (default is 0.0 -- pure LDP).
        k : int
            The domain size (default is 2).
        random_state : int
            The random seed for reproducibility (default is 42).
        n_jobs : int
            The number of CPU cores to use for parallel processing (-1 uses all available cores, default is None).
        """

        # Validations
        if not isinstance(nb_trials, int) or nb_trials < 0:
            raise ValueError("nb_trials must be a non-negative integer")
        if not isinstance(alpha, float) or alpha < 0:
            raise ValueError("alpha must be a non-negative float")
        if not isinstance(epsilon, (int, float)) or epsilon < 0:
            raise ValueError("epsilon must be a non-negative float")
        if not isinstance(delta, (int, float)) or delta < 0 or delta > 1:
            raise ValueError("delta must be a float between 0 and 1 (inclusive)")
        if not isinstance(k, int) or k < 2:
            raise ValueError("k must be an integer greater than or equal to 2")
        if not isinstance(random_state, int) or random_state < 0:
            raise ValueError("random_state must be a non-negative integer")
        if not isinstance(n_jobs, int) or n_jobs < -1 or n_jobs == 0:
            raise ValueError("n_jobs must be a positive integer > 0 or -1")
        if not isinstance(rmax_alpha, float) or not (0.0 < rmax_alpha < 1.0):
            raise ValueError("rmax_alpha must be in (0,1)")
        if not isinstance(c, (int, float)) or c <= 0:
            raise ValueError("c must be a positive float for clipping.")

        # for reproducibility
        self.random_state: int = random_state

        # audit parameters
        self.nb_trials: int = nb_trials
        self.alpha: float = alpha
        self.k: int = k  # domain size
        self.v1: int = 0
        self.v2: int = k - 1
        self.eps_emp: float = 0 # estimated empirical epsilon
        self.eps_ci: tuple[float, float] = (0, 0) 

        # theoretical LDP guarantees
        self.delta: float | int = delta
        self.epsilon: float | int = epsilon

        self.rmax_alpha: float = rmax_alpha
        self.spec: MixtureSpec | None = spec
        self.c: float = float(c)

        self.B: float = set_B_from_quantile(self.spec, alpha=self.rmax_alpha)
        print("Set B:", self.B)
        self.logRmax_eff: float = log_Rmax_gaussian(self.spec, self.B)
        print("Set logRmax_eff:", self.logRmax_eff, "B: ", self.B)

        # for ray parallelism
        cpu_count: int | None = psutil.cpu_count()
        available_cores: int = int(cpu_count) if cpu_count is not None else 1
        if n_jobs == -1:
            self.nb_cores: int = available_cores
        else:
            self.nb_cores = min(available_cores, n_jobs)

        self.dynamic_nb_trials: int = 0
        self._apply_dynamic_nb_trials()

        self.lst_trial_per_core: list[int] = []
        self._refresh_trial_splits()

        # possible protocols to audit
        self.protocols: dict[str, Any] = {
            # pure LDP protocols
            "GRR": self.audit_grr,
            "SS": self.audit_ss,
            "SUE": self.audit_sue,
            "OUE": self.audit_oue,
            "THE": self.audit_the,
            "SHE": self.audit_she,
            "BLH": self.audit_blh,
            "OLH": self.audit_olh,
            # approximate LDP protocols
            "AGRR": self.audit_agrr,
            "ASUE": self.audit_asue,
            "AGM": self.audit_agm,
            "GM": self.audit_gm,
            "ABLH": self.audit_ablh,
            "AOLH": self.audit_aolh,
            # UE protocols from pure-ldp package (version 1.1.2)
            "SUE_pure_ldp_pck": self.audit_sue_pure_ldp_pck,
            "OUE_pure_ldp_pck": self.audit_oue_pure_ldp_pck,
            "LRT": "SPECIAL",  # 特殊扱い（run_audit 内で分岐して直に実行）
        }

    def set_params(self, **params) -> Self:
        """
        Code modified from scikit-learn package (Copyright (c) 2007-2024 The scikit-learn developers): 
        https://github.com/scikit-learn/scikit-learn/blob/5491dc695/sklearn/base.py#L251
        
        Parameters:
        ----------
        **params : dict
            Auditor parameters.

        Returns:
        -------
        self : auditor instance
            Auditor instance.
        """
        if not params:
            # Simple optimization to gain speed (inspect is slow)
            return self
        valid_params: dict[str, Any] = self.get_params()

        nested_params: defaultdict[str, dict[str, Any]] = defaultdict(dict)  # grouped by prefix
        for key, value in params.items():
            key, delim, sub_key = key.partition("__")
            if key not in valid_params:
                raise ValueError(
                    f"Invalid parameter {key!r} for auditor {self}. "
                    f"Valid parameters are: {list(valid_params.keys())!r}."
                )

            if delim:
                nested_params[key][sub_key] = value
            else:
                setattr(self, key, value)
                valid_params[key] = value

        for key, sub_params in nested_params.items():
            valid_params[key].set_params(**sub_params)

        self._apply_dynamic_nb_trials()
        self._refresh_trial_splits()

        return self

    def get_params(self) -> dict[str, Any]:
        """
        Get the parameters of the LDPAuditor.

        Returns:
        -------
        params : dict
            Parameter names mapped to their values.
        """
        return {
            'nb_trials': self.nb_trials,
            'alpha': self.alpha,
            'epsilon': self.epsilon,
            'delta': self.delta,
            'k': self.k,
            'random_state': self.random_state,
            'n_jobs': self.nb_cores,
            'spec': self.spec,
            'rmax_alpha': self.rmax_alpha,
            'logRmax_eff': self.logRmax_eff,
            'c': self.c,
            'dynamic_nb_trials': self.dynamic_nb_trials,
        }

    def _apply_dynamic_nb_trials(self) -> None:
        """
        Decide nb_trials based on clipping floor c following Guideline 1 of DP-Sniper.

        Guideline 1 (Empirical Precision) roughly says:
            For clipping c in (0,1], accuracy parameter omega > 0, and failure prob beta in (0,1),
            it suffices to choose

                N >= max{ 2(1-c)/(omega^2 c),  8(1-c)/c * (erf^{-1}(1-2beta))^2 }

            to ensure that the c-power of the learned attack is within omega of the optimal one
            with probability at least 1 - beta.

        Here:
            - self.c      : clipping floor for probabilities
            - self.alpha  : already used as significance level; we reuse it as default for beta
            - self.omega  : (optional) desired accuracy (default 0.005 if not set)
            - self.beta   : (optional) failure probability (default = self.alpha if not set)

        We then set:
            nb_trials = max(current nb_trials, N_suggested)
        with an upper cap to avoid exploding runtime.
        """
        if not self.dynamic_nb_trials:
            return

        # --- parameters ---
        # clipping floor
        c_val: float = float(self.c)
        # c must be in (0,1); if user passes weird value, fall back slightly inside the interval
        if c_val <= 0.0:
            c_val = 1e-6
        if c_val >= 1.0:
            c_val = 1.0 - 1e-6

        # desired accuracy omega (how close to optimal power)
        omega: float = float(getattr(self, "omega", 0.005))
        if omega <= 0.0:
            omega = 0.005

        # failure probability beta (probability that we *miss* the optimal attack)
        beta: float = float(getattr(self, "beta", self.alpha))
        # keep beta in a sane open interval (0, 0.5)
        beta = min(max(beta, 1e-6), 0.49)

        # --- Guideline 1 terms ---
        # N1 = 2(1-c)/(omega^2 c)
        term1: float = 2.0 * (1.0 - c_val) / (omega**2 * c_val)

        # N2 = 8(1-c)/c * (erf^{-1}(1-2beta))^2
        erf_arg: float = 1.0 - 2.0 * beta
        # numerical safety: clip into (-1,1)
        erf_arg = float(np.clip(erf_arg, -0.999999, 0.999999))
        inv_val: float = float(erfinv(erf_arg))
        term2: float = 8.0 * (1.0 - c_val) / c_val * (inv_val**2)

        suggested: int = int(np.ceil(max(term1, term2)))

        # avoid runaway trial counts
        max_cap: int = int(1e7)
        suggested = int(min(max(suggested, 1), max_cap))

        # only ever increase nb_trials automatically; userの手動設定がもっと大きければそれを優先
        self.nb_trials = max(int(self.nb_trials), suggested, 1)

    def _refresh_trial_splits(self) -> None:
        """Recompute per-core trial allocation when nb_trials or nb_cores change."""
        if self.nb_cores < 1:
            self.nb_cores = 1
        if self.nb_trials < 1:
            self.nb_trials = 1
        self.lst_trial_per_core = [
            len(list(_)) for _ in np.array_split(range(self.nb_trials), self.nb_cores)
        ]


    @ray.remote
    def audit_grr(
        self, random_state, trials, v, k, epsilon, delta, test_statistic
    ) -> int:
        """
        Audits the Generalized Randomized Response (GRR) protocol.

        Parameters:
        ----------
        random_state : int
            Random seed for reproducibility.
        trials : int
            Number of trials to run.
        v : int
            True value.
        k : int
            Domain size.
        epsilon : float
            Theoretical privacy budget.
        delta : float
            Privacy parameter for approximate LDP (not used for pure LDP).
        test_statistic : int
            Statistic to test against.

        Returns:
        -------
        int
            Count of successful inferences based on the test statistic.
        """
        
        # Suppress specific Numba warnings
        warnings.simplefilter('ignore', category=NumbaExperimentalFeatureWarning)

        setting_seed(random_state)
        np.random.seed(random_state)

        count = 0
        for _ in range(trials):            
            count += GRR_Client(v, k, epsilon) == test_statistic
        return count
    
    ############# LRT 監査用の関数群 #############

    ##### L2 ボール投影（eta を壊さず logRmax を厳密に計算するため） #####
    def project_l2_ball(self, X: np.ndarray, B: float) -> np.ndarray:
        """
        各行 x に対して L2 ボール半径 B への射影を行う
        X: shape (n, d)
        """
        norms: np.ndarray = np.linalg.norm(X, axis=1, keepdims=True)
        scale: np.ndarray = np.minimum(1.0, B / np.maximum(norms, 1e-12))
        return X * scale

    def _sample_p_theta(self, y_input: int, n: int, rng: np.random.Generator) -> np.ndarray:
        """
        入力 y_input を固定して，出力 (x, tilde y) を n 個サンプルし，
        確率 p_theta(y=1|(x, tilde y)) を計算して渡す
        Parameters:
        ----------
        y_input : int
            固定する入力ラベル
        n : int
            サンプル数
        rng : np.random.Generator
            乱数生成器
        Returns:
        -------
        p : np.ndarray
            確率 p_theta(y=1|(x, tilde y)) の配列
        """
        # y_input を固定して M(y_input) を N_final 回サンプル
        if self.spec is None:
            raise ValueError("spec must be set.")
        X: np.ndarray = sample_x_given_y(n=n, spec=self.spec, y=y_input, rng=rng)
        # 追加：L2 ボールへ投影
        X = self.project_l2_ball(X, B=self.B)
        # eta(x)=P(Y=1|X=x)
        eta: np.ndarray = posterior_probs_from_x(X, self.spec)[:, 1]
        ytilde: np.ndarray = grr_sample_binary(y=y_input, n=n, epsilon=float(self.epsilon), rng=rng)
        # llr: log-likelihood ratio
        llr: np.ndarray = attack_lrt_scores(ytilde, eta, float(self.epsilon))
        p: np.ndarray = 1.0 / (1.0 + np.exp(-llr))
        return p
    
    def evaluate_eps_with_dp_sniper_cp(self, tau: float, q: float, rng: np.random.Generator) -> dict:
        """
        固定した (tau,q) の攻撃 S^{tau,q} を N_final=self.nb_trials 回評価し，
        eps_emp と Clopper-Pearson による eps の上下界を返す

        注意：TPR だけでなく FPR も必要なので、y=1 側・y=0 側の両方を評価する
        """
        if self.spec is None:
            raise ValueError("spec must be set for LRT-CF evaluation.")
        
        N_final = int(self.nb_trials)

        def _sample_attack_outputs(p: np.ndarray) -> int:
            # S^{tau,q}(.) を実際にサンプルして「1 を出した回数」を返す
            atol = 1e-12
            gt: np.ndarray = p > tau + atol
            eq: np.ndarray = np.abs(p - tau) <= atol
            out: np.ndarray = np.zeros(p.shape[0], dtype=np.int64)
            out[gt] = 1
            if np.any(eq):
                out[eq] = (rng.random(np.sum(eq)) < q).astype(np.int64)
            return int(out.sum())

        # --- TPR: y=1 を入れた時に攻撃が 1 を出す確率 ---
        p1: np.ndarray = self._sample_p_theta(1, N_final, rng)
        TP: int = _sample_attack_outputs(p1)

        # --- FPR: y=0 を入れた時に攻撃が 1 を出す確率 ---
        p0: np.ndarray = self._sample_p_theta(0, N_final, rng)
        FP: int = _sample_attack_outputs(p0)

        tpr_hat: float = TP / N_final
        fpr_hat: float = FP / N_final

        # Clopper-Pearson（両側 alpha, 95% CI）
        tpr_lo, tpr_hi = proportion_confint(TP, N_final, alpha=self.alpha, method="beta")
        fpr_lo, fpr_hi = proportion_confint(FP, N_final, alpha=self.alpha, method="beta")

        eps_floor = 1e-12
        eps_emp = float(np.log(max(tpr_hat, eps_floor) / max(fpr_hat, eps_floor)))
        eps_lower = float(np.log(max(tpr_lo, eps_floor) / max(fpr_hi, eps_floor)))
        eps_upper = float(np.log(max(tpr_hi, eps_floor) / max(fpr_lo, eps_floor)))

        return {
            "TP": TP,
            "FP": FP,
            "N_final": N_final,
            "tpr_hat": float(tpr_hat),
            "fpr_hat": float(fpr_hat),
            "tpr_ci": (float(tpr_lo), float(tpr_hi)),  # type: ignore
            "fpr_ci": (float(fpr_lo), float(fpr_hi)),  # type: ignore
            "eps_emp": eps_emp,
            "eps_ci": (eps_lower, eps_upper),
        }

    
    def _run_lrt_cf_once(self) -> float:
        """
        LRT 監査を一度実行する. ray は並列化のためにあり, LRT では N 個のデータを直接使うため不要
        """
        rng: np.random.Generator = np.random.default_rng(self.random_state)

        # val 用 rng / test 用 rng を分離
        rng_val = np.random.default_rng(rng.integers(1 << 32))
        rng_test = np.random.default_rng(rng.integers(1 << 32))

        # ------------- 検証（検証用データで最適な tau を決める） --------------
        # N_sniper 回ぶんの (x_i, ytilde_i) を作る
        N_sniper: int = self.nb_trials
        scores_val: np.ndarray = self._sample_p_theta(0, N_sniper, rng_val)
        tau, q = dp_sniper_threshold_from_scores(scores_val, c=self.c)

        # ---  Effective log R_max を追加 ---
        if self.logRmax_eff and np.isfinite(self.logRmax_eff) and self.logRmax_eff > 0:
            # すでに外部から与えられている場合はそれを使う
            print("\nUsing externally provided logRmax_eff:", self.logRmax_eff)
            pass
        elif self.spec is not None:
            # 解析式で ln R_max^(alpha) を計算（理論に沿う）
            # ガウスならこっち
            print("Computing logRmax_eff from spec:", self.spec)
            self.logRmax_eff = float(log_Rmax_gaussian(self.spec, B=self.B))
        else:
            # 分布がわからない時はこっち
            # なんだろうこれ？？TODO
            # e: np.ndarray = np.clip(eta_val.astype(float), 1e-6, 1 - 1e-6)
            # # 各 i において大きい方を格納
            # rmax_x: np.ndarray = np.maximum(e / (1 - e), (1 - e) / e)
            # logR: np.ndarray = np.abs(np.log(rmax_x))
            # self.logRmax_eff = float(np.quantile(logR, 0.99))
            raise ValueError("spec must be set to compute logRmax_eff.")
        
        # -------------- 評価（CP 付き eps_emp） --------------
        # c-clipping は compute_tpr_fpr 内で実装
        results: dict = self.evaluate_eps_with_dp_sniper_cp(tau=tau, q=q, rng=rng_test)
        self.eps_emp = results['eps_emp']
        eps_lo, eps_hi = results['eps_ci']
        self.eps_ci = (float(eps_lo), float(eps_hi))

        # ------------- DEBUG 出力 -------------
        print("=== LRT DEBUG ===")
        print(f"eps={self.epsilon}, seed={self.random_state}, c={self.c}")
        print(f"tau={tau:.6g}, q={q:.6g}")
        print(f"tpr={results['tpr_hat']:.6g}, fpr={results['fpr_hat']:.6g}")
        print(f"eps_emp(computed)={np.log(max(results['tpr_hat'],1e-12)/max(results['fpr_hat'],1e-12)):.6g}")
        print(f"eps_lo={eps_lo:.6g}, eps_hi={eps_hi:.6g}")
        print("=== /LRT DEBUG ===")
        return self.eps_emp


    @ray.remote
    def audit_ss(
        self, random_state, trials, v, k, epsilon, delta, test_statistic
    ) -> int:
        """
        Audits the Subset Selection (SS) protocol.

        Parameters:
        ----------
        random_state : int
            Random seed for reproducibility.
        trials : int
            Number of trials to run.
        v : int
            True value.
        k : int
            Domain size.
        epsilon : float
            Theoretical privacy budget.
        delta : float
            Privacy parameter for approximate LDP (not used for pure LDP).
        test_statistic : int
            Statistic to test against.

        Returns:
        -------
        int
            Count of successful inferences based on the test statistic.
        """

        # Suppress specific Numba warnings
        warnings.simplefilter('ignore', category=NumbaExperimentalFeatureWarning)

        setting_seed(random_state)
        np.random.seed(random_state)

        count = 0
        for _ in range(trials):
            count +=  attack_ss(SS_Client(v, k, epsilon)) == test_statistic
        return count

    @ray.remote
    def audit_sue(
        self, random_state, trials, v, k, epsilon, delta, test_statistic
    ) -> int:
        """
        Audits the Symmetric Unary Encoding (SUE -- a.k.a. RAPPOR) protocol.

        Parameters:
        ----------
        random_state : int
            Random seed for reproducibility.
        trials : int
            Number of trials to run.
        v : int
            True value.
        k : int
            Domain size.
        epsilon : float
            Theoretical privacy budget.
        delta : float
            Privacy parameter for approximate LDP (not used for pure LDP).
        test_statistic : int
            Statistic to test against.

        Returns:
        -------
        int
            Count of successful inferences based on the test statistic.
        """

        # Suppress specific Numba warnings
        warnings.simplefilter('ignore', category=NumbaExperimentalFeatureWarning)
        
        setting_seed(random_state)
        np.random.seed(random_state)

        count = 0
        for _ in range(trials):
            count += attack_ue(UE_Client(v, k, epsilon, False), k) == test_statistic
        return count
    
    @ray.remote
    def audit_oue(
        self, random_state, trials, v, k, epsilon, delta, test_statistic
    ) -> int:
        """
        Audits the Optimized Unary Encoding (OUE) protocol.

        Parameters:
        ----------
        random_state : int
            Random seed for reproducibility.
        trials : int
            Number of trials to run.
        v : int
            True value.
        k : int
            Domain size.
        epsilon : float
            Theoretical privacy budget.
        delta : float
            Privacy parameter for approximate LDP (not used for pure LDP).
        test_statistic : int
            Statistic to test against.

        Returns:
        -------
        int
            Count of successful inferences based on the test statistic.
        """

        # Suppress specific Numba warnings
        warnings.simplefilter('ignore', category=NumbaExperimentalFeatureWarning)
        
        setting_seed(random_state)
        np.random.seed(random_state)

        count = 0
        for _ in range(trials):
            count += attack_ue(UE_Client(v, k, epsilon, True), k) == test_statistic
        return count

    @ray.remote
    def audit_the(
        self, random_state, trials, v, k, epsilon, delta, test_statistic
    ) -> int:
        """
        Audits the Thresholding with Histogram Encoding (THE) protocol.

        Parameters:
        ----------
        random_state : int
            Random seed for reproducibility.
        trials : int
            Number of trials to run.
        v : int
            True value.
        k : int
            Domain size.
        epsilon : float
            Theoretical privacy budget.
        delta : float
            Privacy parameter for approximate LDP (not used for pure LDP).
        test_statistic : int
            Statistic to test against.

        Returns:
        -------
        int
            Count of successful inferences based on the test statistic.
        """

        # Suppress specific Numba warnings
        warnings.simplefilter('ignore', category=NumbaExperimentalFeatureWarning)
        
        setting_seed(random_state)
        np.random.seed(random_state)

        res: OptimizeResult = minimize_scalar(
            find_tresh,
            bounds=[0.5, 1],
            method="bounded",
            args=(epsilon,),
        ) # type: ignore
        thresh = float(res.x)
        count = 0
        for _ in range(trials):
            count += attack_the(HE_Client(v, k, epsilon), k, thresh) == test_statistic
        return count
    
    @ray.remote
    def audit_she(
        self, random_state, trials, v, k, epsilon, delta, test_statistic
    ) -> int:
        """
        Audits the Summation with Histogram Encoding (SHE) protocol.

        Parameters:
        ----------
        random_state : int
            Random seed for reproducibility.
        trials : int
            Number of trials to run.
        x : int
            True value.
        k : int
            Domain size.
        epsilon : float
            Theoretical privacy budget.
        delta : float
            Privacy parameter for approximate LDP (not used for pure LDP).
        test_statistic : int
            Statistic to test against.

        Returns:
        -------
        int
            Count of successful inferences based on the test statistic.
        """

        # Suppress specific Numba warnings
        warnings.simplefilter('ignore', category=NumbaExperimentalFeatureWarning)
        
        setting_seed(random_state)
        np.random.seed(random_state)

        count = 0
        for _ in range(trials):            
            count += attack_she(HE_Client(v, k, epsilon), k, epsilon) == test_statistic
        return count
    
    @ray.remote
    def audit_blh(
        self, random_state, trials, v, k, epsilon, delta, test_statistic
    ) -> int:
        """
        Audits the Binary Local Hashing (BLH) protocol.

        Parameters:
        ----------
        random_state : int
            Random seed for reproducibility.
        trials : int
            Number of trials to run.
        x : int
            True value.
        k : int
            Domain size.
        epsilon : float
            Theoretical privacy budget.
        delta : float
            Privacy parameter for approximate LDP (not used for pure LDP).
        test_statistic : int
            Statistic to test against.

        Returns:
        -------
        int
            Count of successful inferences based on the test statistic.
        """

        # Suppress specific Numba warnings
        warnings.simplefilter('ignore', category=NumbaExperimentalFeatureWarning)
        
        setting_seed(random_state)
        np.random.seed(random_state)

        g = 2 # Binary LH (BLH) parameter
        count = 0
        for _ in range(trials):
            count += attack_lh(LH_Client(v, k, epsilon, False), k, g) == test_statistic
        return count
    
    @ray.remote
    def audit_olh(
        self, random_state, trials, v, k, epsilon, delta, test_statistic
    ) -> int:
        """
        Audits the Optimal Local Hashing (OLH) protocol.

        Parameters:
        ----------
        random_state : int
            Random seed for reproducibility.
        trials : int
            Number of trials to run.
        x : int
            True value.
        k : int
            Domain size.
        epsilon : float
            Theoretical privacy budget.
        delta : float
            Privacy parameter for approximate LDP (not used for pure LDP).
        test_statistic : int
            Statistic to test against.

        Returns:
        -------
        int
            Count of successful inferences based on the test statistic.
        """

        # Suppress specific Numba warnings
        warnings.simplefilter('ignore', category=NumbaExperimentalFeatureWarning)
        
        setting_seed(random_state)
        np.random.seed(random_state)

        g = int(np.round(np.exp(epsilon))) + 1 # Optimal LH (OLH) parameter
        count = 0
        for _ in range(trials):
            count += attack_lh(LH_Client(v, k, epsilon, True), k, g) == test_statistic
        return count

    #================================= Audit Methods for Approximate LDP Protocols =================================
    @ray.remote
    def audit_agrr(
        self, random_state, trials, v, k, epsilon, delta, test_statistic
    ) -> int:
        """
        Audits the Approximate Generalized Randomized Response (AGRR) protocol.

        Parameters:
        ----------
        random_state : int
            Random seed for reproducibility.
        trials : int
            Number of trials to run.
        v : int
            True value.
        k : int
            Domain size.
        epsilon : float
            Theoretical privacy budget.
        delta : float
            Privacy parameter for approximate LDP.
        test_statistic : int
            Statistic to test against.

        Returns:
        -------
        int
            Count of successful inferences based on the test statistic.
        """

        # Suppress specific Numba warnings
        warnings.simplefilter('ignore', category=NumbaExperimentalFeatureWarning)
        
        setting_seed(random_state)
        np.random.seed(random_state)

        count = 0
        for _ in range(trials):
            count += AGRR_Client(v, k, epsilon, delta) == test_statistic
        return count
    
    @ray.remote
    def audit_asue(
        self, random_state, trials, v, k, epsilon, delta, test_statistic
    ) -> int:
        """
        Audits the Approximate Symmetric Unary Encoding (ASUE) protocol.

        Parameters:
        ----------
        random_state : int
            Random seed for reproducibility.
        trials : int
            Number of trials to run.
        v : int
            True value.
        k : int
            Domain size.
        epsilon : float
            Theoretical privacy budget.
        delta : float
            Privacy parameter for approximate LDP.
        test_statistic : int
            Statistic to test against.

        Returns:
        -------
        int
            Count of successful inferences based on the test statistic.
        """

        # Suppress specific Numba warnings
        warnings.simplefilter('ignore', category=NumbaExperimentalFeatureWarning)
        
        setting_seed(random_state)
        np.random.seed(random_state)

        count = 0
        for _ in range(trials):
            count += attack_ue(ASUE_Client(v, k, epsilon, delta), k) == test_statistic
        return count

    @ray.remote
    def audit_agm(
        self, random_state, trials, v, k, epsilon, delta, test_statistic
    ) -> int:
        """
        Audits the Analytical Gaussian Mechanism (AGM).

        Parameters:
        ----------
        random_state : int
            Random seed for reproducibility.
        trials : int
            Number of trials to run.
        v : int
            True value.
        k : int
            Domain size.
        epsilon : float
            Theoretical privacy budget.
        delta : float
            Privacy parameter for approximate LDP.
        test_statistic : int
            Statistic to test against.

        Returns:
        -------
        int
            Count of successful inferences based on the test statistic.
        """

        # Suppress specific Numba warnings
        warnings.simplefilter('ignore', category=NumbaExperimentalFeatureWarning)
        
        setting_seed(random_state)
        np.random.seed(random_state)

        # Analytical Gaussian Mechanism (AGM)
        Delta_2 = np.sqrt(2)
        sigma = find_scale(epsilon, delta, Delta_2) 

        count = 0
        for _ in range(trials):
            count += attack_gm(GM_Client(v, k, sigma), k, sigma) == test_statistic
        return count

    @ray.remote
    def audit_gm(
        self, random_state, trials, v, k, epsilon, delta, test_statistic
    ) -> int:
        """
        Audits the Gaussian Mechanism (GM).

        Parameters:
        ----------
        random_state : int
            Random seed for reproducibility.
        trials : int
            Number of trials to run.
        v : int
            True value.
        k : int
            Domain size.
        epsilon : float
            Theoretical privacy budget.
        delta : float
            Privacy parameter for approximate LDP.
        test_statistic : int
            Statistic to test against.

        Returns:
        -------
        int
            Count of successful inferences based on the test statistic.
        """

        # Suppress specific Numba warnings
        warnings.simplefilter('ignore', category=NumbaExperimentalFeatureWarning)
        
        setting_seed(random_state)
        np.random.seed(random_state)

        # Standard Gaussian Mechanism (GM)
        Delta_2 = np.sqrt(2)
        sigma = (Delta_2 / epsilon) * np.sqrt(2 * np.log(1.25 / delta))

        count = 0
        for _ in range(trials):
            count += attack_gm(GM_Client(v, k, sigma), k, sigma) == test_statistic
        return count

    @ray.remote
    def audit_ablh(self, random_state, trials, v, k, epsilon, delta, test_statistic):
        """
        Audits the Approximate Binary Local Hashing (ABLH) protocol.

        Parameters:
        ----------
        random_state : int
            Random seed for reproducibility.
        trials : int
            Number of trials to run.
        v : int
            True value.
        k : int
            Domain size.
        epsilon : float
            Theoretical privacy budget.
        delta : float
            Privacy parameter for approximate LDP.
        test_statistic : int
            Statistic to test against.

        Returns:
        -------
        int
            Count of successful inferences based on the test statistic.
        """

        # Suppress specific Numba warnings
        warnings.simplefilter('ignore', category=NumbaExperimentalFeatureWarning)
        
        setting_seed(random_state)
        np.random.seed(random_state)

        g = 2
        count = 0
        for _ in range(trials):
            count += attack_lh(ALH_Client(v, epsilon, delta, False), k, g) == test_statistic
        return count

    @ray.remote
    def audit_aolh(self, random_state, trials, v, k, epsilon, delta, test_statistic):
        """
        Audits the Approximate Optimal Local Hashing (AOLH) protocol.

        Parameters:
        ----------
        random_state : int
            Random seed for reproducibility.
        trials : int
            Number of trials to run.
        v : int
            True value.
        k : int
            Domain size.
        epsilon : float
            Theoretical privacy budget.
        delta : float
            Privacy parameter for approximate LDP.
        test_statistic : int
            Statistic to test against.

        Returns:
        -------
        int
            Count of successful inferences based on the test statistic.
        """

        # Suppress specific Numba warnings
        warnings.simplefilter('ignore', category=NumbaExperimentalFeatureWarning)
        
        setting_seed(random_state)
        np.random.seed(random_state)

        g = max(2, int(np.round((-3 * np.exp(epsilon) * delta - np.sqrt(np.exp(epsilon) - 1) * np.sqrt((1 - delta) * (np.exp(epsilon) + delta - 9 * np.exp(epsilon) * delta - 1)) + np.exp(epsilon) + 3 * delta - 1) / (2 * delta))))
        count = 0

        for _ in range(trials):
            count += attack_lh(ALH_Client(v, epsilon, delta, True), k, g) == test_statistic
        return count

    #================================= Audit Methods for UE Protocols from Pure-LDP Package =================================

    @ray.remote
    def audit_sue_pure_ldp_pck(self, random_state, trials, v, k, epsilon, delta, test_statistic):
        """
        Audits the Symmetric Unary Encoding (SUE -- a.k.a. RAPPOR) protocol.

        Parameters:
        ----------
        random_state : int
            Random seed for reproducibility.
        trials : int
            Number of trials to run.
        v : int
            True value.
        k : int
            Domain size.
        epsilon : float
            Theoretical privacy budget.
        delta : float
            Privacy parameter for approximate LDP (not used for pure LDP).
        test_statistic : int
            Statistic to test against.

        Returns:
        -------
        int
            Count of successful inferences based on the test statistic.
        """

        # Suppress specific Numba warnings
        warnings.simplefilter('ignore', category=NumbaExperimentalFeatureWarning)
        
        setting_seed(random_state)
        np.random.seed(random_state)

        SUE_Client = UEClient(epsilon, k, use_oue=False, index_mapper=lambda x:x)

        count = 0
        for _ in range(trials):
            count += attack_ue(SUE_Client.privatise(v), k) == test_statistic
        return count
    
    @ray.remote
    def audit_oue_pure_ldp_pck(self, random_state, trials, v, k, epsilon, delta, test_statistic):
        """
        Audits the Symmetric Unary Encoding (SUE -- a.k.a. RAPPOR) protocol.

        Parameters:
        ----------
        random_state : int
            Random seed for reproducibility.
        trials : int
            Number of trials to run.
        v : int
            True value.
        k : int
            Domain size.
        epsilon : float
            Theoretical privacy budget.
        delta : float
            Privacy parameter for approximate LDP (not used for pure LDP).
        test_statistic : int
            Statistic to test against.

        Returns:
        -------
        int
            Count of successful inferences based on the test statistic.
        """

        # Suppress specific Numba warnings
        warnings.simplefilter('ignore', category=NumbaExperimentalFeatureWarning)
        
        setting_seed(random_state)
        np.random.seed(random_state)

        OUE_Client = UEClient(epsilon, k, use_oue=True, index_mapper=lambda x:x)

        count = 0
        for _ in range(trials):
            count += attack_ue(OUE_Client.privatise(v), k) == test_statistic
        return count
    
    #================================= General Audit Method =================================
    def run_audit(self, protocol_name) -> float:
        """
        Runs the audit for the specified LDP protocol.

        Parameters:
        ----------
        protocol_name : str
            The name of the LDP protocol to audit.

        Returns:
        -------
        float
            The estimated empirical epsilon value.

        Raises:
        ------
        ValueError
            If the specified protocol name is not supported.
        """

        if protocol_name not in self.protocols:
            raise ValueError(f"Unsupported protocol: {protocol_name}")

        if protocol_name == "LRT":
            # LRT は Ray 不要。確実に止める
            try:
                ray.shutdown()
            except Exception:
                pass
            self.nb_cores = 1
            return self._run_lrt_cf_once()
        
        # それ以外のプロトコルはここで Ray を準備
        try:
            ray.shutdown()
        except Exception:
            pass
        ray.init(num_cpus=self.nb_cores)

        protocol = self.protocols[protocol_name]

        TP, FP = [], []  # initialize list for parallelized results
        for idx in range(self.nb_cores):
            unique_seed: int = xxhash.xxh32(protocol_name).intdigest() + self.random_state + idx
            TP.append(protocol.remote(self, unique_seed, self.lst_trial_per_core[idx], self.v1, self.k, self.epsilon, self.delta, self.v1))
            FP.append(protocol.remote(self, unique_seed, self.lst_trial_per_core[idx], self.v2, self.k, self.epsilon, self.delta, self.v1))

        # take results with get function from Ray library
        TP_int: int = sum(ray.get(TP))
        FP_int: int = sum(ray.get(FP))

        # Clopper-Pearson confidence intervals
        TPR = proportion_confint(TP_int, self.nb_trials, self.alpha / 2, 'beta')[0]
        FPR = proportion_confint(FP_int, self.nb_trials, self.alpha / 2, 'beta')[1]

        # empirical epsilon estimation 
        self.eps_emp = float(np.log((TPR - self.delta) / FPR))

        if self.eps_emp > self.epsilon:
            warnings.warn(f"Empirical epsilon ({self.eps_emp}) exceeds theoretical epsilon ({self.epsilon}). There might be an error in the LDP-Auditor code or the LDP protocol being audited is wrong.")
        
        return self.eps_emp
    

# # Example
# seed = 42
# nb_trials = int(1e6)
# alpha = 1e-2
# epsilon = 1
# k = 25

# print('=====Auditing pure LDP protocols=====')
# delta = 0.0
# pure_ldp_protocols = ['GRR', 'SS', 'SUE', 'OUE', 'THE', 'SHE', 'BLH', 'OLH']
# auditor_pure_ldp = LDP_Auditor(nb_trials=nb_trials, alpha=alpha, epsilon=epsilon, delta=delta, k=k, random_state=seed, n_jobs=-1)

# for protocol in pure_ldp_protocols:
#     eps_emp = auditor_pure_ldp.run_audit(protocol)
#     print("{} eps_emp:".format(protocol), eps_emp)

# print('\n=====Auditing approximate LDP protocols=====')
# delta = 1e-5
# approx_ldp_protocols = ['AGRR', 'ASUE', 'AGM', 'GM', 'ABLH', 'AOLH']
# auditor_approx_ldp = LDP_Auditor(nb_trials=nb_trials, alpha=alpha, epsilon=epsilon, delta=delta, k=k, random_state=seed, n_jobs=-1)

# for protocol in approx_ldp_protocols:
#     eps_emp = auditor_approx_ldp.run_audit(protocol)
#     print("{} eps_emp:".format(protocol), eps_emp)

# if __name__ == "__main__":
#     main()