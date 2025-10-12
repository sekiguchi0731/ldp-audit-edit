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
from statsmodels.stats.proportion import proportion_confint

from .approximate_ldp import AGRR_Client, ALH_Client, ASUE_Client, GM_Client, find_scale
from .attacks import (
    attack_gm,
    attack_lh,
    attack_she,
    attack_ss,
    attack_the,
    attack_ue,
    attack_lrt_scores,
    compute_tpr_fpr,
    choose_tau_max_tpr_over_fpr,
)

# Our imports
from .utils import find_tresh, setting_seed
from .simulation import MixtureSpec, log_Rmax_effective_gaussian

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
        rmax_alpha: float = 0.01,
        spec: MixtureSpec | None = None,
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

        # for reproducibility
        self.random_state: int = random_state

        # audit parameters
        self.nb_trials: int = nb_trials
        self.alpha: float = alpha
        self.k: int = k  # domain size
        self.v1: int = 0
        self.v2: int = k - 1
        self.eps_emp: float = 0 # estimated empirical epsilon

        # theoretical LDP guarantees
        self.delta: float | int = delta
        self.epsilon: float | int = epsilon

        self.eta_val: np.ndarray | None = None
        self.eta_test: np.ndarray | None = None

        self.rmax_alpha: float = rmax_alpha
        self.spec: MixtureSpec | None = spec

        self.logRmax_eff: float = 0.0  # Effective log R_max for LRT-CF
        # for ray parallelism
        cpu_count: int | None = psutil.cpu_count()
        available_cores: int = int(cpu_count) if cpu_count is not None else 1
        if n_jobs == -1:
            self.nb_cores: int = available_cores
        else:
            self.nb_cores = min(available_cores, n_jobs)
        
        if hasattr(self, "eta_val") and self.eta_val is not None:
            ray.shutdown()  # avoid Ray init for LRT-CF
            self.nb_cores = 1
        else:
            ray.shutdown()
            ray.init(num_cpus=self.nb_cores)
        
        self.lst_trial_per_core: list[int] = [
            len(list(_)) for _ in np.array_split(range(nb_trials), self.nb_cores)
        ]

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
            'eta_val': self.eta_val,
            'eta_test': self.eta_test,
            'spec': self.spec,
            'rmax_alpha': self.rmax_alpha,
            'logRmax_eff': self.logRmax_eff,
        }

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
    
    def _run_lrt_cf_once(self) -> float:
        if self.eta_val is None or self.eta_test is None:
            raise ValueError("eta_val / eta_test を set_params で渡してください。")

        rng: np.random.Generator = np.random.default_rng(self.random_state)

        # --- 検証 ---
        eta_val: np.ndarray = np.asarray(self.eta_val, dtype=float)
        n_val: int = eta_val.shape[0]
        y_val: np.ndarray = (rng.random(n_val) < eta_val).astype(np.int64) # 正解ラベル
        ytilde_val: np.ndarray = np.empty(n_val, dtype=np.int64)
        for i in range(n_val):
            ytilde_val[i] = GRR_Client(int(y_val[i]), 2, float(self.epsilon))  # ラベルRRは k=2

        scores_val: np.ndarray = attack_lrt_scores(ytilde_val, eta_val, float(self.epsilon))
        tau, _, _ = choose_tau_max_tpr_over_fpr(scores_val, y_val, clip=1e-12)
        # ---  Effective log R_max を追加 ---
        if self.logRmax_eff and np.isfinite(self.logRmax_eff) and self.logRmax_eff > 0:
            # すでに外部から与えられている場合はそれを使う
            print("Using externally provided logRmax_eff:", self.logRmax_eff)
            pass
        elif self.spec is not None:
            # 解析式で ln R_max^(alpha) を計算（理論に沿う）
            print("Computing logRmax_eff from spec:", self.spec)
            self.logRmax_eff = float(log_Rmax_effective_gaussian(self.spec, alpha=self.rmax_alpha))
        else:
            e: np.ndarray = np.clip(eta_val.astype(float), 1e-6, 1 - 1e-6)
            # 各 i において大きい方を格納
            rmax_x: np.ndarray = np.maximum(e / (1 - e), (1 - e) / e)
            logR: np.ndarray = np.abs(np.log(rmax_x))
            self.logRmax_eff = float(np.quantile(logR, 0.99))
        # --- 評価 ---
        eta_test: np.ndarray = np.asarray(self.eta_test, dtype=float)
        n_test: int = eta_test.shape[0]
        y_test: np.ndarray = (rng.random(n_test) < eta_test).astype(np.int64)
        ytilde_test: np.ndarray = np.empty(n_test, dtype=np.int64)
        for i in range(n_test):
            ytilde_test[i] = GRR_Client(int(y_test[i]), 2, float(self.epsilon))

        scores_test: np.ndarray = attack_lrt_scores(ytilde_test, eta_test, float(self.epsilon))
        tau = float(np.clip(tau, scores_test.min(), scores_test.max()))
        tpr, fpr = compute_tpr_fpr(scores_test, y_test, tau)

        clip = 1e-12
        eps_emp = float(np.log(max(tpr, clip) / max(fpr, clip)))
        self.eps_emp = eps_emp
        print("=== LRT DEBUG ===")
        print(f"eps={self.epsilon}, seed={self.random_state}")
        print(f"val: n={n_val}, scores=[{scores_val.min():.4g}, {scores_val.max():.4g}]")
        print(f"test: n={n_test}, scores=[{scores_test.min():.4g}, {scores_test.max():.4g}]")
        print(f"tau={tau:.6g}")
        print("y_val counts:", np.unique(y_val, return_counts=True))
        print("y_test counts:", np.unique(y_test, return_counts=True))
        print("ytilde_test counts:", np.unique(ytilde_test, return_counts=True))
        print(f"tpr={tpr:.6g}, fpr={fpr:.6g}")
        print(f"eps_emp(computed)={np.log(max(tpr,1e-12)/max(fpr,1e-12)):.6g}")
        print("=== /LRT DEBUG ===")
        return eps_emp


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
        )
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