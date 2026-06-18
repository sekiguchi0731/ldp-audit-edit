# General imports
import warnings
from collections import defaultdict
from typing import Any, Literal, NoReturn, Self, Callable, Sequence, cast

import pandas as pd
import numpy as np
import psutil

# Imports for parallelization
try:
    import ray
except ModuleNotFoundError:
    class _MissingRayRemote:
        def __init__(self, func) -> None:
            self.func = func

        def remote(self, *args, **kwargs) -> NoReturn:
            raise ModuleNotFoundError(
                "ray is required for non-LRT parallel audits. Install ray or run an LRT/decomp path."
            )

    class _MissingRay:
        @staticmethod
        def remote(func=None, **_kwargs):# -> Callable[..., _MissingRayRemote] | _MissingRayRemote:
            if func is None:
                return lambda f: _MissingRayRemote(f)
            return _MissingRayRemote(func)

        @staticmethod
        def shutdown() -> None:
            return None

        @staticmethod
        def init(*_args, **_kwargs) -> NoReturn:
            raise ModuleNotFoundError(
                "ray is required for non-LRT parallel audits. Install ray or run an LRT/decomp path."
            )

        @staticmethod
        def get(_refs):
            raise ModuleNotFoundError(
                "ray is required for non-LRT parallel audits. Install ray or run an LRT/decomp path."
            )

    ray = _MissingRay()
import xxhash

from scipy.optimize import OptimizeResult, minimize_scalar
from scipy.special import erfinv, expit
from statsmodels.stats.proportion import proportion_confint

from ldp_audit.eta_models import EtaModelConfig

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
from .eta_models import (
    get_default_eta_model_configs,
    fit_and_select_eta_model,
    predict_eta,
)
from .simulation import (
    MixtureSpec,
    log_Rmax_gaussian,
    posterior_probs_from_x,
    gaussian_x_lr_affine_params,
    sample_x_given_y_truncated,
    set_B_from_quantile,
    sample_attack_trainset_from_spec,
)

# Our imports
from .utils import find_tresh, setting_seed

warnings.simplefilter("ignore")

FeatureMatrix = np.ndarray | pd.DataFrame

try:
    from numba.core.errors import NumbaExperimentalFeatureWarning   # type: ignore
except Exception:
    class NumbaExperimentalFeatureWarning(Warning):
        pass


def _get_grr_client():
    from multi_freq_ldpy.pure_frequency_oracles.GRR import GRR_Client
    return GRR_Client


def _get_he_client():
    from multi_freq_ldpy.pure_frequency_oracles.HE import HE_Client
    return HE_Client


def _get_lh_client():
    from multi_freq_ldpy.pure_frequency_oracles.LH import LH_Client
    return LH_Client


def _get_ss_client():
    from multi_freq_ldpy.pure_frequency_oracles.SS import SS_Client
    return SS_Client


def _get_ue_client():
    from multi_freq_ldpy.pure_frequency_oracles.UE import UE_Client
    return UE_Client


def _get_pure_ldp_ue_client():
    from pure_ldp.frequency_oracles.unary_encoding import UEClient
    return UEClient


def _get_approximate_ldp_symbols():
    from .approximate_ldp import AGRR_Client, ALH_Client, ASUE_Client, GM_Client, find_scale
    return AGRR_Client, ALH_Client, ASUE_Client, GM_Client, find_scale


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
        dynamic_nb_trials: bool = True,
        sim_hat: bool = True,
        real_val_X: FeatureMatrix | None = None,
        real_val_y: np.ndarray | None = None,
        real_threshold_X: FeatureMatrix | None = None,
        real_threshold_y: np.ndarray | None = None,
        real_final_X: FeatureMatrix | None = None,
        real_final_y: np.ndarray | None = None,
        eta_class_prior: tuple[float, float] | None = None,
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
        if not isinstance(dynamic_nb_trials, bool):
            raise ValueError("dynamic_nb_trials must be a boolean value.")

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
        self.sim_hat: bool = bool(sim_hat)
        self.real_val_X: FeatureMatrix | None = real_val_X
        self.real_val_y: np.ndarray | None = real_val_y
        self.real_threshold_X: FeatureMatrix | None = real_threshold_X
        self.real_threshold_y: np.ndarray | None = real_threshold_y
        self.real_final_X: FeatureMatrix | None = real_final_X
        self.real_final_y: np.ndarray | None = real_final_y
        self.eta_class_prior: tuple[float, float] | None = self._validate_eta_class_prior(
            eta_class_prior
        )

        if not self.sim_hat:
            if (
                self.real_val_X is None
                or self.real_val_y is None
                or self.real_threshold_X is None
                or self.real_threshold_y is None
                or self.real_final_X is None
                or self.real_final_y is None
            ):
                raise ValueError(
                    "When sim_hat=False, real_val_X/real_val_y/real_threshold_X/"
                    "real_threshold_y/real_final_X/real_final_y must be provided."
                )

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

        # dynamic に計算するか
        self.dynamic_nb_trials: bool = dynamic_nb_trials
        self._apply_dynamic_nb_trials()
        print("Final nb_trials:", self.nb_trials)

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
            "LRT_tilde1": "SPECIAL",  # tilde y = 1 only
            "LRT_decomp": "SPECIAL",  # X-only decomposition audit
            "LRT_indirect": "SPECIAL",  # backwards-compatible alias for LRT_decomp
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
                if key == "eta_class_prior":
                    value: tuple[float, float] | None = self._validate_eta_class_prior(value)
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
            'eta_class_prior': self.eta_class_prior,
        }

    @staticmethod
    def _validate_eta_class_prior(
        class_prior: tuple[float, float] | None,
    ) -> tuple[float, float] | None:
        """Validate class prior as (P[Y=0], P[Y=1])."""
        if class_prior is None:
            return None
        if len(class_prior) != 2:
            raise ValueError("eta_class_prior must be a tuple (P[Y=0], P[Y=1]).")
        prior_0, prior_1 = (float(class_prior[0]), float(class_prior[1]))
        if prior_0 <= 0.0 or prior_1 <= 0.0:
            raise ValueError("eta_class_prior entries must be positive.")
        total: float = prior_0 + prior_1
        if not np.isfinite(total) or total <= 0.0:
            raise ValueError("eta_class_prior must be finite and positive.")
        return (prior_0 / total, prior_1 / total)

    def _log_eta_prior_correction(self) -> float:
        """
        Convert posterior odds eta/(1-eta) to class-conditional density ratio.

        eta_hat(x) estimates P(Y=1|X=x), so
            eta/(1-eta) = p(x|Y=1) P(Y=1) / (p(x|Y=0) P(Y=0)).
        The DP-Sniper score compares p(z|Y=1) / p(z|Y=0), hence the correction
        factor P(Y=0) / P(Y=1).  Simulation data in this repo has equal priors,
        so the default correction is zero.
        """
        if self.eta_class_prior is None:
            return 0.0
        prior_0, prior_1 = self.eta_class_prior
        return float(np.log(prior_0) - np.log(prior_1))

    def _eta_to_x_lr_log_score(self, eta: np.ndarray) -> np.ndarray:
        eta = np.clip(np.asarray(eta, dtype=np.float64), 1e-15, 1.0 - 1e-15)
        return np.log(eta) - np.log1p(-eta) + self._log_eta_prior_correction()

    def _eta_to_x_lr_prob(self, eta: np.ndarray) -> np.ndarray:
        return expit(self._eta_to_x_lr_log_score(eta))

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
        # base terms inside max(.)
        # B1 = 2(1-c)/(omega^2 c)
        base1: float = 2.0 * (1.0 - c_val) / (omega**2 * c_val)

        # B2 = 8(1-c)/c
        base2: float = 8.0 * (1.0 - c_val) / c_val

        # common factor: (erf^{-1}(1-2beta))^2
        erf_arg: float = 1.0 - 2.0 * beta
        # numerical safety: clip into (-1,1)
        erf_arg = float(np.clip(erf_arg, -0.999999, 0.999999))
        inv_val: float = float(erfinv(erf_arg))
        erf_sq: float = inv_val**2

        suggested: int = int(np.ceil(max(base1, base2) * erf_sq))
        print("Suggested nb_trials from dynamic adjustment:", suggested)

        # avoid runaway trial counts
        # max_cap: int = int(1e7)
        # suggested = int(min(max(suggested, 1), max_cap))

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
        GRR_Client = _get_grr_client()

        count = 0
        for _ in range(trials):            
            count += GRR_Client(v, k, epsilon) == test_statistic
        return count
    
    ############# LRT 監査用の関数群 #############

    ##### eta 学習モデルをつかう #####
    def _take_feature_rows(
        self,
        X: FeatureMatrix,
        row_selector: np.ndarray,
    ) -> FeatureMatrix:
        if isinstance(X, pd.DataFrame):
            if row_selector.dtype == bool:
                return X.iloc[np.flatnonzero(row_selector)]
            return X.iloc[row_selector]
        return X[row_selector]

    def _sample_x_given_y_for_eta_hat(
        self,
        *,
        y_input: int,
        n: int,
        rng: np.random.Generator,
        sample_source: Literal["val", "threshold", "final"] = "final",
    ) -> FeatureMatrix:
        if self.sim_hat:
            if self.spec is None:
                raise ValueError("spec must be set.")
            X: np.ndarray = sample_x_given_y_truncated(
                n=n,
                spec=self.spec,
                y=y_input,
                rng=rng,
                B=self.B,
            )
            return self.project_l2_ball(X, B=self.B)

        if sample_source == "val":
            X_pool: FeatureMatrix | None = self.real_val_X
            y_pool: np.ndarray | None = self.real_val_y
        elif sample_source == "threshold":
            X_pool: FeatureMatrix | None = self.real_threshold_X
            y_pool: np.ndarray | None = self.real_threshold_y
        else:
            X_pool: FeatureMatrix | None = self.real_final_X
            y_pool: np.ndarray | None = self.real_final_y
        assert X_pool is not None and y_pool is not None

        idx_all: np.ndarray = np.flatnonzero(y_pool == y_input)
        if idx_all.size == 0:
            raise ValueError(
                f"No real-data samples with y={y_input} were found in {sample_source} pool."
            )
        chosen: np.ndarray = rng.choice(idx_all, size=int(n), replace=True)
        return self._take_feature_rows(X_pool, chosen)

    def _sample_p_theta_hat(
        self,
        y_input: int,
        n: int,
        rng: np.random.Generator,
        eta_model,
        sample_source: Literal["val", "threshold", "final"] = "final",
    ) -> np.ndarray:
        """
        完全LRT用：eta(x) を学習モデルで推定する版
        p((x,\tilde y)|Y=1) / p((x,\tilde y)|Y=0) の単調変換を返す
        """
        X: FeatureMatrix = self._sample_x_given_y_for_eta_hat(
            y_input=y_input,
            n=n,
            rng=rng,
            sample_source=sample_source,
        )

        eta_hat: np.ndarray = predict_eta(eta_model, X)
        ytilde: np.ndarray = grr_sample_binary(y=y_input, n=n, epsilon=float(self.epsilon), rng=rng)

        # LRT の対数スコアを計算する。
        # attack_lrt_scores は eta posterior odds を使うため、実データの不均衡 prior は
        # p(z|Y=1)/p(z|Y=0) に直す補正 log(P[Y=0]/P[Y=1]) を足す。
        # model の predect は P(Y=1|x,\tilde y)/p(Y=0|x, \tilde y)　だから、
        # そこから LRT で使う値である p(x, \tilde y|Y=1)/p(x, \tilde y|Y=0) に変換する場合、
        # Pr[Y=0]/Pr[Y=1] をかける（つまり対数での加算をする）必要がある
        llr: np.ndarray = (
            attack_lrt_scores(ytilde, eta_hat, float(self.epsilon))
            + self._log_eta_prior_correction()
        )
        p = expit(llr)
        return p


    def _sample_score_x_hat(
        self,
        y_input: int,
        n: int,
        rng: np.random.Generator,
        eta_model,
        sample_source: Literal["val", "threshold", "final"] = "final",
    ) -> np.ndarray:
        """
        X-only（Decomposition）用：
        score_x = logit(eta_hat(x)) + log(P[Y=0]/P[Y=1])
        """
        X: FeatureMatrix = self._sample_x_given_y_for_eta_hat(
            y_input=y_input,
            n=n,
            rng=rng,
            sample_source=sample_source,
        )

        eta_hat: np.ndarray = predict_eta(eta_model, X)
        score_x = self._eta_to_x_lr_log_score(eta_hat)
        return score_x

    def evaluate_eps_with_dp_sniper_cp_hat(
        self,
        tau: float,
        q: float,
        rng: np.random.Generator,
        eta_model,
        attack: Literal["complete_LRT_hat", "tilde y=y_alt only LRT_hat", "indirect_LRT_hat"],
        y_alt: int = 1,
        y_null: int = 0,
        sample_source: Literal["val", "threshold", "final"] = "final",
        n_alt_eval: int | None = None,
        n_null_eval: int | None = None,
    ) -> dict:
        """
        学習etaモデルを使う版の評価
        TP, FP, tpr_hat, fpr_hat, eps_emp, eps_ci などを入れた dict を返す
        sample_source: "val" or "final" -- 
        etaモデルの学習に使ったデータから（データ数で）サンプリングするか、最終評価用のデータからサンプリングするか
        """
        if self.sim_hat and self.spec is None:
            raise ValueError("spec must be set for evaluation.")

        def _resolve_eval_counts() -> tuple[int, int]:
            if n_alt_eval is not None and n_null_eval is not None:
                n_alt: int = int(n_alt_eval)
                n_null: int = int(n_null_eval)
            elif not self.sim_hat:
                if sample_source == "val":
                    y_pool: np.ndarray | None = self.real_val_y
                elif sample_source == "threshold":
                    y_pool: np.ndarray | None = self.real_threshold_y
                else:
                    y_pool: np.ndarray | None = self.real_final_y
                assert y_pool is not None
                n_alt = int(np.sum(y_pool == y_alt))
                n_null = int(np.sum(y_pool == y_null))
            else:
                n_alt = int(self.nb_trials)
                n_null = int(self.nb_trials)

            if min(n_alt, n_null) <= 0:
                raise ValueError(
                    f"Evaluation counts must be positive. Got n_alt={n_alt}, n_null={n_null}, "
                    f"sample_source={sample_source}."
                )
            return n_alt, n_null

        n_alt, n_null = _resolve_eval_counts()

        def _sample_attack_outputs(p: np.ndarray) -> int:
            # S^{tau,q}(.) を実際にサンプルして「1 を出した回数」を返す
            atol = 1e-12
            tau_eff: float = max(float(tau), 1e-15)
            gt: np.ndarray = p > tau_eff + atol
            eq: np.ndarray = np.abs(p - tau_eff) <= atol
            out: np.ndarray = np.zeros(p.shape[0], dtype=np.int64)
            out[gt] = 1
            if np.any(eq):
                out[eq] = (rng.random(np.sum(eq)) < q).astype(np.int64)
            return int(out.sum())

        if attack == "complete_LRT_hat":
            p1: np.ndarray = self._sample_p_theta_hat(
                y_alt,
                n_alt,
                rng,
                eta_model=eta_model,
                sample_source=sample_source,
            )
            p0: np.ndarray = self._sample_p_theta_hat(
                y_null,
                n_null,
                rng,
                eta_model=eta_model,
                sample_source=sample_source,
            )

        elif attack == "tilde y=y_alt only LRT_hat":        # 無視で良い
            score1 = self._sample_score_x_hat(
                y_alt,
                n_alt,
                rng,
                eta_model=eta_model,
                sample_source=sample_source,
            )
            score0 = self._sample_score_x_hat(
                y_null,
                n_null,
                rng,
                eta_model=eta_model,
                sample_source=sample_source,
            )

            ytilde1: np.ndarray = grr_sample_binary(
                y=y_alt, n=n_alt, epsilon=float(self.epsilon), rng=rng
            )
            ytilde0: np.ndarray = grr_sample_binary(
                y=y_null, n=n_null, epsilon=float(self.epsilon), rng=rng
            )

            p1: np.ndarray = np.zeros(n_alt, dtype=np.float64)
            p0: np.ndarray = np.zeros(n_null, dtype=np.float64)

            m1: np.ndarray = ytilde1 == y_alt
            m0: np.ndarray = ytilde0 == y_alt

            p1[m1] = expit(score1[m1])
            p0[m0] = expit(score0[m0])

        elif attack == "indirect_LRT_hat":
            score1: np.ndarray = self._sample_score_x_hat(
                y_alt,
                n_alt,
                rng,
                eta_model=eta_model,
                sample_source=sample_source,
            )
            score0: np.ndarray = self._sample_score_x_hat(
                y_null,
                n_null,
                rng,
                eta_model=eta_model,
                sample_source=sample_source,
            )
            p1 = expit(score1)
            p0 = expit(score0)

        else:
            raise ValueError("Unknown attack type")

        TP: int = _sample_attack_outputs(p1)
        FP: int = _sample_attack_outputs(p0)

        tpr_hat: float = TP / n_alt
        fpr_hat: float = FP / n_null

        tpr_lo, tpr_hi = proportion_confint(TP, n_alt, alpha=self.alpha, method="beta")
        fpr_lo, fpr_hi = proportion_confint(FP, n_null, alpha=self.alpha, method="beta")

        # DP-Sniper の c-power: ln_{>=c}(x)=ln(max(c, x)) を使う
        # FPR が validation/test で c を割り込んだ場合でも下駄履きで安定化する
        c_floor: float = max(float(self.c), 1e-12)
        eps_emp = float(np.log(max(tpr_hat, c_floor) / max(fpr_hat, c_floor)))
        eps_lower = float(np.log(max(tpr_lo, c_floor) / max(fpr_hi, c_floor)))
        eps_upper = float(np.log(max(tpr_hi, c_floor) / max(fpr_lo, c_floor)))

        return {
            "TP": TP,
            "FP": FP,
            "N_final": int(self.nb_trials),
            "N_alt_eval": n_alt,
            "N_null_eval": n_null,
            "tpr_hat": float(tpr_hat),
            "fpr_hat": float(fpr_hat),
            "tpr_ci": (float(tpr_lo), float(tpr_hi)),       # type: ignore
            "fpr_ci": (float(fpr_lo), float(fpr_hi)),       # type: ignore
            "eps_emp": eps_emp,
            "eps_ci": (eps_lower, eps_upper),
        }

    def _sample_attack_scores_hat(
        self,
        *,
        y_input: int,
        n: int,
        rng: np.random.Generator,
        eta_model,
        attack: Literal["complete_LRT_hat", "tilde y=y_alt only LRT_hat", "indirect_LRT_hat"],
        sample_source: Literal["val", "threshold", "final"] = "final",
        y_alt: int = 1,
    ) -> np.ndarray:
        """Sample the scalar attack score used by threshold attacks."""
        if attack == "complete_LRT_hat":
            return self._sample_p_theta_hat(
                y_input=y_input,
                n=n,
                rng=rng,
                eta_model=eta_model,
                sample_source=sample_source,
            )

        if attack == "indirect_LRT_hat":
            return expit(
                self._sample_score_x_hat(
                    y_input=y_input,
                    n=n,
                    rng=rng,
                    eta_model=eta_model,
                    sample_source=sample_source,
                )
            )

        if attack == "tilde y=y_alt only LRT_hat":
            score: np.ndarray = self._sample_score_x_hat(
                y_input=y_input,
                n=n,
                rng=rng,
                eta_model=eta_model,
                sample_source=sample_source,
            )
            ytilde: np.ndarray = grr_sample_binary(
                y=y_input,
                n=n,
                epsilon=float(self.epsilon),
                rng=rng,
            )
            out = np.zeros(n, dtype=np.float64)
            mask: np.ndarray = ytilde == y_alt
            out[mask] = expit(score[mask])
            return out

        raise ValueError(f"Unknown attack type: {attack}")

    @staticmethod
    def _counts_above_thresholds(scores: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
        """Return counts of scores satisfying scores > threshold for every threshold."""
        scores_sorted: np.ndarray = np.sort(np.asarray(scores, dtype=np.float64))
        thresholds_arr: np.ndarray = np.asarray(thresholds, dtype=np.float64)
        idx: np.ndarray = np.searchsorted(scores_sorted, thresholds_arr + 1e-12, side="right")
        return (scores_sorted.size - idx).astype(np.int64)

    @staticmethod
    def _beta_cp_interval(
        counts: np.ndarray,
        n: int,
        *,
        one_sided_alpha: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        cp_alpha: float = float(np.clip(2.0 * one_sided_alpha, 1e-300, 1.0 - 1e-12))
        lo, hi = proportion_confint(counts, n, alpha=cp_alpha, method="beta")
        return np.asarray(lo, dtype=np.float64), np.asarray(hi, dtype=np.float64)

    def evaluate_eps_with_theory_grid_cp_hat(
        self,
        *,
        tau_grid_scores: np.ndarray,
        rng: np.random.Generator,
        eta_model,
        attack: Literal["complete_LRT_hat", "tilde y=y_alt only LRT_hat", "indirect_LRT_hat"],
        y_alt: int = 1,
        y_null: int = 0,
        sample_source: Literal["val", "threshold", "final"] = "final",
        n_alt_eval: int | None = None,
        n_null_eval: int | None = None,
    ) -> dict:
        """
        Evaluate a finite threshold grid with CP-LCB and choose the best threshold.

        The grid must be built independently of the evaluation sample.  We use a
        Bonferroni split across all grid thresholds and the one-sided CP events
        used here: numerator lower/upper bounds, denominator upper bound, and
        denominator lower bound for conservative c-feasibility.
        """
        if self.sim_hat and self.spec is None:
            raise ValueError("spec must be set for evaluation.")

        tau_grid_arr: np.ndarray = np.asarray(tau_grid_scores, dtype=np.float64)
        grid: np.ndarray = np.unique(tau_grid_arr[np.isfinite(tau_grid_arr)])
        grid = np.clip(grid, 1e-15, 1.0 - 1e-15)
        grid = np.unique(grid)
        if grid.size == 0:
            raise ValueError("tau_grid_scores must contain at least one finite value.")

        def _resolve_eval_counts() -> tuple[int, int]:
            if n_alt_eval is not None and n_null_eval is not None:
                n_alt: int = int(n_alt_eval)
                n_null: int = int(n_null_eval)
            elif not self.sim_hat:
                if sample_source == "val":
                    y_pool: np.ndarray | None = self.real_val_y
                elif sample_source == "threshold":
                    y_pool = self.real_threshold_y
                else:
                    y_pool = self.real_final_y
                assert y_pool is not None
                n_alt = int(np.sum(y_pool == y_alt))
                n_null = int(np.sum(y_pool == y_null))
            else:
                n_alt = int(self.nb_trials)
                n_null = int(self.nb_trials)

            if min(n_alt, n_null) <= 0:
                raise ValueError(
                    f"Evaluation counts must be positive. Got n_alt={n_alt}, n_null={n_null}, "
                    f"sample_source={sample_source}."
                )
            return n_alt, n_null

        n_alt, n_null = _resolve_eval_counts()
        p_alt: np.ndarray = self._sample_attack_scores_hat(
            y_input=y_alt,
            n=n_alt,
            rng=rng,
            eta_model=eta_model,
            attack=attack,
            sample_source=sample_source,
            y_alt=y_alt,
        )
        p_null: np.ndarray = self._sample_attack_scores_hat(
            y_input=y_null,
            n=n_null,
            rng=rng,
            eta_model=eta_model,
            attack=attack,
            sample_source=sample_source,
            y_alt=y_alt,
        )

        tp_by_t: np.ndarray = self._counts_above_thresholds(p_alt, grid)
        fp_by_t: np.ndarray = self._counts_above_thresholds(p_null, grid)
        tpr_hat_by_t: np.ndarray = tp_by_t / float(n_alt)
        fpr_hat_by_t: np.ndarray = fp_by_t / float(n_null)

        one_sided_alpha: float = float(self.alpha) / max(4 * int(grid.size), 1)
        tpr_lo_by_t, tpr_hi_by_t = self._beta_cp_interval(
            tp_by_t,
            n_alt,
            one_sided_alpha=one_sided_alpha,
        )
        fpr_lo_by_t, fpr_hi_by_t = self._beta_cp_interval(
            fp_by_t,
            n_null,
            one_sided_alpha=one_sided_alpha,
        )

        eps_floor: float = 1e-300
        eps_emp_by_t: np.ndarray = np.log(
            np.maximum(tpr_hat_by_t, eps_floor) / np.maximum(fpr_hat_by_t, eps_floor)
        )
        eps_lower_by_t: np.ndarray = np.log(
            np.maximum(tpr_lo_by_t, eps_floor) / np.maximum(fpr_hi_by_t, eps_floor)
        )
        eps_upper_by_t: np.ndarray = np.log(
            np.maximum(tpr_hi_by_t, eps_floor) / np.maximum(fpr_lo_by_t, eps_floor)
        )

        c_floor: float = max(float(self.c), 1e-12)
        feasible: np.ndarray = fpr_lo_by_t >= c_floor
        if np.any(feasible):
            candidate_indices: np.ndarray = np.flatnonzero(feasible)
        else:
            # Strict c-feasibility can be empty for small final samples.  Keep the
            # run inspectable without pretending the c-clipped theorem applies.
            candidate_indices = np.arange(grid.size)
            eps_lower_by_t = np.full_like(eps_lower_by_t, -np.inf, dtype=np.float64)

        best_relative_idx: int = int(np.argmax(eps_lower_by_t[candidate_indices]))
        best_idx: int = int(candidate_indices[best_relative_idx])

        return {
            "TP": int(tp_by_t[best_idx]),
            "FP": int(fp_by_t[best_idx]),
            "N_final": int(self.nb_trials),
            "N_alt_eval": int(n_alt),
            "N_null_eval": int(n_null),
            "tpr_hat": float(tpr_hat_by_t[best_idx]),
            "fpr_hat": float(fpr_hat_by_t[best_idx]),
            "tpr_ci": (float(tpr_lo_by_t[best_idx]), float(tpr_hi_by_t[best_idx])),
            "fpr_ci": (float(fpr_lo_by_t[best_idx]), float(fpr_hi_by_t[best_idx])),
            "eps_emp": float(eps_emp_by_t[best_idx]),
            "eps_ci": (float(eps_lower_by_t[best_idx]), float(eps_upper_by_t[best_idx])),
            "tau": float(grid[best_idx]),
            "q": 0.0,
            "tau_selection": "theory",
            "tau_grid_size": int(grid.size),
            "tau_grid_feasible_size": int(np.sum(feasible)),
            "tau_cp_one_sided_alpha": float(one_sided_alpha),
            "tau_selected_index": int(best_idx),
        }
    
    def _select_score_fn_for_eta_model(
        self,
        *,
        X_val: FeatureMatrix,
        y_val: np.ndarray,
        selection: Literal["eps_lower", "eps_emp", "tpr_at_fpr"] = "eps_lower",
        rng_seed_for_val: int = 0,
        attack_for_selection: Literal[
            "indirect_LRT_hat", "complete_LRT_hat"
        ] = "indirect_LRT_hat",
        y_alt: int = 1,
        y_null: int = 0,
    ) -> Callable[..., float]:
        """
        eta_model（sklearn pipeline）を受け取って,val上のスカラーを返す score_fn を作る.
        attack_for_selection に応じて、null 側スコアから dp-sniper の tau,q を決める。
        - indirect_LRT_hat: y=0 側 predicted eta から tau,q
        - complete_LRT_hat: y=0 側の complete-LRT pseudo-probability から tau,q
        """
        rng_local: np.random.Generator = np.random.default_rng(rng_seed_for_val)

        def score_fn(model) -> float:
            # attack_for_selection に応じて null 側スコアから tau,q を作る
            if attack_for_selection == "indirect_LRT_hat":
                X_null = self._take_feature_rows(X_val, y_val == y_null)
                scores_null: np.ndarray = self._eta_to_x_lr_prob(
                    predict_eta(model, X_null)
                )
            elif attack_for_selection == "complete_LRT_hat":
                rng_tau: np.random.Generator = np.random.default_rng(rng_seed_for_val + 12345)
                scores_null = self._sample_p_theta_hat(
                    y_input=y_null,
                    n=int(np.sum(y_val == y_null)),
                    rng=rng_tau,
                    eta_model=model,
                    sample_source="val",
                )
            else:
                raise ValueError(f"Unknown attack_for_selection: {attack_for_selection}")

            tau, q = dp_sniper_threshold_from_scores(scores_null, c=self.c)
            tau = float(np.clip(tau, 1e-15, 1.0 - 1e-15))
            q = float(np.clip(q, 0.0, 1.0))

            # valで監査指標を測る
            res: dict = self.evaluate_eps_with_dp_sniper_cp_hat(
                tau=tau,
                q=q,
                rng=rng_local,
                eta_model=model,
                attack=(
                    "indirect_LRT_hat"
                    if attack_for_selection == "indirect_LRT_hat"
                    else "complete_LRT_hat"
                ),
                y_alt=y_alt,
                y_null=y_null,
                sample_source="val",
                n_alt_eval=int(np.sum(y_val == y_alt)),
                n_null_eval=int(np.sum(y_val == y_null)),
            )

            if selection == "eps_emp":
                return float(res["eps_emp"])
            if selection == "tpr_at_fpr":
                return float(res["tpr_hat"])
            return float(res["eps_ci"][0])

        return score_fn


    def run_eta_model_comparison_4way(
        self,
        *,
        # 共通
        seed: int = 0,
        selection: Literal["eps_lower", "eps_emp", "tpr_at_fpr"] = "eps_lower",
        tau_selection: Literal["cN", "theory"] = "cN",
        report_attacks: Sequence[
            Literal["complete_LRT_hat", "tilde y=y_alt only LRT_hat", "indirect_LRT_hat"]
        ] | None = None,
        eta_model_cfgs: Sequence[EtaModelConfig] | None = None,
        # simulation用（実データなら None でOK）
        sim_n_train: int | None = 4000,
        sim_n_val: int | None = 2000,
        sim_n_threshold: int | None = 2000,
        # 実データ用（simulationなら None でOK）
        X_train: FeatureMatrix | None = None,
        y_train: np.ndarray | None = None,
        X_val: FeatureMatrix | None = None,
        y_val: np.ndarray | None = None,
        X_threshold: FeatureMatrix | None = None,
        y_threshold: np.ndarray | None = None,
        prefit_model_best_by_attack: dict[str, dict[str, dict[str, Any]]] | None = None,
        y_alt: int = 1,
        y_null: int = 0,
    ) -> dict:
        """
        4モデルそれぞれで best hyperparam を選び、test評価（= evaluate）を返す。
        - simulation: sim_n_train/sim_n_val/sim_n_threshold を指定し、specからデータ生成
        - real: train/model_val/threshold split を渡す
        """
        if self.spec is None:
            raise ValueError("spec must be set.")
        if tau_selection not in ("cN", "theory"):
            raise ValueError("tau_selection must be either 'cN' or 'theory'.")

        rng: np.random.Generator = np.random.default_rng(seed)
        rng_train: np.random.Generator = np.random.default_rng(rng.integers(1 << 32))
        rng_val: np.random.Generator = np.random.default_rng(rng.integers(1 << 32))
        rng_threshold: np.random.Generator = np.random.default_rng(rng.integers(1 << 32))
        rng_test: np.random.Generator = np.random.default_rng(rng.integers(1 << 32))

        # --- train/val データを用意 （simulation data 用） ---
        if X_train is None:
            if sim_n_train is None or sim_n_val is None or sim_n_threshold is None:
                raise ValueError("For simulation mode, set sim_n_train, sim_n_val, and sim_n_threshold.")
            # y も含めてサンプリング
            X_train, y_train = sample_attack_trainset_from_spec(
                n=sim_n_train,
                spec=self.spec,
                rng=rng_train,
                B=self.B,
                project_fn=self.project_l2_ball,
            )
            X_val, y_val = sample_attack_trainset_from_spec(
                n=sim_n_val,
                spec=self.spec,
                rng=rng_val,
                B=self.B,
                project_fn=self.project_l2_ball,
            )
            X_threshold, y_threshold = sample_attack_trainset_from_spec(
                n=sim_n_threshold,
                spec=self.spec,
                rng=rng_threshold,
                B=self.B,
                project_fn=self.project_l2_ball,
            )
        else:
            # real mode
            assert (
                y_train is not None
                and X_val is not None
                and y_val is not None
                and X_threshold is not None
                and y_threshold is not None
            )

        cfgs: list[EtaModelConfig] = (
            list(eta_model_cfgs) if eta_model_cfgs is not None else get_default_eta_model_configs()
        )

        report_attack_list: list[
            Literal["complete_LRT_hat", "tilde y=y_alt only LRT_hat", "indirect_LRT_hat"]
        ] = (
            list(report_attacks)
            if report_attacks is not None
            else ["complete_LRT_hat", "indirect_LRT_hat"]
        )
        report_attack_list = list(dict.fromkeys(report_attack_list))
        if len(report_attack_list) == 0:
            raise ValueError("report_attacks must be non-empty.")
        primary_report_attack: Literal['complete_LRT_hat', 'tilde y=y_alt only LRT_hat', 'indirect_LRT_hat'] = report_attack_list[0]

        selection_attack_list: list[Literal["complete_LRT_hat", "indirect_LRT_hat"]] = []
        for attack_name in report_attack_list:
            if attack_name not in ("complete_LRT_hat", "indirect_LRT_hat"):
                raise ValueError(
                    "Per-attack model selection currently supports only "
                    "'complete_LRT_hat' and 'indirect_LRT_hat'."
                )
            selection_attack_list.append(cast(Literal["complete_LRT_hat", "indirect_LRT_hat"], attack_name))

        per_attack_model_best: dict[str, dict[str, dict[str, Any]]] = {}
        for selection_attack in selection_attack_list:
            cached_best: dict[str, dict[str, Any]] | None = (
                prefit_model_best_by_attack.get(str(selection_attack))
                if prefit_model_best_by_attack is not None
                else None
            )
            if cached_best is not None:
                per_attack_model_best[str(selection_attack)] = cached_best
                continue

            score_fn: Callable[..., float] = self._select_score_fn_for_eta_model(
                X_val=X_val,
                y_val=y_val,
                selection=selection,
                rng_seed_for_val=int(rng.integers(1 << 31)),
                attack_for_selection=selection_attack,
                y_alt=y_alt,
                y_null=y_null,
            )

            per_model_best: dict[str, dict[str, Any]] = {}
            for cfg in cfgs:
                # best = {"score": s,"params": dict(params),"model": model}
                # 各モデルごとに fit + select （val上で score_fn が最大となるハイパーパラメータを探索）
                best: dict[str, Any] = fit_and_select_eta_model(
                    cfg=cfg,
                    X_train=X_train,
                    y_train=y_train,
                    X_val=X_val,
                    y_val=y_val,
                    seed=int(rng.integers(1 << 31)),
                    score_fn=score_fn,
                )
                per_model_best[cfg.name] = best
            per_attack_model_best[str(selection_attack)] = per_model_best

        # --- report: 各モデルの best を使って test評価（attack ごとに tau,q を作る） ---
        report = {}
        for cfg in cfgs:
            name: str = cfg.name
            tests_by_attack: dict[str, dict[str, Any]] = {}
            tau_q_by_attack: dict[str, dict[str, Any]] = {}
            best_by_attack: dict[str, dict[str, Any]] = {}
            for attack_name in report_attack_list:
                best: dict[str, Any] = per_attack_model_best[str(attack_name)][name]
                model: Any = best["model"]
                best_by_attack[str(attack_name)] = {
                    "best_val_score": float(best["score"]),
                    "params": best["params"],
                }

                if attack_name == "indirect_LRT_hat":
                    X_null = self._take_feature_rows(X_threshold, y_threshold == y_null)
                    scores_null: np.ndarray = self._eta_to_x_lr_prob(
                        predict_eta(model, X_null)
                    )
                elif attack_name == "complete_LRT_hat":
                    rng_tau: np.random.Generator = np.random.default_rng(int(rng.integers(1 << 31)))
                    X_null = self._take_feature_rows(X_threshold, y_threshold == y_null)
                    eta_hat_null: np.ndarray = predict_eta(model, X_null)
                    ytilde_null: np.ndarray = grr_sample_binary(
                        y=y_null,
                        n=int(np.sum(y_threshold == y_null)),
                        epsilon=float(self.epsilon),
                        rng=rng_tau,
                    )
                    scores_null = expit(
                        attack_lrt_scores(ytilde_null, eta_hat_null, float(self.epsilon))
                        + self._log_eta_prior_correction()
                    )
                else:
                    scores_null = predict_eta(model, self._take_feature_rows(X_threshold, y_threshold == y_null))

                rng_for_attack: np.random.Generator = np.random.default_rng(rng_test.integers(1 << 32))
                if tau_selection == "cN":
                    tau, q = dp_sniper_threshold_from_scores(scores_null, c=self.c)
                    tau = float(np.clip(tau, 1e-15, 1.0 - 1e-15))
                    q = float(np.clip(q, 0.0, 1.0))
                    tau_q_by_attack[str(attack_name)] = {
                        "tau": tau,
                        "q": q,
                        "tau_selection": tau_selection,
                    }

                    res_test: dict = self.evaluate_eps_with_dp_sniper_cp_hat(
                        tau=tau,
                        q=q,
                        rng=rng_for_attack,
                        eta_model=model,
                        attack=attack_name,
                        y_alt=y_alt,
                        y_null=y_null,
                        sample_source="final",
                    )
                else:
                    res_test = self.evaluate_eps_with_theory_grid_cp_hat(
                        tau_grid_scores=scores_null,
                        rng=rng_for_attack,
                        eta_model=model,
                        attack=attack_name,
                        y_alt=y_alt,
                        y_null=y_null,
                        sample_source="final",
                    )
                    tau_q_by_attack[str(attack_name)] = {
                        "tau": float(res_test["tau"]),
                        "q": float(res_test["q"]),
                        "tau_selection": tau_selection,
                        "tau_grid_size": int(res_test["tau_grid_size"]),
                        "tau_grid_feasible_size": int(res_test["tau_grid_feasible_size"]),
                        "tau_cp_one_sided_alpha": float(res_test["tau_cp_one_sided_alpha"]),
                        "tau_selected_index": int(res_test["tau_selected_index"]),
                    }
                tests_by_attack[str(attack_name)] = res_test

            report[name] = {
                "selected_by": selection,
                "tau_selection": tau_selection,
                "best_val_score": float(best_by_attack[str(primary_report_attack)]["best_val_score"]),
                "params": best_by_attack[str(primary_report_attack)]["params"],
                "best_by_attack": best_by_attack,
                "tau": tau_q_by_attack[str(primary_report_attack)]["tau"],
                "q": tau_q_by_attack[str(primary_report_attack)]["q"],
                "tau_q_attack": primary_report_attack,
                "tau_q_by_attack": tau_q_by_attack,
                "test": tests_by_attack[str(primary_report_attack)],
                "tests_by_attack": tests_by_attack,
                "report_attacks": [str(a) for a in report_attack_list],
            }

        prior_0, prior_1 = self.eta_class_prior if self.eta_class_prior is not None else (0.5, 0.5)
        return {
            "selection": selection,
            "tau_selection": tau_selection,
            "report_attacks": [str(a) for a in report_attack_list],
            "model_best_by_attack": per_attack_model_best,
            "eta_class_prior": {
                "prior_0": float(prior_0),
                "prior_1": float(prior_1),
                "log_prior_0_over_prior_1": self._log_eta_prior_correction(),
            },
            "results": report,
        }


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
        X: np.ndarray = sample_x_given_y_truncated(n=n, spec=self.spec, y=y_input, rng=rng, B=self.B)
        # 追加：L2 ボールへ投影
        X = self.project_l2_ball(X, B=self.B)
        # eta(x)=P(Y=1|X=x)
        eta: np.ndarray = posterior_probs_from_x(X, self.spec)[:, 1]
        ytilde: np.ndarray = grr_sample_binary(y=y_input, n=n, epsilon=float(self.epsilon), rng=rng)
        # llr: log-likelihood ratio
        # sigmoid による単調変換をすることで tau と q が[0,1]に収まる
        # これは，S={z: logLambda(z) >= tau} と S_t:={z: sigmoid(logLambda(z))=p(z) >= sigmoid(tau)} の受理集合が一致しているから
        llr: np.ndarray = attack_lrt_scores(ytilde, eta, float(self.epsilon))
        p: np.ndarray = 1.0 / (1.0 + np.exp(-llr))
        return p
    
    ######## tilde y=k only LRT 監査用の関数 #######
    def _sample_score_x(self, y_input: int, n: int, rng: np.random.Generator) -> np.ndarray:
        """
        入力 y_input を固定して, X を n 個サンプルし，
        score_x = logit(eta(x)) = log(eta/(1-eta)) を返す（= log r(x) と単調同値）
        """
        if self.spec is None:
            raise ValueError("spec must be set.")
        X: np.ndarray = sample_x_given_y_truncated(n=n, spec=self.spec, y=y_input, rng=rng, B=self.B)
        X = self.project_l2_ball(X, B=self.B)

        eta: np.ndarray = posterior_probs_from_x(X, self.spec)[:, 1]
        eta = np.clip(eta.astype(float), 1e-15, 1.0 - 1e-15)

        score_x: np.ndarray = np.log(eta) - np.log(1.0 - eta)   # logit(eta)
        return score_x

    def _class_mean_from_spec(self, y: int) -> np.ndarray:
        if self.spec is None:
            raise ValueError("spec must be set.")
        mu: np.ndarray = np.zeros(self.spec.d, dtype=np.float64)
        mu[int(y) % self.spec.d] = float(self.spec.mean_shift)
        return mu

    def _x_lr_affine_params(self, y_plus: int = 1, y_minus: int = 0) -> tuple[np.ndarray, float]:
        """
        Return w, beta for log p(x|Y=y_plus) / p(x|Y=y_minus) = w^T x + beta.
        """
        return gaussian_x_lr_affine_params(
            self.spec,
            y_plus=y_plus,
            y_minus=y_minus,
        )

    def true_score_tail_lipschitz_bound(
        self,
        *,
        y_plus: int = 1,
        y_minus: int = 0,
        rho_lower: float | None = None,
    ) -> float:
        """
        Bound the density of the true Gaussian log-LR score under either class.

        This is a Lipschitz constant for the tail probabilities a(t), b(t), not
        for log a(t), log b(t).  With truncation to the common L2 ball, the
        acceptance probability is lower-bounded by 1 - rmax_alpha.
        """
        if self.spec is None:
            raise ValueError("spec must be set.")
        w, _ = self._x_lr_affine_params(y_plus=y_plus, y_minus=y_minus)
        tau: float = float(self.spec.sigma) * float(np.linalg.norm(w))
        if tau <= 0.0 or not np.isfinite(tau):
            raise ValueError(
                "The true LR score is degenerate. Check that the class means differ."
            )
        rho: float = float(1.0 - self.rmax_alpha if rho_lower is None else rho_lower)
        rho = max(rho, 1e-300)
        return float(1.0 / (rho * tau * np.sqrt(2.0 * np.pi)))

    def true_score_log_tail_lipschitz_bound(
        self,
        *,
        c_floor: float,
        y_plus: int = 1,
        y_minus: int = 0,
        rho_lower: float | None = None,
    ) -> float:
        """
        Conservative Lipschitz bound for log a(t), log b(t) on a region where
        both tail probabilities are at least c_floor.
        """
        c_safe: float = max(float(c_floor), 1e-300)
        return self.true_score_tail_lipschitz_bound(
            y_plus=y_plus,
            y_minus=y_minus,
            rho_lower=rho_lower,
        ) / c_safe

    def _directional_score_range_on_ball(self, y_plus: int, y_minus: int) -> tuple[float, float]:
        w, beta = self._x_lr_affine_params(y_plus=y_plus, y_minus=y_minus)
        radius: float = float(np.linalg.norm(w)) * float(self.B)
        return float(beta - radius), float(beta + radius)

    @staticmethod
    def _uniform_threshold_grid(
        *,
        low: float,
        high: float,
        width: float,
        max_grid_size: int | None = None,
    ) -> np.ndarray:
        if not np.isfinite(width) or width <= 0.0:
            raise ValueError(f"grid width must be positive and finite, got {width}.")
        lo: float = float(min(low, high))
        hi: float = float(max(low, high))
        n_steps: int = int(np.ceil(max(hi - lo, 0.0) / float(width)))
        grid_size: int = n_steps + 1
        if max_grid_size is not None and grid_size > int(max_grid_size):
            raise ValueError(
                f"The theory grid would contain {grid_size} thresholds, exceeding "
                f"max_grid_size={max_grid_size}. Increase --max_grid_size or use a "
                "smaller n/beta setting."
            )
        grid: np.ndarray = lo + float(width) * np.arange(grid_size, dtype=np.float64)
        if grid.size == 0:
            return np.array([lo], dtype=np.float64)
        grid[-1] = min(float(grid[-1]), hi)
        return np.unique(grid)

    def _sample_directional_x_scores(
        self,
        *,
        y_input: int,
        y_plus: int,
        y_minus: int,
        n: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        base_score: np.ndarray = self._sample_score_x(y_input, n, rng)
        if (y_plus, y_minus) == (1, 0):
            return base_score
        if (y_plus, y_minus) == (0, 1):
            return -base_score
        raise ValueError("finite-grid decomp audit currently supports binary directions only.")

    def _evaluate_decomp_direction_finite_grid_cp(
        self,
        *,
        y_plus: int,
        y_minus: int,
        grid: np.ndarray,
        c_n: float,
        alpha_cp: float,
        rng: np.random.Generator,
        n_plus: int,
        n_minus: int,
    ) -> dict[str, Any]:
        s_plus: np.ndarray = self._sample_directional_x_scores(
            y_input=y_plus,
            y_plus=y_plus,
            y_minus=y_minus,
            n=n_plus,
            rng=rng,
        )
        s_minus: np.ndarray = self._sample_directional_x_scores(
            y_input=y_minus,
            y_plus=y_plus,
            y_minus=y_minus,
            n=n_minus,
            rng=rng,
        )

        k_plus: np.ndarray = self._counts_above_thresholds(s_plus, grid)
        k_minus: np.ndarray = self._counts_above_thresholds(s_minus, grid)
        a_hat: np.ndarray = k_plus / float(n_plus)
        b_hat: np.ndarray = k_minus / float(n_minus)

        a_lo, a_hi = self._beta_cp_interval(
            k_plus,
            n_plus,
            one_sided_alpha=alpha_cp,
        )
        b_lo, b_hi = self._beta_cp_interval(
            k_minus,
            n_minus,
            one_sided_alpha=alpha_cp,
        )

        eps_lower: np.ndarray = np.full(grid.shape, -np.inf, dtype=np.float64)
        feasible: np.ndarray = b_hat >= float(c_n)
        valid_lower: np.ndarray = feasible & (a_lo > 0.0) & (b_hi > 0.0)
        eps_lower[valid_lower] = np.log(a_lo[valid_lower] / b_hi[valid_lower])

        eps_emp: np.ndarray = np.full(grid.shape, -np.inf, dtype=np.float64)
        valid_emp: np.ndarray = (a_hat > 0.0) & (b_hat > 0.0)
        eps_emp[valid_emp] = np.log(a_hat[valid_emp] / b_hat[valid_emp])

        eps_upper: np.ndarray = np.full(grid.shape, np.inf, dtype=np.float64)
        valid_upper: np.ndarray = (a_hi > 0.0) & (b_lo > 0.0)
        eps_upper[valid_upper] = np.log(a_hi[valid_upper] / b_lo[valid_upper])

        if np.any(np.isfinite(eps_lower)):
            best_idx: int = int(np.nanargmax(eps_lower))
        else:
            best_idx = 0

        return {
            "direction": f"{y_plus}{y_minus}",
            "y_plus": int(y_plus),
            "y_minus": int(y_minus),
            "tau": float(grid[best_idx]),
            "tau_selected_index": int(best_idx),
            "tau_grid_size": int(grid.size),
            "tau_grid_feasible_size": int(np.sum(feasible)),
            "TP": int(k_plus[best_idx]),
            "FP": int(k_minus[best_idx]),
            "N_plus_eval": int(n_plus),
            "N_minus_eval": int(n_minus),
            "a_hat": float(a_hat[best_idx]),
            "b_hat": float(b_hat[best_idx]),
            "a_ci": (float(a_lo[best_idx]), float(a_hi[best_idx])),
            "b_ci": (float(b_lo[best_idx]), float(b_hi[best_idx])),
            "eps_emp_x": float(eps_emp[best_idx]),
            "eps_lower_x": float(eps_lower[best_idx]),
            "eps_upper_x": float(eps_upper[best_idx]),
            "alpha_cp": float(alpha_cp),
        }

    def evaluate_decomp_finite_grid_cp(
        self,
        *,
        beta: float,
        failure_delta: float | None = None,
        lipschitz_bound: float | None = None,
        lipschitz_c_floor: float | None = None,
        n_plus_eval: int | None = None,
        n_minus_eval: int | None = None,
        max_grid_size: int | None = None,
    ) -> dict[str, Any]:
        """
        Algorithm 1 style finite-grid CP-LCB decomp audit with the true score.

        c_n = n_min^{-beta}.  If lipschitz_bound is not supplied, a conservative
        bound for log a/log b on the clipped region is computed as
        density_bound / lipschitz_c_floor, with lipschitz_c_floor defaulting to c_n.
        """
        if self.spec is None:
            raise ValueError("spec must be set for finite-grid decomp audit.")
        beta_float: float = float(beta)
        if not (0.0 < beta_float < 1.0):
            raise ValueError(f"beta must be in (0, 1), got {beta}.")

        delta_eff: float = float(self.alpha if failure_delta is None else failure_delta)
        if not (0.0 < delta_eff < 1.0):
            raise ValueError(f"failure_delta must be in (0, 1), got {delta_eff}.")

        n_plus: int = int(self.nb_trials if n_plus_eval is None else n_plus_eval)
        n_minus: int = int(self.nb_trials if n_minus_eval is None else n_minus_eval)
        n_min: int = min(n_plus, n_minus)
        if n_min <= 0:
            raise ValueError("nb_trials must be positive for finite-grid decomp audit.")

        c_n: float = float(n_min ** (-beta_float))
        c_for_lipschitz: float = float(c_n if lipschitz_c_floor is None else lipschitz_c_floor)
        L_log: float = float(
            lipschitz_bound
            if lipschitz_bound is not None
            else self.true_score_log_tail_lipschitz_bound(c_floor=c_for_lipschitz)
        )
        if not np.isfinite(L_log) or L_log <= 0.0:
            raise ValueError(f"lipschitz bound must be positive and finite, got {L_log}.")

        h_n: float = float((1.0 / (4.0 * L_log)) * n_min ** (-(1.0 - beta_float) / 2.0))
        directions: tuple[tuple[int, int], ...] = ((1, 0), (0, 1))

        grids: dict[str, np.ndarray] = {}
        total_cp_events: int = 0
        for y_plus, y_minus in directions:
            low, high = self._directional_score_range_on_ball(y_plus, y_minus)
            grid: np.ndarray[tuple[Any, ...], np.dtype[Any]] = self._uniform_threshold_grid(
                low=low,
                high=high,
                width=h_n,
                max_grid_size=max_grid_size,
            )
            grids[f"{y_plus}{y_minus}"] = grid
            total_cp_events += 2 * int(grid.size) + 2

        alpha_cp: float = float(delta_eff / max(total_cp_events, 1))
        rng: np.random.Generator = np.random.default_rng(self.random_state)
        direction_results: list[dict[str, Any]] = []
        for y_plus, y_minus in directions:
            direction_results.append(
                self._evaluate_decomp_direction_finite_grid_cp(
                    y_plus=y_plus,
                    y_minus=y_minus,
                    grid=grids[f"{y_plus}{y_minus}"],
                    c_n=c_n,
                    alpha_cp=alpha_cp,
                    rng=rng,
                    n_plus=n_plus,
                    n_minus=n_minus,
                )
            )

        best_direction: dict[str, Any] = max(direction_results, key=lambda r: r["eps_lower_x"])
        eps_rr: float = float(self.epsilon)
        eps_emp_total: float = float(best_direction["eps_emp_x"] + eps_rr)
        eps_lower_total: float = float(best_direction["eps_lower_x"] + eps_rr)
        eps_upper_total: float = float(best_direction["eps_upper_x"] + eps_rr)

        self.eps_emp = eps_emp_total
        self.eps_ci = (eps_lower_total, eps_upper_total)
        result: dict[str, Any] = {
            "eps_emp": eps_emp_total,
            "eps_ci": self.eps_ci,
            "eps_lower": eps_lower_total,
            "eps_upper": eps_upper_total,
            "eps_emp_x": float(best_direction["eps_emp_x"]),
            "eps_lower_x": float(best_direction["eps_lower_x"]),
            "eps_upper_x": float(best_direction["eps_upper_x"]),
            "eps_rr": eps_rr,
            "beta": beta_float,
            "c_n": c_n,
            "h_n": h_n,
            "L_log": L_log,
            "L_tail": self.true_score_tail_lipschitz_bound(),
            "lipschitz_c_floor": c_for_lipschitz,
            "failure_delta": delta_eff,
            "alpha_cp": alpha_cp,
            "total_cp_events": int(total_cp_events),
            "n_min": int(n_min),
            "N_plus_eval": int(n_plus),
            "N_minus_eval": int(n_minus),
            "best_direction": str(best_direction["direction"]),
            "tau": float(best_direction["tau"]),
            "tau_grid_size": int(best_direction["tau_grid_size"]),
            "tau_grid_feasible_size": int(best_direction["tau_grid_feasible_size"]),
            "direction_results": direction_results,
        }
        self.last_decomp_grid_result = result

        print("=== finite-grid decomp CP-LCB DEBUG ===")
        print(f"eps={self.epsilon}, seed={self.random_state}, beta={beta_float}, n_min={n_min}")
        print(f"c_n={c_n:.6g}, L_log={L_log:.6g}, h_n={h_n:.6g}, alpha_cp={alpha_cp:.6g}")
        print(
            "grid_sizes="
            + ", ".join(f"{key}:{grid.size}" for key, grid in grids.items())
        )
        print(f"best_direction={best_direction['direction']}, tau={best_direction['tau']:.6g}")
        print(f"eps_lower_x={best_direction['eps_lower_x']:.6g}, eps_lower_total={eps_lower_total:.6g}")
        print("=== /finite-grid decomp CP-LCB DEBUG ===")
        return result

    def evaluate_eps_with_dp_sniper_cp(
        self,
        tau: float,
        q: float,
        rng: np.random.Generator,
        attack: Literal["complete_LRT", "tilde y=y_alt only LRT", "indirect_LRT"],
        y_alt: int = 1,
        y_null: int = 0,
    ) -> dict:
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
            tau_eff: float = max(float(tau), 1e-15)
            gt: np.ndarray = p > tau_eff + atol
            eq: np.ndarray = np.abs(p - tau_eff) <= atol
            out: np.ndarray = np.zeros(p.shape[0], dtype=np.int64)
            out[gt] = 1
            if np.any(eq):
                out[eq] = (rng.random(np.sum(eq)) < q).astype(np.int64)
            return int(out.sum())

        # TPR = P(attack=1 | Y=y_alt), FPR = P(attack=1 | Y=y_null)
        if attack == "complete_LRT":
            ######## 完全 LRT 監査 ########
            p_alt: np.ndarray = self._sample_p_theta(y_alt, N_final, rng)
            p_null: np.ndarray = self._sample_p_theta(y_null, N_final, rng)
        elif attack == "tilde y=y_alt only LRT":
            ######## tilde y=y_alt only LRT 監査 ########
            # 1) X のスコア
            score_alt: np.ndarray = self._sample_score_x(y_alt, N_final, rng)
            score_null: np.ndarray = self._sample_score_x(y_null, N_final, rng)

            # 2) tilde y を生成（入力 y を固定してGRR）
            ytilde_alt: np.ndarray = grr_sample_binary(
                y=y_alt, n=N_final, epsilon=float(self.epsilon), rng=rng
            )
            ytilde_null: np.ndarray = grr_sample_binary(
                y=y_null, n=N_final, epsilon=float(self.epsilon), rng=rng
            )

            # 3) 「受理条件」: tilde y == y_alt のときだけスコアで判定、それ以外は必ず0
            #    dp_sniper_threshold_from_scores が「大きいほど1」前提なので score を logistic にして渡す
            p_alt = np.zeros(N_final, dtype=np.float64)
            p_null = np.zeros(N_final, dtype=np.float64)

            m_alt: np.ndarray = ytilde_alt == y_alt
            m_null: np.ndarray = ytilde_null == y_alt

            # score -> pseudo-prob (単調変換ならOK): sigmoid(score)
            p_alt[m_alt] = 1.0 / (1.0 + np.exp(-score_alt[m_alt]))
            p_null[m_null] = 1.0 / (1.0 + np.exp(-score_null[m_null]))
        elif attack == "indirect_LRT":
            ######## 間接 LRT 監査 ########
            # 1) X のスコア: logit(eta(x))
            p_alt: np.ndarray = self._sample_score_x(y_alt, N_final, rng)
            p_null: np.ndarray = self._sample_score_x(y_null, N_final, rng)

            p_alt = 1.0 / (1.0 + np.exp(-p_alt))  # sigmoid(score)
            p_null = 1.0 / (1.0 + np.exp(-p_null))  # sigmoid(score)
        else:
            raise ValueError("Unknown attack type for LRT-CF evaluation.")


        TP: int = _sample_attack_outputs(p_alt)
        FP: int = _sample_attack_outputs(p_null)

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
        rng_val: np.random.Generator = np.random.default_rng(rng.integers(1 << 32))
        rng_test: np.random.Generator = np.random.default_rng(rng.integers(1 << 32))

        # ------------- 検証（検証用データで最適な tau を決める） --------------
        # N_sniper 回ぶんの (x_i, ytilde_i) を作る
        N_sniper: int = self.nb_trials

        # ----- direction01 : y=0 側のスコアから tau を決める（TPR=Pr[S=1|Y=1], FPR=Pr[S=1|Y=0] の FPR を下駄履き安定化） -----
        scores_null0: np.ndarray = self._sample_p_theta(0, N_sniper, rng_val)
        tau_null0, q_null0 = dp_sniper_threshold_from_scores(scores_null0, c=self.c)

        # ----- direction10 : y=1 側のスコアから tau を決める（TPR=Pr[S=1|Y=0], FPR=Pr[S=1|Y=1] の FPR を下駄履き安定化） -----
        scores_null1: np.ndarray = self._sample_p_theta(1, N_sniper, rng_val)
        tau_null1, q_null1 = dp_sniper_threshold_from_scores(scores_null1, c=self.c)

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
        # direction01 も direction10 も試して、より厳しい方を採用する
        results10: dict = self.evaluate_eps_with_dp_sniper_cp(
            tau=tau_null0, q=q_null0, rng=rng_test, attack="complete_LRT", y_alt=1, y_null=0)
        results01: dict = self.evaluate_eps_with_dp_sniper_cp(
            tau=tau_null1, q=q_null1, rng=rng_test, attack="complete_LRT", y_alt=0, y_null=1)
        results: dict = results01 if results01["eps_emp"] > results10["eps_emp"] else results10
        self.eps_emp = results['eps_emp']
        eps_lo, eps_hi = results['eps_ci']
        self.eps_ci = (float(eps_lo), float(eps_hi))

        # ------------- DEBUG 出力 -------------
        print("=== LRT DEBUG ===")
        print(f"eps={self.epsilon}, seed={self.random_state}, c={self.c}")
        print(f"tau01={tau_null1:.6g}, q01={q_null1:.6g}")
        print(f"tau10={tau_null0:.6g}, q10={q_null0:.6g}")
        print(f"tpr={results['tpr_hat']:.6g}, fpr={results['fpr_hat']:.6g}")
        print(f"eps_emp(computed)={np.log(max(results['tpr_hat'],1e-12)/max(results['fpr_hat'],1e-12)):.6g}")
        print(f"eps_lo={eps_lo:.6g}, eps_hi={eps_hi:.6g}")
        print("=== /LRT DEBUG ===")
        return self.eps_emp


    ############# tilde y=y_alt only LRT 監査用の関数 #############
    # メンテナンスしていない - y_alt になっていない部分多数
    def _run_lrt_tilde_y_alt_once(self) -> float: 
        """
        tilde y=y_alt のみ受理する攻撃者：
        - 閾値は X-only score (=logit eta) の Y=0 側分布から dp-sniper で決める
        - a = log(TPR_x/FPR_x) を CP で推定し、最後に + epsilon_GRR する
        """
        rng: np.random.Generator = np.random.default_rng(self.random_state)
        rng_val: np.random.Generator = np.random.default_rng(rng.integers(1 << 32))
        rng_test: np.random.Generator = np.random.default_rng(rng.integers(1 << 32))

        N_sniper: int = self.nb_trials

        # Y=0 側の score_x から閾値
        scores_val: np.ndarray = self._sample_score_x(0, N_sniper, rng_val)
        p_val = 1.0 / (1.0 + np.exp(-scores_val))          # sigmoid
        # sigmoid(logit(eta)) = eta に注意
        tau, q = dp_sniper_threshold_from_scores(p_val, c=self.c)

        # 数値安全：tau が 0 に落ちる事故を避ける
        tau = float(np.clip(tau, 1e-15, 1.0 - 1e-15))
        q = float(np.clip(q, 0.0, 1.0))

        # a を評価 （X-only）
        res_a: dict = self.evaluate_eps_with_dp_sniper_cp(tau=tau, q=q, rng=rng_test, attack="indirect_LRT")
        a_emp: float = float(res_a["eps_emp"])
        a_lo, a_hi = res_a["eps_ci"]

        # tilde y=1 受理の“合成”に相当する分だけ +epsilon_GRR
        eps_grr: float = float(self.epsilon)  # GRR の ε は既知

        self.eps_emp = a_emp
        self.eps_ci  = (float(a_lo), float(a_hi))

        print("=== LRT_tilde1 DEBUG ===")
        print(f"eps={self.epsilon}, seed={self.random_state}, c={self.c}")
        print(f"tau={tau:.6g}, q={q:.6g}")
        print(f"a_emp={a_emp:.6g}, a_lo={a_lo:.6g}, a_hi={a_hi:.6g}")
        print(f"eps_emp=a+eps_grr={self.eps_emp:.6g}")
        print(f"eps_ci=({self.eps_ci[0]:.6g}, {self.eps_ci[1]:.6g})")
        print("=== /LRT_tilde1 DEBUG ===")

        return float(self.eps_emp)

    ############# decomp LRT 監査用の関数 #############
    def _run_lrt_indirect_once(self) -> float:
        """
        攻撃者：
        - 閾値は X-only score (=logit eta) の Y=0 側分布から dp-sniper で決める
        - a = log(TPR_x/FPR_x) を CP で推定し、最後に + epsilon_GRR する
        """
        rng: np.random.Generator = np.random.default_rng(self.random_state)
        rng_val: np.random.Generator = np.random.default_rng(rng.integers(1 << 32))
        rng_test: np.random.Generator = np.random.default_rng(rng.integers(1 << 32))

        N_sniper: int = self.nb_trials

        # direction10 : y=0 側のスコアから tau を決める（TPR=Pr[S=1|Y=1], FPR=Pr[S=1|Y=0] の FPR を下駄履き安定化）
        scores_null0: np.ndarray = self._sample_score_x(0, N_sniper, rng_val)
        p_null0 = 1.0 / (1.0 + np.exp(-scores_null0))          # sigmoid
        tau_null0, q_null0 = dp_sniper_threshold_from_scores(p_null0, c=self.c)

        # a を評価 （X-only）
        res_a10: dict = self.evaluate_eps_with_dp_sniper_cp(
            tau=tau_null0, q=q_null0, rng=rng_test, attack="indirect_LRT", y_alt=1, y_null=0
        )
        a_emp10 : float = float(res_a10["eps_emp"])
        a_lo10, a_hi10 = res_a10["eps_ci"]

        # direction01 : y=1 側のスコアから tau を決める（TPR=Pr[S=1|Y=0], FPR=Pr[S=1|Y=1] の FPR を下駄履き安定化）
        scores_null1: np.ndarray = self._sample_score_x(1, N_sniper, rng_val)
        p_null1 = 1.0 / (1.0 + np.exp(-scores_null1))          # sigmoid
        tau_null1, q_null1 = dp_sniper_threshold_from_scores(p_null1, c=self.c)

        # a を評価 （X-only）
        res_a01: dict = self.evaluate_eps_with_dp_sniper_cp(
            tau=tau_null1, q=q_null1, rng=rng_test, attack="indirect_LRT", y_alt=0, y_null=1
        )
        a_emp01 : float = float(res_a01["eps_emp"])
        a_lo01, a_hi01 = res_a01["eps_ci"]

        # tilde y=y_alt 受理の“合成”に相当する分だけ +epsilon_GRR
        eps_grr: float = float(self.epsilon)  # GRR の ε は既知

        self.eps_emp = max(a_emp01, a_emp10) + eps_grr
        self.eps_ci = (float(max(a_lo01, a_lo10) + eps_grr), float(max(a_hi01, a_hi10) + eps_grr))

        print("=== decomp LRT DEBUG ===")
        print(f"eps={self.epsilon}, seed={self.random_state}, c={self.c}")
        print(f"tau_null0={tau_null0:.6g}, q_null0q={q_null0:.6g}")
        print(f"a_lo10={a_lo10:.6g}, a_hi10={a_hi10:.6g}")
        print(f"tau_null1={tau_null1:.6g}, q_null1={q_null1:.6g}")
        print(f"a_lo01={a_lo01:.6g}, a_hi01={a_hi01:.6g}")
        print(f"eps_emp=a+eps_grr={self.eps_emp:.6g}")
        print(f"eps_ci=({self.eps_ci[0]:.6g}, {self.eps_ci[1]:.6g})")
        print("=== /decomp LRT DEBUG ===")

        return float(self.eps_emp)


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
        SS_Client = _get_ss_client()

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
        UE_Client = _get_ue_client()

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
        UE_Client = _get_ue_client()

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
        HE_Client = _get_he_client()

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
        HE_Client = _get_he_client()

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
        LH_Client = _get_lh_client()

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
        LH_Client = _get_lh_client()

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
        AGRR_Client, _, _, _, _ = _get_approximate_ldp_symbols()

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
        _, _, ASUE_Client, _, _ = _get_approximate_ldp_symbols()

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
        _, _, _, GM_Client, find_scale = _get_approximate_ldp_symbols()

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
        _, _, _, GM_Client, _ = _get_approximate_ldp_symbols()

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
        _, ALH_Client, _, _, _ = _get_approximate_ldp_symbols()

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
        _, ALH_Client, _, _, _ = _get_approximate_ldp_symbols()

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
        UEClient = _get_pure_ldp_ue_client()

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
        UEClient = _get_pure_ldp_ue_client()

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
        elif protocol_name == "LRT_tilde1":
            # tilde y=1 only LRT も Ray 不要。確実に止める
            try:
                ray.shutdown()
            except Exception:
                pass
            self.nb_cores = 1
            return self._run_lrt_tilde_y_alt_once()       
        elif protocol_name in ("LRT_decomp", "LRT_indirect"):
            # decomp LRT も Ray 不要。確実に止める
            try:
                ray.shutdown()
            except Exception:
                pass
            self.nb_cores = 1
            return self._run_lrt_indirect_once()
        
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
# approx_ldp_protocols = ['AGRR', 'ASUE', 'AGM', 'GM', 'ABLH', 'pAOLH']
# auditor_approx_ldp = LDP_Auditor(nb_trials=nb_trials, alpha=alpha, epsilon=epsilon, delta=delta, k=k, random_state=seed, n_jobs=-1)

# for protocol in approx_ldp_protocols:
#     eps_emp = auditor_approx_ldp.run_audit(protocol)
#     print("{} eps_emp:".format(protocol), eps_emp)

# if __name__ == "__main__":
#     main()
