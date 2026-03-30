# General imports
import warnings
from collections import defaultdict
from typing import Any, Literal, Self, Callable, Sequence, cast

from matplotlib.pylab import Generator
from sklearn.pipeline import Pipeline
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

from ldp_audit.eta_models import EtaModelConfig

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
from .eta_models import (
    get_default_eta_model_configs,
    fit_and_select_eta_model,
    predict_eta,
)
from .simulation import (
    MixtureSpec,
    log_Rmax_gaussian,
    posterior_probs_from_x,
    sample_x_given_y_truncated,
    set_B_from_quantile,
    sample_attack_trainset_from_spec,
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
        dynamic_nb_trials: bool = True,
        sim_hat: bool = True,
        real_val_X: np.ndarray | None = None,
        real_val_y: np.ndarray | None = None,
        real_final_X: np.ndarray | None = None,
        real_final_y: np.ndarray | None = None,
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
        self.real_val_X: np.ndarray | None = real_val_X
        self.real_val_y: np.ndarray | None = real_val_y
        self.real_final_X: np.ndarray | None = real_final_X
        self.real_final_y: np.ndarray | None = real_final_y

        if not self.sim_hat:
            if (
                self.real_val_X is None
                or self.real_val_y is None
                or self.real_final_X is None
                or self.real_final_y is None
            ):
                raise ValueError(
                    "When sim_hat=False, real_val_X/real_val_y/real_final_X/real_final_y must be provided."
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
            "LRT_indirect": "SPECIAL",  # indirect
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

        count = 0
        for _ in range(trials):            
            count += GRR_Client(v, k, epsilon) == test_statistic
        return count
    
    ############# LRT 監査用の関数群 #############

    ##### eta 学習モデルをつかう #####
    def _sample_x_given_y_for_eta_hat(
        self,
        *,
        y_input: int,
        n: int,
        rng: np.random.Generator,
        sample_source: Literal["val", "final"] = "final",
    ) -> np.ndarray:
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

        X_pool: np.ndarray | None = self.real_val_X if sample_source == "val" else self.real_final_X
        y_pool: np.ndarray | None = self.real_val_y if sample_source == "val" else self.real_final_y
        assert X_pool is not None and y_pool is not None

        idx_all: np.ndarray = np.flatnonzero(y_pool == y_input)
        if idx_all.size == 0:
            raise ValueError(
                f"No real-data samples with y={y_input} were found in {sample_source} pool."
            )
        chosen: np.ndarray = rng.choice(idx_all, size=int(n), replace=True)
        return np.asarray(X_pool[chosen], dtype=float)

    def _sample_p_theta_hat(
        self,
        y_input: int,
        n: int,
        rng: np.random.Generator,
        eta_model,
        sample_source: Literal["val", "final"] = "final",
    ) -> np.ndarray:
        """
        完全LRT用：eta(x) を学習モデルで推定する版
        Pr[Y=1 | x, \tilde y] スコアの ndarray を返す
        """
        X: np.ndarray = self._sample_x_given_y_for_eta_hat(
            y_input=y_input,
            n=n,
            rng=rng,
            sample_source=sample_source,
        )

        eta_hat: np.ndarray = predict_eta(eta_model, X)
        ytilde: np.ndarray = grr_sample_binary(y=y_input, n=n, epsilon=float(self.epsilon), rng=rng)

        # LRT の対数スコア Λ(x,\tilde y) を計算
        llr: np.ndarray = attack_lrt_scores(ytilde, eta_hat, float(self.epsilon))
        # sigmoid = つまり Pr[Y=1|x,\tilde y]
        p = 1.0 / (1.0 + np.exp(-llr))
        return p


    def _sample_score_x_hat(
        self,
        y_input: int,
        n: int,
        rng: np.random.Generator,
        eta_model,
        sample_source: Literal["val", "final"] = "final",
    ) -> np.ndarray:
        """
        X-only（Decomposition）用：score_x = logit(eta_hat(x))
        """
        X: np.ndarray = self._sample_x_given_y_for_eta_hat(
            y_input=y_input,
            n=n,
            rng=rng,
            sample_source=sample_source,
        )

        eta_hat: np.ndarray = predict_eta(eta_model, X)
        score_x = np.log(eta_hat) - np.log(1.0 - eta_hat)
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
        sample_source: Literal["val", "final"] = "final",
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
                y_pool: np.ndarray | None = self.real_val_y if sample_source == "val" else self.real_final_y
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

            p1[m1] = 1.0 / (1.0 + np.exp(-score1[m1]))
            p0[m0] = 1.0 / (1.0 + np.exp(-score0[m0]))

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
            p1 = 1.0 / (1.0 + np.exp(-score1))
            p0 = 1.0 / (1.0 + np.exp(-score0))

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
    
    def _select_score_fn_for_eta_model(
        self,
        *,
        X_val: np.ndarray,
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
        rng_local: Generator = np.random.default_rng(rng_seed_for_val)

        def score_fn(model) -> float:
            # attack_for_selection に応じて null 側スコアから tau,q を作る
            if attack_for_selection == "indirect_LRT_hat":
                scores_null: np.ndarray = predict_eta(model, X_val[y_val == y_null])
            elif attack_for_selection == "complete_LRT_hat":
                rng_tau: Generator = np.random.default_rng(rng_seed_for_val + 12345)
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
        report_attacks: Sequence[
            Literal["complete_LRT_hat", "tilde y=y_alt only LRT_hat", "indirect_LRT_hat"]
        ] | None = None,
        eta_model_cfgs: Sequence[EtaModelConfig] | None = None,
        # simulation用（実データなら None でOK）
        sim_n_train: int | None = 4000,
        sim_n_val: int | None = 2000,
        # 実データ用（simulationなら None でOK）
        X_train: np.ndarray | None = None,
        y_train: np.ndarray | None = None,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        y_alt: int = 1,
        y_null: int = 0,
    ) -> dict:
        """
        4モデルそれぞれで best hyperparam を選び、test評価（= evaluate）を返す。
        - simulation: sim_n_train/sim_n_val を指定し、specからデータ生成
        - real: X_train,y_train,X_val,y_val を渡す
        """
        if self.spec is None:
            raise ValueError("spec must be set.")

        rng: Generator = np.random.default_rng(seed)
        rng_train: Generator = np.random.default_rng(rng.integers(1 << 32))
        rng_val: Generator = np.random.default_rng(rng.integers(1 << 32))
        rng_test: Generator = np.random.default_rng(rng.integers(1 << 32))

        # --- train/val データを用意 （simulation data 用） ---
        if X_train is None:
            if sim_n_train is None or sim_n_val is None:
                raise ValueError("For simulation mode, set sim_n_train and sim_n_val.")
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
        else:
            # real mode
            assert y_train is not None and X_val is not None and y_val is not None

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
            tau_q_by_attack: dict[str, dict[str, float]] = {}
            best_by_attack: dict[str, dict[str, Any]] = {}
            for attack_name in report_attack_list:
                best: dict[str, Any] = per_attack_model_best[str(attack_name)][name]
                model: Pipeline = best["model"]
                best_by_attack[str(attack_name)] = {
                    "best_val_score": float(best["score"]),
                    "params": best["params"],
                }

                if attack_name == "indirect_LRT_hat":
                    scores_null: np.ndarray = predict_eta(model, X_val[y_val == y_null])
                elif attack_name == "complete_LRT_hat":
                    rng_tau: Generator = np.random.default_rng(int(rng.integers(1 << 31)))
                    scores_null = self._sample_p_theta_hat(
                        y_input=y_null,
                        n=int(np.sum(y_val == y_null)),
                        rng=rng_tau,
                        eta_model=model,
                        sample_source="val",
                    )
                else:
                    scores_null = predict_eta(model, X_val[y_val == y_null])

                tau, q = dp_sniper_threshold_from_scores(scores_null, c=self.c)
                tau = float(np.clip(tau, 1e-15, 1.0 - 1e-15))
                q = float(np.clip(q, 0.0, 1.0))
                tau_q_by_attack[str(attack_name)] = {"tau": tau, "q": q}

                rng_for_attack: Generator = np.random.default_rng(rng_test.integers(1 << 32))
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
                tests_by_attack[str(attack_name)] = res_test

            report[name] = {
                "selected_by": selection,
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

        return {
            "selection": selection,
            "report_attacks": [str(a) for a in report_attack_list],
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
        rng_val: Generator = np.random.default_rng(rng.integers(1 << 32))
        rng_test: Generator = np.random.default_rng(rng.integers(1 << 32))

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

    ############# indirect LRT 監査用の関数 #############
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

        print("=== indirect LRT DEBUG ===")
        print(f"eps={self.epsilon}, seed={self.random_state}, c={self.c}")
        print(f"tau_null0={tau_null0:.6g}, q_null0q={q_null0:.6g}")
        print(f"a_lo10={a_lo10:.6g}, a_hi10={a_hi10:.6g}")
        print(f"tau_null1={tau_null1:.6g}, q_null1={q_null1:.6g}")
        print(f"a_lo01={a_lo01:.6g}, a_hi01={a_hi01:.6g}")
        print(f"eps_emp=a+eps_grr={self.eps_emp:.6g}")
        print(f"eps_ci=({self.eps_ci[0]:.6g}, {self.eps_ci[1]:.6g})")
        print("=== /indirect LRT DEBUG ===")

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
        elif protocol_name == "LRT_tilde1":
            # tilde y=1 only LRT も Ray 不要。確実に止める
            try:
                ray.shutdown()
            except Exception:
                pass
            self.nb_cores = 1
            return self._run_lrt_tilde_y_alt_once()       
        elif protocol_name == "LRT_indirect":
            # indirect LRT も Ray 不要。確実に止める
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
