# general imports
import logging
import os
from typing import Iterable, List, Sequence, TypedDict

import pandas as pd
from tqdm import tqdm

# Our imports
from ldp_audit.base_auditor import LDPAuditor
from plot_functions import (
    plot_results_example_audit,
    plot_results_pure_ldp_protocols,
)
from ldp_audit.simulation import MixtureSpec, simulate_eta_split

# ---------------------------------------------------------------------
# Logging / IO setup
# ---------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
os.makedirs('results', exist_ok=True)

# ---------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------
class ResultsDict(TypedDict):
    """結果CSVへ書き出す各列の型"""

    seed: List[int]
    protocol: List[str]
    k: List[int]  # カテゴリ数（ラベルの種類数）
    delta: List[float]
    epsilon: List[float]
    eps_emp: List[float]  # 監査で得た経験的 ε（スカラー）
    logRmax_eff: List[float]


# ---------------------------------------------------------------------
# Main audit results
# ---------------------------------------------------------------------
def run_main_experiments(
    nb_trials: int,
    alpha: float,
    lst_protocols: Sequence[str],
    lst_seed: Iterable[int],
    lst_k: Sequence[int],
    lst_eps: Sequence[float],
    delta: float,
    analysis: str,
) -> pd.DataFrame:
    """
    乱数シード・カテゴリ数 k・ε（および δ）・プロトコルの組み合わせを受け取り
    LDP 監査を実行し，経験的な ε（eps_emp）を集計して CSV に保存し，
    同じ内容を pd.DataFrame として返す関数

    Parameters
    ----------
    nb_trials : int
        監査内部で用いる試行回数（モンテカルロ反復やサンプル生成回数など）
        大きいほど推定精度は上がるが計算コストが増加
    alpha : float
        有意水準（監査の統計的判定で使用）.例: 1e-2 (= 0.01)
    lst_protocols : Sequence[str]
        評価対象の LDP プロトコル名（例: 'GRR', 'OUE', 'BLH' など）の列
        Sequence は list, tuple, str, ndarray などの，抽象的な「順序付きコレクション」
        Iterable とは異なり，インデックスアクセスや len() が可能
    lst_seed : Iterable[int]
        乱数シードの集まり（range や list）.再現性・分散評価のために複数指定可能
        Iterable は, for で回せるオブジェクトのこと
        list, tuple, set, dict, str, range, generator などが該当
    lst_k : Sequence[int]
        カテゴリ数 k の候補（例: [25, 50, 100, ...]）。
    lst_eps : Sequence[float]
        プライバシー予算 ε の候補（例: [0.25, 0.5, 1, 2, ...]）。
    delta : float
        近似 DP の δ。純粋 DP の場合は 0.0。
    analysis : str
        出力 CSV のファイル名識別子（例: 'main_pure_ldp_protocols'）。

    Returns
    -------
    pd.DataFrame
        列: ['seed','protocol','k','delta','epsilon','eps_emp'] を持つ結果データフレーム。
        同内容を `results/ldp_audit_results_{analysis}.csv` にも保存。
    """
    # Initialize dictionary to save results
    results: dict[str, list] = {
        "seed": [],
        "protocol": [],
        "k": [],
        "delta": [],
        "epsilon": [],
        "eps_emp": [],
        "logRmax_eff": [],
    }

    # Initialize LDP-Auditor（初期値は後で set_params で上書きする）
    auditor: LDPAuditor = LDPAuditor(
        nb_trials=nb_trials,
        alpha=alpha,
        epsilon=lst_eps[0],
        delta=delta,
        k=lst_k[0],
        random_state=next(iter(lst_seed)) if hasattr(lst_seed, "__iter__") else 0,
        n_jobs=-1,
        rmax_alpha=0.01,
    )

    # Run the experiments
    for seed in lst_seed:    
        logging.info(f"Running experiments for seed: {seed}")
        # --- ここでシミュレーションから η を生成し、監査器へ渡す ---
        spec = MixtureSpec(num_classes=2, d=2, sigma=1.0, mean_shift=2.0)
        eta_val, eta_test = simulate_eta_split(
            N_total=200_000, spec=spec, val_ratio=0.8, seed=seed
        )
        auditor.set_params(spec=spec, eta_val=eta_val, eta_test=eta_test)
        # logRmax_seed: float = estimate_log_Rmax_from_eta(eta_val, clip=1e-6, q=0.99)
        for k in lst_k:
            for epsilon in lst_eps:
                
                # Update the auditor parameters
                auditor.set_params(epsilon=epsilon, k=k, random_state=seed)

                for protocol in tqdm(lst_protocols, desc=f'seed={seed}, k={k}, epsilon={epsilon}'):
                    eps_emp: float = auditor.run_audit(protocol)
                    logRmax_eff_seed: float | None = getattr(auditor, "logRmax_eff", None)
                    if protocol == 'LRT':
                        
                        logging.info(
                            f"  Estimated log R_max (seed={seed}): {logRmax_eff_seed:.4f}"
                        )
                    results['seed'].append(seed)
                    results['protocol'].append(protocol)
                    results['k'].append(k)
                    results['delta'].append(delta)
                    results['epsilon'].append(epsilon)
                    results['eps_emp'].append(eps_emp)
                    results['logRmax_eff'].append(logRmax_eff_seed)

    # Convert and save results in csv file
    df = pd.DataFrame(results)
    print('==== Results DataFrame ====')
    print('df : \n', df)
    df.to_csv('results/ldp_audit_results_{}.csv'.format(analysis), index=False)
    
    return df


# ---------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------
if __name__ == "__main__":
    ## General parameters
    lst_eps: list[float] = [0.25, 0.5]
    # lst_eps: list[float] = [0.25, 0.5, 0.75, 1, 2, 4, 6, 10]
    lst_k: list[int] = [2]
    lst_seed = range(2)
    # lst_seed = range(5)
    nb_trials = int(1e6)
    alpha = 1e-2

    ## pure LDP protocols
    pure_ldp_protocols: list[str] = [
        "GRR",
        "SS",
        "SUE",
        # "OUE",
        # "THE",
        # "SHE",
        # "BLH",
        # "OLH",
        "LRT"
    ]
    delta = 0.0 
    analysis_pure = 'main_pure_ldp_protocols'
    df_pure: pd.DataFrame = run_main_experiments(
        nb_trials,
        alpha,
        pure_ldp_protocols,
        lst_seed,
        lst_k,
        lst_eps,
        delta,
        analysis_pure,
    )
    # Figure 1: Example of audit results for pure LDP protocols
    plot_results_example_audit(df_pure, pure_ldp_protocols, epsilon=0.5, k=2)  # Example of audit results -- Figure 1 in paper
    # Figure 2: Main results for pure LDP protocols
    plot_results_pure_ldp_protocols(df_pure, analysis_pure, pure_ldp_protocols, lst_eps, lst_k)  # Main results -- Figure 2 in paper
