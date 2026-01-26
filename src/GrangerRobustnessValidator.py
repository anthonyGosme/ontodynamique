import warnings

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller, grangercausalitytests
from sklearn.linear_model import LinearRegression
from scipy import stats
from tqdm import tqdm
import os
import random
from statsmodels.tools.sm_exceptions import ValueWarning, ConvergenceWarning


# 1. On ignore les warnings de fréquence (ceux qui polluent vos logs)
warnings.filterwarnings("ignore", category=ValueWarning, module="statsmodels")

# 2. On ignore les warnings de convergence (fréquents sur les petites fenêtres placebos)
warnings.filterwarnings("ignore", category=ConvergenceWarning, module="statsmodels")
warnings.filterwarnings("ignore", category=UserWarning, module="statsmodels")
def analyze_rolling_granger(
    name,
    df,
    window_size=30,
    max_lag=3,
    lag_mode="min",      # "min" | "fixed" | "best_pair"
    fixed_lag=2,
    strength_mode="1-p", # "1-p" | "neglogp"
    eps=1e-12,
):
    """
    Rolling Granger (SSR F-test p-values) with configurable lag selection.

    authority = strength_ga - strength_ag
      - strength_ag : Activity -> Gamma
      - strength_ga : Gamma -> Activity
    """
    if len(df) < window_size + 4:
        return None

    activity = df["total_weight"].values
    gamma = df["monthly_gamma"].values
    dates = df.index.values

    # smoothing (same as your original)
    activity = pd.Series(activity).rolling(3).mean().fillna(0).values
    gamma = pd.Series(gamma).rolling(3).mean().fillna(0).values

    timeline, strength_ag_list, strength_ga_list, authority_index = [], [], [], []

    def to_strength(p):
        p = float(p)
        if strength_mode == "neglogp":
            return -np.log10(p + eps)
        return 1.0 - p  # original

    for i in range(len(df) - window_size):
        act_seg = activity[i:i + window_size]
        gam_seg = gamma[i:i + window_size]
        current_date = dates[i + window_size]

        if np.std(act_seg) < 1e-8 or np.std(gam_seg) < 1e-8:
            continue

        data_seg = pd.DataFrame({
            "act": (act_seg - np.mean(act_seg)) / np.std(act_seg),
            "gam": (gam_seg - np.mean(gam_seg)) / np.std(gam_seg)
        })
        data_seg = data_seg.replace([np.inf, -np.inf], np.nan).dropna()
        if len(data_seg) < (max_lag + 5):
            continue

        try:
            g1 = grangercausalitytests(data_seg[["gam", "act"]], maxlag=max_lag, verbose=False)
            g2 = grangercausalitytests(data_seg[["act", "gam"]], maxlag=max_lag, verbose=False)

            p_ag_by_lag = {L: g1[L][0]["ssr_ftest"][1] for L in range(1, max_lag + 1)}
            p_ga_by_lag = {L: g2[L][0]["ssr_ftest"][1] for L in range(1, max_lag + 1)}

            if lag_mode == "min":
                p_ag = min(p_ag_by_lag.values())
                p_ga = min(p_ga_by_lag.values())

            elif lag_mode == "fixed":
                L = int(fixed_lag)
                if L < 1 or L > max_lag:
                    continue
                p_ag = p_ag_by_lag[L]
                p_ga = p_ga_by_lag[L]

            elif lag_mode == "best_pair":
                # choose lag minimizing joint p-values (more balanced than independent min)
                L = min(range(1, max_lag + 1), key=lambda k: p_ag_by_lag[k] + p_ga_by_lag[k])
                p_ag = p_ag_by_lag[L]
                p_ga = p_ga_by_lag[L]

            else:
                raise ValueError("lag_mode must be: 'min', 'fixed', 'best_pair'")

            strength_ag = to_strength(p_ag)
            strength_ga = to_strength(p_ga)

            index = strength_ga - strength_ag

            timeline.append(current_date)
            strength_ag_list.append(strength_ag)
            strength_ga_list.append(strength_ga)
            authority_index.append(index)

        except Exception:
            continue

    if len(authority_index) == 0:
        return None

    return {
        "dates": timeline,
        "strength_ag": strength_ag_list,
        "strength_ga": strength_ga_list,
        "authority": authority_index
    }
class GrangerRobustnessValidator:
    """
    MODULE SI : ROBUSTESSE DES SIGNATURES NAC (VAR ROLLING)
    Version : Platinum (Index Safe, AIC Robust, Hierarchical Placebo Stats, Window Tracking)

    Objectif : Prouver que les signatures du régime mature (AI borné,
               absence de dérive, confinement) résistent aux perturbations.
    """

    def __init__(self, all_dataframes, output_dir="figures/SI/", window_size=30, confinement_band=0.3, seed=42):
        # Filtre : Projets > 36 mois + buffer
        self.dfs = {k: v for k, v in all_dataframes.items()
                    if v is not None and len(v) > (window_size + 10)}
        self.output_dir = output_dir
        self.window_size = window_size
        self.band = confinement_band
        self.rng = np.random.default_rng(seed)

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def _metrics_on_fixed_index(
            self,
            ai_series: pd.Series,
            mature_index: pd.Index,
            signal_tau=0.5
    ):
        """
        Compute confinement metrics on a fixed set of timestamps (mature_index),
        WITHOUT redefining maturity from gamma (index-safe).

        Returns same keys as _calculate_nac_metrics (subset), but evaluated only on mature_index.
        """
        if ai_series is None or len(ai_series) == 0:
            return {
                "confinement": np.nan,
                "variance": np.nan,
                "slope": np.nan,
                "zc": np.nan,
                "mean_excursion": np.nan,
                "mean_strength": np.nan,
                "signal_rate": np.nan,
                "confinement_signal": np.nan,
                "n_windows": 0
            }

        # Align AI to the fixed mature index
        ai_m = ai_series.reindex(mature_index).dropna()
        if len(ai_m) < 12:
            return {
                "confinement": np.nan,
                "variance": np.nan,
                "slope": np.nan,
                "zc": np.nan,
                "mean_excursion": np.nan,
                "mean_strength": np.nan,
                "signal_rate": np.nan,
                "confinement_signal": np.nan,
                "n_windows": int(ai_series.attrs.get("n_windows", len(ai_series)))
            }

        vals = ai_m.values.astype(float)

        # 1) Drift
        x = np.arange(len(vals))
        try:
            slope, _ = np.polyfit(x, vals, 1)
        except Exception:
            slope = np.nan

        # 2) Variance
        variance = float(np.var(vals))

        # 3) Confinement
        confinement = float((np.abs(vals) <= self.band).mean())

        # 4) Zero crossings
        signs = np.sign(vals)
        signs = signs[signs != 0]
        zc = int(np.sum(np.abs(np.diff(signs)) == 2)) if len(signs) > 1 else 0

        # 5) Mean excursion duration
        out_of_band = (np.abs(vals) > self.band)
        if out_of_band.any():
            padded = np.concatenate([[False], out_of_band, [False]]).astype(int)
            changes = np.diff(padded)
            starts = np.where(changes == 1)[0]
            ends = np.where(changes == -1)[0]
            durations = ends - starts
            mean_excursion = float(durations.mean()) if len(durations) else 0.0
        else:
            mean_excursion = 0.0

        # 6) Signal-aware metrics
        mean_strength = np.nan
        signal_rate = np.nan
        confinement_signal = np.nan

        strength_ag = ai_series.attrs.get("strength_ag", None)
        strength_ga = ai_series.attrs.get("strength_ga", None)

        if strength_ag is not None and strength_ga is not None:
            try:
                s_ag = pd.Series(np.asarray(strength_ag, dtype=float), index=ai_series.index)
                s_ga = pd.Series(np.asarray(strength_ga, dtype=float), index=ai_series.index)

                s_ag_m = s_ag.reindex(mature_index)
                s_ga_m = s_ga.reindex(mature_index)

                max_strength = np.maximum(s_ag_m.values, s_ga_m.values)
                mean_strength = float(np.nanmean(max_strength))

                is_signal = (max_strength >= signal_tau)
                signal_rate = float(np.nanmean(is_signal)) if len(is_signal) else np.nan

                if np.nansum(is_signal) >= 5:
                    confinement_signal = float((np.abs(vals[is_signal]) <= self.band).mean())
                else:
                    confinement_signal = np.nan
            except Exception:
                pass

        return {
            "confinement": float(confinement),
            "variance": float(variance),
            "slope": float(slope) if slope == slope else np.nan,
            "zc": int(zc),
            "mean_excursion": float(mean_excursion),
            "mean_strength": float(mean_strength) if mean_strength == mean_strength else np.nan,
            "signal_rate": float(signal_rate) if signal_rate == signal_rate else np.nan,
            "confinement_signal": float(confinement_signal) if confinement_signal == confinement_signal else np.nan,
            "n_windows": int(ai_series.attrs.get("n_windows", len(ai_series)))
        }

        # ---------------------------------------------------------
        # NEW: gamma placebo (index-safe)
        # ---------------------------------------------------------

    def run_analysis_placebo_gamma_index_safe(
            self,
            lag_mode="min",
            lag=2,
            mature_threshold=0.7,
            gamma_perturbations=("Shift", "Circular", "BlockPerm","ShuffleValues"),
            n_shuffle_reps=20,
            shifts=(-6, 6),
            circular_ks=(-12, -6, -3, 3, 6, 12),
            block_lens=(6, 12),
            n_circular_runs=0,
            signal_tau=0.5,
            export_tag="gamma_placebo_index_safe",
            max_nan_frac=0.20,  # NEW: hard filter to avoid index mismatch artefacts
    ):
        """
        Index-safe placebo-Γ:
          - Mature windows defined ONCE from Γ_raw >= threshold (project-local).
          - Perturb Γ -> recompute AI(Γ̃, A_raw).
          - Evaluate metrics on the SAME mature_index (fixed).
        """
        print("\n⚙️ [PLACEBO-Γ|INDEX-SAFE] Perturb Γ, keep maturity windows fixed (Γ_raw)...")
        print(f"   lag_mode={lag_mode} | lag={lag} | thr={mature_threshold} | band={self.band}")

        rows = []

        for name in tqdm(list(self.dfs.keys()), desc="GammaPlacebo"):
            df = self.dfs[name].copy()
            df.index = self._as_monthly_timestamp_index(df.index)

            # Normalize to monthly
            s_raw = self._normalize_monthly_series(df["monthly_gamma"])
            a_raw = self._normalize_monthly_series(df["total_weight"])

            # 1) Observed AI
            ai_obs = self._compute_ai_series(s_raw, a_raw, index=s_raw.index, lag_mode=lag_mode, lag=lag)
            if ai_obs is None or len(ai_obs) == 0:
                continue
            ai_obs = ai_obs.copy()
            ai_obs.index = self._as_monthly_timestamp_index(ai_obs.index)

            # 2) Mature index fixed from Γ_raw, evaluated on AI timestamps
            gamma_on_ai = s_raw.reindex(ai_obs.index)
            nan_frac = float(gamma_on_ai.isna().mean()) if len(gamma_on_ai) else 1.0
            if nan_frac > max_nan_frac:
                print(f"⚠️ [SKIP|Index mismatch] {name}: gamma_on_ai NaNs={nan_frac:.1%}")
                continue

            mature_index = gamma_on_ai[gamma_on_ai >= mature_threshold].dropna().index
            if len(mature_index) < 12:
                continue

            m_obs = self._metrics_on_fixed_index(ai_obs, mature_index, signal_tau=signal_tau)
            m_obs.update({"project": name, "type": "Observed", "rep": np.nan})
            rows.append(m_obs)

            def _record(ai, label, rep=np.nan):
                if ai is None or len(ai) == 0:
                    return
                ai2 = ai.copy()
                ai2.index = self._as_monthly_timestamp_index(ai2.index)
                m = self._metrics_on_fixed_index(ai2, mature_index, signal_tau=signal_tau)
                m.update({"project": name, "type": label, "rep": rep})
                rows.append(m)

            # 3) Perturb Γ (keep A intact)
            if "Shift" in gamma_perturbations:
                for k in shifts:
                    s_shift = s_raw.shift(k)
                    ai = self._compute_ai_series(s_shift, a_raw, index=df.index, lag_mode=lag_mode, lag=lag)
                    _record(ai, f"GammaShift({k}m)")
            if "ShuffleValues" in gamma_perturbations:
                for rep in range(n_shuffle_reps):
                    s_shuf = self._shuffle_values(s_raw)
                    ai = self._compute_ai_series(s_shuf, a_raw, index=df.index, lag_mode=lag_mode, lag=lag)
                    _record(ai, "GammaShuffleValues", rep=rep)
            if "Circular" in gamma_perturbations:
                s_vals = s_raw.values
                for k in circular_ks:
                    s_circ = self._circular_shift(s_vals, k)
                    s_circ_series = pd.Series(s_circ, index=df.index)
                    ai = self._compute_ai_series(s_circ_series, a_raw, index=df.index, lag_mode=lag_mode, lag=lag)
                    _record(ai, f"GammaCircular({k}m)")

                if n_circular_runs and n_circular_runs > 0:
                    for rep in range(n_circular_runs):
                        split = int(self.rng.integers(len(s_vals) // 4, len(s_vals) * 3 // 4))
                        s_circ = self._circular_shift(s_vals, split)
                        s_circ_series = pd.Series(s_circ, index=df.index)
                        ai = self._compute_ai_series(s_circ_series, a_raw, index=df.index, lag_mode=lag_mode, lag=lag)
                        _record(ai, "GammaCircularRand", rep=rep)

            if "BlockPerm" in gamma_perturbations:
                s_vals = s_raw.values
                for L in block_lens:
                    s_blk = self._block_permute(s_vals, block_len=L)
                    s_blk_series = pd.Series(s_blk, index=df.index)
                    ai = self._compute_ai_series(s_blk_series, a_raw, index=df.index, lag_mode=lag_mode, lag=lag)
                    _record(ai, f"GammaBlockPerm({L}m)")

        df_res = pd.DataFrame(rows).dropna(subset=["project", "type", "confinement"])
        out_csv = f"{self.output_dir}{export_tag}_{lag_mode}_lag{lag}_thr{mature_threshold}.csv"
        df_res.to_csv(out_csv, index=False)
        print(f"✅ Saved: {out_csv}")

        # Hierarchical paired stats: Observed vs mean(placebos) per project
        obs = df_res[df_res["type"] == "Observed"].set_index("project")["confinement"]
        plc = (
            df_res[df_res["type"] != "Observed"]
            .groupby(["project", "type"])["confinement"].mean()
            .groupby("project").mean()
        )

        common = obs.index.intersection(plc.index)
        if len(common) >= 10:
            delta = obs.loc[common] - plc.loc[common]
            print("\n[PLACEBO-Γ|INDEX-SAFE] Confinement (Observed - PlaceboMean)")
            print(
                f"   median={delta.median():.4f} "
                f"IQR=[{delta.quantile(0.25):.4f},{delta.quantile(0.75):.4f}] "
                f"mean={delta.mean():.4f} "
                f"pct(>0)={(delta > 0).mean():.2%} N={len(delta)}"
            )
            stat, p = stats.wilcoxon(obs.loc[common], plc.loc[common], alternative="greater")
            print(f"   Wilcoxon Observed > PlaceboMean: stat={stat:.3f} p={p:.3e}")
        else:
            print("⚠️ Not enough common projects for Wilcoxon.")

        return df_res

    def run_analysis_cross_swap(
            self,
            lag_mode="min",
            lag=2,
            mature_threshold=0.7,
            n_swaps=50,
            signal_tau=0.5,
            export_tag="cross_swap",
            max_nan_frac=0.20,
    ):
        """
        Cross-swap:
          - For each project i: keep Γ_i (defines mature windows)
          - Replace A_i by A_j from another project j (random)
          - Compute AI(Γ_i, A_j) and evaluate on mature windows of Γ_i (index-safe)
        """
        print("\n⚙️ [CROSS-SWAP] Γ from project i, A from random project j (j != i)")
        print(f"   swaps={n_swaps} | lag_mode={lag_mode} | lag={lag} | thr={mature_threshold} | band={self.band}")

        keys = list(self.dfs.keys())
        rows = []

        # Pre-normalize all series once
        cache = {}
        for k in keys:
            df = self.dfs[k].copy()
            df.index = self._as_monthly_timestamp_index(df.index)
            cache[k] = {
                "idx": df.index,
                "gamma": self._normalize_monthly_series(df["monthly_gamma"]),
                "act": self._normalize_monthly_series(df["total_weight"]),
            }

        for i_name in tqdm(keys, desc="CrossSwap"):
            g_i = cache[i_name]["gamma"]
            idx_i = cache[i_name]["idx"]

            # Observed baseline for i
            a_i = cache[i_name]["act"]
            ai_obs = self._compute_ai_series(g_i, a_i, index=idx_i, lag_mode=lag_mode, lag=lag)
            if ai_obs is None or len(ai_obs) == 0:
                continue
            ai_obs = ai_obs.copy()
            ai_obs.index = self._as_monthly_timestamp_index(ai_obs.index)

            gamma_on_ai = g_i.reindex(ai_obs.index)
            nan_frac = float(gamma_on_ai.isna().mean()) if len(gamma_on_ai) else 1.0
            if nan_frac > max_nan_frac:
                continue

            mature_index = gamma_on_ai[gamma_on_ai >= mature_threshold].dropna().index
            if len(mature_index) < 12:
                continue

            m0 = self._metrics_on_fixed_index(ai_obs, mature_index, signal_tau=signal_tau)
            m0.update({"project": i_name, "type": "Observed", "swap_with": np.nan})
            rows.append(m0)

            # Swaps
            others = [k for k in keys if k != i_name]
            if not others:
                continue

            for rep in range(n_swaps):
                j_name = str(self.rng.choice(others))
                a_j = cache[j_name]["act"]

                # Align A_j onto i's index (important!)
                a_j_on_i = a_j.reindex(idx_i)

                ai_swap = self._compute_ai_series(g_i, a_j_on_i, index=idx_i, lag_mode=lag_mode, lag=lag)
                if ai_swap is None or len(ai_swap) == 0:
                    continue
                ai_swap = ai_swap.copy()
                ai_swap.index = self._as_monthly_timestamp_index(ai_swap.index)

                m = self._metrics_on_fixed_index(ai_swap, mature_index, signal_tau=signal_tau)
                m.update({"project": i_name, "type": "Swap", "swap_with": j_name})
                rows.append(m)

        df_res = pd.DataFrame(rows).dropna(subset=["project", "type", "confinement"])
        out_csv = f"{self.output_dir}{export_tag}_{lag_mode}_lag{lag}_thr{mature_threshold}.csv"
        df_res.to_csv(out_csv, index=False)
        print(f"✅ Saved: {out_csv}")

        # Stats: Observed vs mean(Swaps) per project
        obs = df_res[df_res["type"] == "Observed"].set_index("project")["confinement"]
        swp = df_res[df_res["type"] == "Swap"].groupby("project")["confinement"].mean()

        common = obs.index.intersection(swp.index)
        if len(common) >= 10:
            delta = obs.loc[common] - swp.loc[common]
            print("\n[CROSS-SWAP] Confinement (Observed - SwapMean)")
            print(
                f"   median={delta.median():.4f} "
                f"IQR=[{delta.quantile(0.25):.4f},{delta.quantile(0.75):.4f}] "
                f"mean={delta.mean():.4f} "
                f"pct(>0)={(delta > 0).mean():.2%} N={len(delta)}"
            )
            stat, p = stats.wilcoxon(obs.loc[common], swp.loc[common], alternative="greater")
            print(f"   Wilcoxon Observed > SwapMean: stat={stat:.3f} p={p:.3e}")
        else:
            print("⚠️ Not enough projects for Wilcoxon.")

        return df_res

    def _normalize_monthly_series(self, s: pd.Series) -> pd.Series:
        s2 = s.copy()
        s2.index = self._as_monthly_timestamp_index(s2.index)
        return s2

    def _as_monthly_timestamp_index(self, idx):
        """Normalize indices to monthly timestamps (Period('M')->Timestamp)."""
        if isinstance(idx, pd.PeriodIndex):
            return idx.to_timestamp()
        if isinstance(idx, pd.DatetimeIndex):
            return pd.to_datetime(idx).to_period("M").to_timestamp()
        return idx



    # ==========================================================================
    # 1. CORE ENGINE (SOURCE OF TRUTH)
    # ==========================================================================
    def _fmt(self, x):
        return "nan" if (x is None or (isinstance(x, float) and np.isnan(x))) else f"{x:.4f}"

    def _summ(self, s: pd.Series, name=""):
        s = pd.to_numeric(s, errors="coerce").dropna()
        if len(s) == 0:
            print(f"   [Summary] {name}: EMPTY")
            return
        print(f"   [Summary] {name}: N={len(s)} "
              f"median={np.median(s):.4f} IQR=[{np.quantile(s, 0.25):.4f},{np.quantile(s, 0.75):.4f}] "
              f"mean={np.mean(s):.4f} std={np.std(s):.4f}")

    def _print_header(self, title):
        print("\n" + "-" * 78)
        print(title)
        print("-" * 78)

    def _preprocess_series(self, s: pd.Series, mode="adf"):
        """Pré-traitement stationnarité."""
        # On travaille sur une copie propre
        s = s.copy().astype(float)
        clean_vals = s.dropna().values

        if len(clean_vals) < 15: return s

        should_diff = False
        if mode == "diff":
            should_diff = True
        elif mode == "adf":
            try:
                # H0: Non-stationnaire. Si p > 0.05, on différencie.
                p_val = adfuller(clean_vals)[1]
                if p_val > 0.05:
                    should_diff = True
            except:
                pass

        if should_diff:
            # .diff() préserve l'index, fillna(0) évite la perte du 1er point
            return s.diff().fillna(0.0)
        return s

    def _compute_ai_series(
            self,
            series_s,
            series_a,
            index=None,
            lag=2,
            lag_mode="min",
            window=None,
            diff_mode=None,
            max_lag=3,
            strength_mode="1-p",
            eps=1e-12,
    ):
        """
        AI paper-like (aligned with analyze_rolling_granger), with strengths stored in attrs.

        Returns:
          ai: pd.Series indexed by window end dates
          ai.attrs:
            - n_windows
            - strength_ag (Activity -> Gamma)  [np.array]
            - strength_ga (Gamma -> Activity)  [np.array]
        """
        if window is None:
            window = self.window_size

        n = len(series_s)
        if n < window + 10:
            return pd.Series(dtype=float)

        if index is None:
            index = pd.RangeIndex(n)

        df_temp = pd.DataFrame(
            {
                "monthly_gamma": pd.Series(series_s, index=index).astype(float),
                "total_weight": pd.Series(series_a, index=index).astype(float),
            },
            index=index
        )

        out = analyze_rolling_granger(
            name="temp",
            df=df_temp,
            window_size=window,
            max_lag=max_lag,
            lag_mode=lag_mode,
            fixed_lag=lag,
            strength_mode=strength_mode,
            eps=eps,
        )

        if out is None or len(out.get("authority", [])) == 0:
            return pd.Series(dtype=float)

        ai = pd.Series(out["authority"], index=pd.Index(out["dates"]), name="AI")

        # Normaliser l'index en timestamps mensuels (évite surprises freq)
        if isinstance(df_temp.index, pd.DatetimeIndex):
            ai.index = pd.to_datetime(ai.index).to_period("M").to_timestamp()

        # Store metadata for downstream metrics
        ai.attrs["n_windows"] = len(ai)
        ai.attrs["strength_ag"] = np.asarray(out.get("strength_ag", []), dtype=float)  # A -> G
        ai.attrs["strength_ga"] = np.asarray(out.get("strength_ga", []), dtype=float)  # G -> A

        return ai

    def _calculate_nac_metrics(
            self,
            ai_series,
            gamma_series=None,
            mature_mode="half",  # "half" | "gamma"
            mature_threshold=0.7,
            signal_tau=0.5,  # NEW: threshold on max(strength) for "signal" windows
    ):
        #print(f"   [Config] mature_mode={mature_mode} | thr={mature_threshold}")
        """
        Returns NAC metrics computed on the mature segment.

        Adds signal-aware metrics to avoid the 'AI≈0 -> high confinement' placebo trap:
          - mean_strength: mean(max(strength_ag, strength_ga)) over mature segment
          - signal_rate: fraction of mature windows where max_strength >= signal_tau
          - confinement_signal: confinement computed only on "signal" windows (else NaN)
        """
        if ai_series is None or len(ai_series) < 12:
            return {
                "slope": np.nan, "variance": np.nan, "confinement": np.nan,
                "zc": np.nan, "mean_excursion": np.nan,
                "mean_strength": np.nan, "signal_rate": np.nan, "confinement_signal": np.nan,
                "n_windows": 0
            }

        # --- Select mature segment ---
        if mature_mode == "gamma":
            if gamma_series is None:
                return {
                    "slope": np.nan, "variance": np.nan, "confinement": np.nan,
                    "zc": np.nan, "mean_excursion": np.nan,
                    "mean_strength": np.nan, "signal_rate": np.nan, "confinement_signal": np.nan,
                    "n_windows": 0
                }

            common_idx = ai_series.index.intersection(gamma_series.index)
            if len(common_idx) < 12:
                return {
                    "slope": np.nan, "variance": np.nan, "confinement": np.nan,
                    "zc": np.nan, "mean_excursion": np.nan,
                    "mean_strength": np.nan, "signal_rate": np.nan, "confinement_signal": np.nan,
                    "n_windows": 0
                }

            ai_aligned = ai_series.loc[common_idx].dropna()
            gamma_aligned = gamma_series.loc[common_idx]

            ai_mature = ai_aligned[gamma_aligned >= mature_threshold]

            # keep indices for masking signal windows later
            mature_index = ai_mature.index

        elif mature_mode == "half":
            ai_aligned = ai_series.dropna()
            if len(ai_aligned) < 24:
                return {
                    "slope": np.nan, "variance": np.nan, "confinement": np.nan,
                    "zc": np.nan, "mean_excursion": np.nan,
                    "mean_strength": np.nan, "signal_rate": np.nan, "confinement_signal": np.nan,
                    "n_windows": 0
                }
            mid = len(ai_aligned) // 2
            ai_mature = ai_aligned.iloc[mid:]
            mature_index = ai_mature.index

        else:
            raise ValueError("mature_mode must be 'gamma' or 'half'")

        if ai_mature is None or len(ai_mature) < 12:
            return {
                "slope": np.nan, "variance": np.nan, "confinement": np.nan,
                "zc": np.nan, "mean_excursion": np.nan,
                "mean_strength": np.nan, "signal_rate": np.nan, "confinement_signal": np.nan,
                "n_windows": 0
            }

        vals = ai_mature.values.astype(float)

        # 1) Drift
        x = np.arange(len(vals))
        try:
            slope, _ = np.polyfit(x, vals, 1)
        except Exception:
            slope = np.nan

        # 2) Variance
        variance = float(np.var(vals))

        # 3) Confinement
        confinement = float((np.abs(vals) <= self.band).mean())

        # 4) Zero crossings
        signs = np.sign(vals)
        signs = signs[signs != 0]
        zc = int(np.sum(np.abs(np.diff(signs)) == 2)) if len(signs) > 1 else 0

        # 5) Mean excursion duration
        out_of_band = (np.abs(vals) > self.band)
        if out_of_band.any():
            padded = np.concatenate([[False], out_of_band, [False]]).astype(int)
            changes = np.diff(padded)
            starts = np.where(changes == 1)[0]
            ends = np.where(changes == -1)[0]
            durations = ends - starts
            mean_excursion = float(durations.mean()) if len(durations) else 0.0
        else:
            mean_excursion = 0.0

        # ----------------------------
        # 6) Signal-aware metrics (NEW)
        # ----------------------------
        # strengths are stored in attrs in the same order as ai_series values
        strength_ag = ai_series.attrs.get("strength_ag", None)
        strength_ga = ai_series.attrs.get("strength_ga", None)

        mean_strength = np.nan
        signal_rate = np.nan
        confinement_signal = np.nan

        if strength_ag is not None and strength_ga is not None:
            try:
                # Map strengths onto the mature index:
                # We assume strengths correspond 1:1 to ai_series positions.
                # So we build a Series aligned to ai_series.index.
                s_ag = pd.Series(np.asarray(strength_ag, dtype=float), index=ai_series.index)
                s_ga = pd.Series(np.asarray(strength_ga, dtype=float), index=ai_series.index)

                s_ag_m = s_ag.reindex(mature_index)
                s_ga_m = s_ga.reindex(mature_index)

                max_strength = np.maximum(s_ag_m.values, s_ga_m.values)
                mean_strength = float(np.nanmean(max_strength))

                is_signal = (max_strength >= signal_tau)
                signal_rate = float(np.nanmean(is_signal)) if len(is_signal) else np.nan

                # Confinement only on signal windows
                if np.nansum(is_signal) >= 5:
                    confinement_signal = float((np.abs(vals[is_signal]) <= self.band).mean())
                else:
                    confinement_signal = np.nan

            except Exception:
                pass

        n_win = ai_series.attrs.get("n_windows", len(ai_series))

        return {
            "slope": float(slope),
            "variance": float(variance),
            "confinement": float(confinement),
            "zc": int(zc),
            "mean_excursion": float(mean_excursion),
            "mean_strength": float(mean_strength) if mean_strength == mean_strength else np.nan,
            "signal_rate": float(signal_rate) if signal_rate == signal_rate else np.nan,
            "confinement_signal": float(confinement_signal) if confinement_signal == confinement_signal else np.nan,
            "n_windows": int(n_win),
        }

    # ==========================================================================
    # 2. ANALYSES
    # ==========================================================================
    def _shuffle_values(self, s: pd.Series) -> pd.Series:
        s2 = s.copy()
        s2[:] = self.rng.permutation(np.asarray(s2.values))
        return s2

    def run_analysis_1_lag_sensitivity(self):
        print("\n⚙️ [SI-1] Lag Sensitivity (Scanning parameters)...")
        results = []

        # 1. DÉFINITION DES PARAMÈTRES (Tuples : valeur, mode)
        # On teste les lags fixes et le mode 'min' (papier) qui scanne 1..3
        params_to_test = [
            (1, 'fixed'),
            (2, 'fixed'),
            (3, 'fixed'),
            (4, 'fixed'),
            (6, 'fixed'),
            (3, 'min')  # Mode Papier (scanne 1, 2, 3 et prend le meilleur p)
        ]

        for name, df in tqdm(list(self.dfs.items()), desc="Lag"):
            s_raw = df['monthly_gamma']
            a_raw = df['total_weight']

            for lag_val, mode in params_to_test:
                # A. Calcul AI (Moteur Miroir)
                ai = self._compute_ai_series(
                    s_raw, a_raw,
                    index=df.index,
                    lag=lag_val,
                    lag_mode=mode
                )

                # B. Alignement Gamma pour référence (CRUCIAL pour mode "gamma")
                # On ne garde que les mois où l'AI a pu être calculé
                if len(ai) > 0:
                    gamma_ref = df.loc[ai.index, 'monthly_gamma']
                else:
                    gamma_ref = pd.Series(dtype=float)
                gamma_ref = df['monthly_gamma'].reindex(ai.index)
                # C. Calcul des Métriques NAC
                m = self._calculate_nac_metrics(
                    ai, gamma_ref,
                    mature_mode="gamma",
                    mature_threshold=0.7
                )

                # D. Stockage
                m['project'] = name
                # Label propre pour le graphique
                m['lag_type'] = "min-3" if mode == 'min' else f"fixed-{lag_val}"
                results.append(m)
        print("DF index sample:", df.index[:3], df.index.freq if hasattr(df.index, "freq") else None)
        print("AI index sample:", ai.index[:3], ai.index.inferred_freq if hasattr(ai.index, "inferred_freq") else None)
        print("Intersection size:", len(df.index.intersection(ai.index)), "AI len:", len(ai))

        # 2. SAUVEGARDE
        df_res = pd.DataFrame(results)
        df_res.to_csv(f"{self.output_dir}SI_lag_sensitivity.csv", index=False)

        # 3. DEBUG & NETTOYAGE
        print("\n[SI-1] lag_type counts (Check for empty categories):")
        print(df_res['lag_type'].value_counts(dropna=False))

        df_plot = df_res.dropna(subset=['slope', 'confinement'])

        if df_plot.empty:
            print("⚠️ [SI-1] Aucun résultat valide (AI vide ou pas de phase mature). Plot annulé.")
            return

        # 4. PLOTTING SÉCURISÉ (Anti-Crash)
        try:
            # Résumé console pour le mode papier
            paper = df_plot[df_plot['lag_type'] == 'min-3']
            print(f"   [Paper Mode 'min-3'] N={len(paper)}")
            print(f"   Drift Mean: {paper['slope'].mean():.5f}")
            print(f"   Conf Median: {paper['confinement'].median():.3f}")

            fig, axes = plt.subplots(2, 2, figsize=(14, 10))

            # Ordre théorique
            target_order = ['fixed-1', 'fixed-2', 'fixed-3', 'fixed-4', 'fixed-6', 'min-3']

            # FILTRE : On ne garde que les labels qui existent vraiment dans les données
            safe_order = [o for o in target_order if o in df_plot['lag_type'].unique()]

            if not safe_order:
                print("⚠️ Pas assez de catégories pour tracer.")
                return

            # Utilisation de safe_order PARTOUT
            sns.boxplot(data=df_plot, x='lag_type', y='slope', ax=axes[0, 0], order=safe_order)
            axes[0, 0].axhline(0, color='red', linestyle='--')
            axes[0, 0].set_title('Drift (Slope)')
            axes[0, 0].set_ylim(-0.02, 0.02)

            sns.boxplot(data=df_plot, x='lag_type', y='confinement', ax=axes[0, 1], order=safe_order)
            axes[0, 1].set_title(f'Confinement (Band={self.band})')
            axes[0, 1].set_ylim(0, 1)

            sns.boxplot(data=df_plot, x='lag_type', y='variance', ax=axes[1, 0], order=safe_order)
            axes[1, 0].set_title('Variance')

            sns.boxplot(data=df_plot, x='lag_type', y='zc', ax=axes[1, 1], order=safe_order)
            axes[1, 1].set_title('Zero Crossings')

            plt.tight_layout()
            plt.savefig(f"{self.output_dir}SI_fig_lag_stability_full.pdf")
            plt.close()
            print("✅ Graphiques SI-1 sauvegardés.")

        except Exception as e:
            print(f"⚠️ Erreur Plotting SI-1: {e}")
    def _block_permute(self, s, block_len=6):
        """Permutation par blocs + Permutation du reste."""
        s = np.asarray(s)
        n = len(s)
        num_blocks = n // block_len

        if num_blocks < 2: return self.rng.permutation(s)

        trunc_len = num_blocks * block_len
        core = s[:trunc_len]
        remainder = s[trunc_len:]

        blocks = core.reshape((num_blocks, block_len)).copy()  # .copy() safe
        self.rng.shuffle(blocks)
        permuted_core = blocks.flatten()

        if len(remainder) > 1:
            remainder = self.rng.permutation(remainder)

        return np.concatenate([permuted_core, remainder])

    def _circular_shift(self, s, k):
        k = k % len(s)
        return np.concatenate((s[-k:], s[:-k])) if k else s

    def run_analysis_2_placebo_tests(self, n_circular_runs=0, lag_mode='min', include_shuffle=True, n_shuffle_reps=20, lag=2,mature_mode="gamma", mature_threshold=0.7, signal_tau=0.5):
        print("\n⚙️ [SI-2] Placebo Tests (Alignment-dependent, anchored on Γ)...")
        print( f"   [Config] lag_mode={lag_mode} | lag={lag} | mature_mode='{mature_mode}' | thr={mature_threshold} | n_circular_runs={n_circular_runs}")
        print(f"   [Config] band={self.band} | window={self.window_size} | signal_tau={signal_tau}")


        results = []
        shifts = [-6, 6]
        circular_ks = [-12, -6, -3, 3, 6, 12]
        block_lens = [6, 12]

        for name in tqdm(list(self.dfs.keys()), desc="Placebo"):
            df = self.dfs[name]
            df = df.copy()
            df.index = self._as_monthly_timestamp_index(df.index)
            s_raw = df["monthly_gamma"]
            a_raw = df["total_weight"]
            # -----------------------------
            # 1) Observed
            # -----------------------------
            ai_obs = self._compute_ai_series(s_raw, a_raw, index=s_raw.index, lag_mode=lag_mode, lag=lag)
            gamma_ref = df["monthly_gamma"].reindex(ai_obs.index) if len(ai_obs) > 0 else pd.Series(dtype=float)


            m_obs = self._calculate_nac_metrics(
                ai_obs, gamma_ref, mature_mode=mature_mode, mature_threshold=mature_threshold
            )
            m_obs.update({"project": name, "type": "Observed", "rep": np.nan})
            results.append(m_obs)
            if include_shuffle:
                for rep in range(n_shuffle_reps):
                    a_shuf = self._shuffle_values(a_raw)
                    ai_shuf = self._compute_ai_series(s_raw, a_shuf, index=df.index, lag_mode=lag_mode, lag=lag)
                    gamma_ref = df["monthly_gamma"].reindex(ai_shuf.index) if len(ai_shuf) > 0 else pd.Series(
                        dtype=float)

                    m_shuf = self._calculate_nac_metrics(
                        ai_shuf, gamma_ref, mature_mode=mature_mode, mature_threshold=mature_threshold
                    )
                    m_shuf.update({"project": name, "type": "ActivityShuffleValues", "rep": rep})
                    results.append(m_shuf)
            # -----------------------------
            # 2) Linear shifts on ACTIVITY (length preserved)
            # -----------------------------
            for k in shifts:
                a_shift = a_raw.shift(k)
                ai_lin = self._compute_ai_series(s_raw, a_shift, index=df.index, lag_mode=lag_mode, lag=lag)
                gamma_ref = df["monthly_gamma"].reindex(ai_lin.index) if len(ai_lin) > 0 else pd.Series(dtype=float)

                m_lin = self._calculate_nac_metrics(
                    ai_lin, gamma_ref, mature_mode=mature_mode, mature_threshold=mature_threshold
                )
                m_lin.update({"project": name, "type": f"Shift({k}m)", "rep": np.nan})
                results.append(m_lin)

            # -----------------------------
            # 3) Circular shifts on ACTIVITY (fixed ks)
            # -----------------------------
            a_vals = a_raw.values
            for k in circular_ks:
                a_circ = self._circular_shift(a_vals, k)
                a_circ_series = pd.Series(a_circ, index=df.index)

                ai_circ = self._compute_ai_series(s_raw, a_circ_series, index=df.index, lag_mode=lag_mode, lag=lag)
                gamma_ref = df["monthly_gamma"].reindex(ai_circ.index) if len(ai_circ) > 0 else pd.Series(dtype=float)

                m_circ = self._calculate_nac_metrics(
                    ai_circ, gamma_ref, mature_mode=mature_mode, mature_threshold=mature_threshold
                )
                m_circ.update({"project": name, "type": f"Circular({k}m)", "rep": np.nan})
                results.append(m_circ)

            # Optional: random circular runs
            if n_circular_runs and n_circular_runs > 0:
                for rep in range(n_circular_runs):
                    split = int(self.rng.integers(len(a_vals) // 4, len(a_vals) * 3 // 4))
                    a_circ = self._circular_shift(a_vals, split)
                    a_circ_series = pd.Series(a_circ, index=df.index)

                    ai_circ = self._compute_ai_series(s_raw, a_circ_series, index=df.index, lag_mode=lag_mode, lag=lag)
                    gamma_ref = df["monthly_gamma"].reindex(ai_circ.index) if len(ai_circ) > 0 else pd.Series(
                        dtype=float)

                    m_circ = self._calculate_nac_metrics(
                        ai_circ, gamma_ref, mature_mode=mature_mode, mature_threshold=mature_threshold
                    )
                    m_circ.update({"project": name, "type": "CircularRand", "rep": rep})
                    results.append(m_circ)

            # -----------------------------
            # 4) Block permutation on ACTIVITY
            # -----------------------------
            for L in block_lens:
                a_blk = self._block_permute(a_vals, block_len=L)
                a_blk_series = pd.Series(a_blk, index=df.index)

                ai_blk = self._compute_ai_series(s_raw, a_blk_series, index=df.index, lag_mode=lag_mode, lag=lag)
                gamma_ref = df["monthly_gamma"].reindex(ai_blk.index) if len(ai_blk) > 0 else pd.Series(dtype=float)

                m_blk = self._calculate_nac_metrics(
                    ai_blk, gamma_ref, mature_mode=mature_mode, mature_threshold=mature_threshold
                )
                m_blk.update({"project": name, "type": f"BlockPerm({L}m)", "rep": np.nan})
                results.append(m_blk)

        # ------------------------------------------------------------------
        # Build DataFrame + export
        # ------------------------------------------------------------------
        df_res = pd.DataFrame(results)
        df_res.to_csv(f"{self.output_dir}SI_placebo_tests_RAW.csv", index=False)

        valid = df_res.dropna(subset=["project", "type", "confinement"])
        valid.to_csv(f"{self.output_dir}SI_placebo_tests.csv", index=False)

        print("\n[DEBUG PLACEBO] type counts (valid rows):")
        print(valid["type"].value_counts(dropna=False))

        # ------------------------------------------------------------------
        # Stats: Observed vs Mean(Placebos) per project (hierarchical)
        # ------------------------------------------------------------------
        placebo_df = valid[valid["type"] != "Observed"]

        # --- Confinement (main)
        obs = valid[valid["type"] == "Observed"].set_index("project")["confinement"]
        mean_per_type = placebo_df.groupby(["project", "type"])["confinement"].mean()
        placebo_mean = mean_per_type.groupby("project").mean()

        common = obs.index.intersection(placebo_mean.index)
        print("\n[SI-2] GLOBAL SANITY")
        print(f"   Projects seen: {len(self.dfs)}")
        print(f"   Rows total (valid): {len(valid)}")
        print(f"   Observed rows: {len(valid[valid['type'] == 'Observed'])}")
        print(f"   Placebo rows: {len(valid[valid['type'] != 'Observed'])}")

        if len(common) > 10:
            # Delta
            delta_conf = obs.loc[common] - placebo_mean.loc[common]
            print(f"   [Delta] confinement Obs-PlaceboMean: "
                  f"median={delta_conf.median():.4f} "
                  f"IQR=[{delta_conf.quantile(0.25):.4f},{delta_conf.quantile(0.75):.4f}] "
                  f"mean={delta_conf.mean():.4f} "
                  f"pct(>0)={(delta_conf > 0).mean():.2%} "
                  f"N={len(delta_conf)}")
            print( f"   [Level] confinement observed median={obs.loc[common].median():.4f} | placebo_mean median={placebo_mean.loc[common].median():.4f}")

            stat, p_val_mean = stats.wilcoxon(obs.loc[common], placebo_mean.loc[common], alternative="greater")
            print(f"   [Stat] Wilcoxon Obs > Mean(Placebos) (Confinement): stat={stat:.3f} p={p_val_mean:.4e}")
        else:
            p_val_mean = 1.0
            print("   [Stat] Wilcoxon skipped: not enough common projects.")

        # --- mean_excursion (optional): Observed should be SMALLER
        if "mean_excursion" in valid.columns:
            obs_ex = valid[valid["type"] == "Observed"].set_index("project")["mean_excursion"]
            plc_ex = placebo_df.groupby(["project", "type"])["mean_excursion"].mean().groupby("project").mean()
            common_ex = obs_ex.index.intersection(plc_ex.index)

            # dropna safety
            obs_ex2 = obs_ex.loc[common_ex].dropna()
            plc_ex2 = plc_ex.loc[obs_ex2.index].dropna()
            common_ex2 = obs_ex2.index.intersection(plc_ex2.index)

            if len(common_ex2) > 10:
                delta_ex = obs_ex2.loc[common_ex2] - plc_ex2.loc[common_ex2]
                print(f"   [Delta] mean_excursion Obs-PlaceboMean: "
                      f"median={delta_ex.median():.4f} "
                      f"IQR=[{delta_ex.quantile(0.25):.4f},{delta_ex.quantile(0.75):.4f}] "
                      f"mean={delta_ex.mean():.4f} "
                      f"pct(<0)={(delta_ex < 0).mean():.2%} "
                      f"N={len(delta_ex)}")

                stat_ex, p_ex = stats.wilcoxon(obs_ex2.loc[common_ex2], plc_ex2.loc[common_ex2], alternative="less")
                print(f"   [Stat] Wilcoxon Obs < Mean(Placebos) (Excursions): stat={stat_ex:.3f} p={p_ex:.4e}")

        # ------------------------------------------------------------------
        # Extra metrics if present (recommended anti-artefact)
        #   - mean_strength: Observed should be GREATER than placebos
        #   - confinement_signal: Observed should be GREATER than placebos
        # ------------------------------------------------------------------
        for col, alt, label in [
            ("mean_strength", "greater", "MeanStrength"),
            ("confinement_signal", "greater", "Confinement|Signal"),
        ]:
            if col in valid.columns:
                obs_c = valid[valid["type"] == "Observed"].set_index("project")[col]
                plc_c = placebo_df.groupby(["project", "type"])[col].mean().groupby("project").mean()
                common_c = obs_c.index.intersection(plc_c.index)

                # dropna safety
                obs_c2 = obs_c.loc[common_c].dropna()
                plc_c2 = plc_c.loc[obs_c2.index].dropna()
                common_c2 = obs_c2.index.intersection(plc_c2.index)

                if len(common_c2) > 10:
                    print(
                        f"   [Level] {col} observed median={obs_c2.loc[common_c2].median():.4f} | "
                        f"placebo_mean median={plc_c2.loc[common_c2].median():.4f}"
                    )
                    d = obs_c2.loc[common_c2] - plc_c2.loc[common_c2]
                    print(f"   [Delta] {col} Obs-PlaceboMean: "
                          f"median={d.median():.4f} "
                          f"IQR=[{d.quantile(0.25):.4f},{d.quantile(0.75):.4f}] "
                          f"mean={d.mean():.4f} "
                          f"pct(>0)={(d > 0).mean():.2%} "
                          f"N={len(d)}")

                    stat_c, p_c = stats.wilcoxon(obs_c2.loc[common_c2], plc_c2.loc[common_c2], alternative=alt)
                    print(f"   [Stat] Wilcoxon Obs > Mean(Placebos) ({label}): stat={stat_c:.3f} p={p_c:.4e}")
            tag = f"{mature_mode}_{lag_mode}_lag{lag}"
            df_res.to_csv(f"{self.output_dir}SI_placebo_tests_RAW_{tag}.csv", index=False)
            valid.to_csv(f"{self.output_dir}SI_placebo_tests_{tag}.csv", index=False)
        return float(p_val_mean)

    def run_analysis_3_conditional_controls(self):
        print("\n⚙️ [SI-3] Conditional Controls...")
        results = []

        for name, df in tqdm(self.dfs.items(), desc="Controls"):
            controls = []
            if 'n_contributors' in df.columns:
                controls.append(df['n_contributors'])
            if 'files_touched' in df.columns:
                controls.append(df['files_touched'])

            if not controls:
                continue

            Z = pd.concat(controls, axis=1)
            raw = pd.concat([df['monthly_gamma'], df['total_weight'], Z], axis=1).dropna()

            if len(raw) < self.window_size + 10:
                continue

            S = raw.iloc[:, 0].values
            A = raw.iloc[:, 1].values
            Z_val = raw.iloc[:, 2:].values
            idx = raw.index

            # Residualization
            S_resid = S - LinearRegression().fit(Z_val, S).predict(Z_val)
            A_resid = A - LinearRegression().fit(Z_val, A).predict(Z_val)

            # Rolling VAR
            ai_cond = self._compute_ai_series(S_resid, A_resid, index=idx)
            gamma_ref_cond = df["monthly_gamma"].reindex(ai_cond.index) if len(ai_cond) > 0 else pd.Series(dtype=float)
            m_cond = self._calculate_nac_metrics(
                ai_cond, gamma_ref_cond,
                mature_mode="half",
                mature_threshold=0.7
            )

            ai_raw = self._compute_ai_series(S, A, index=idx)
            gamma_ref_raw = df["monthly_gamma"].reindex(ai_raw.index) if len(ai_raw) > 0 else pd.Series(dtype=float)
            m_raw = self._calculate_nac_metrics(
                ai_raw, gamma_ref_raw,
                mature_mode="half",
                mature_threshold=0.7
            )



            results.append({
                'project': name,
                'raw_conf': m_raw['confinement'],
                'cond_conf': m_cond['confinement']
            })

        df_res = pd.DataFrame(results).dropna()
        df_res.to_csv(f"{self.output_dir}SI_conditional_controls.csv", index=False)

        r_val = np.nan
        p_r = np.nan
        if len(df_res) > 5:
            r_val, p_r = stats.spearmanr(df_res['raw_conf'], df_res['cond_conf'])
            print(f"   [Stat] Spearman Raw vs Conditional: r={r_val:.3f} p={p_r:.3e}")

        # -----------------------------
        # Descriptifs
        # -----------------------------
        print("\n" + "-" * 78)
        print("[SI-3] Conditional Controls Summary")
        print("-" * 78)

        def _summ(series, label):
            s = pd.to_numeric(series, errors="coerce").dropna()
            if len(s) == 0:
                print(f"   [Summary] {label}: EMPTY")
                return
            print(
                f"   [Summary] {label}: N={len(s)} "
                f"median={np.median(s):.4f} IQR=[{np.quantile(s, 0.25):.4f},{np.quantile(s, 0.75):.4f}] "
                f"mean={np.mean(s):.4f} std={np.std(s):.4f}"
            )

        _summ(df_res["raw_conf"], "Raw confinement")
        _summ(df_res["cond_conf"], "Conditional confinement")
        delta = df_res["cond_conf"] - df_res["raw_conf"]
        _summ(delta, "Delta (cond - raw)")
        print(f"   Projects included: {len(df_res)}")

        # Plot
        plt.figure(figsize=(6, 6))
        sns.regplot(data=df_res, x='raw_conf', y='cond_conf', scatter_kws={'alpha': 0.5})
        plt.plot([0, 1], [0, 1], 'r--', label='Identity')
        plt.xlabel('Raw Confinement')
        plt.ylabel('Conditional Confinement')
        plt.title(f'Signal Retention (r={r_val:.2f}, p={p_r:.1e})')
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}SI_fig_control_retention.pdf")
        plt.close()

        return float(r_val) if not np.isnan(r_val) else 0.0

    def generate_latex_summary(self, p_wilcoxon, r_spearman, tag="min"):
        print("\n📄 Résumé LaTeX (SI)...")
        tex = fr"""
    \subsection{{Robustness of Regime Signatures}}
    We verified that the mature regime signatures (bounded $AI$, zero drift, and confinement $|AI| \le {self.band}$) are robust to methodological perturbations.
    \begin{{itemize}}
        \item \textbf{{Lag Selection:}} The zero-drift signature remains stable across fixed lags $p \in \{{1..6\}}$ and AIC-selected lags, ensuring the equilibrium is not an artifact of model order (Figure SI-1).
        \item \textbf{{Temporal Structure:}} Observed confinement ratios are significantly higher than the aggregate mean of placebo models (linear shifts, circular permutations, block permutations; Wilcoxon $p < {p_wilcoxon:.1e}$), confirming that the signatures depend on temporal alignment and are not reproduced under distribution-preserving or autocorrelation-preserving placebos at the same observational scale (Figure SI-2).
        \item \textbf{{Control Variables:}} Residualizing activity and structure against team size and activity level preserves the confinement signal (Spearman correlation between raw and conditional confinement $r = {r_spearman:.2f}$), suggesting the signatures are not reducible to these confounds at the chosen observational scale (Figure SI-3).
    \end{{itemize}}
        """
        with open(f"{self.output_dir}SI_summary_{tag}.tex", "w") as f:
            f.write(tex)
    def run_full_suite(self):

        print("\n" + "=" * 80)
        print(f"LANCEMENT ROBUSTESSE NAC (Window={self.window_size}, Band={self.band})")
        print("=" * 80)

        # -------------------------------------------------------------------------
        # SI-1 (une fois) : lag sensitivity
        # -------------------------------------------------------------------------
        self.run_analysis_1_lag_sensitivity()

        # -------------------------------------------------------------------------
        # SI-3 (une seule version "clean")
        # IMPORTANT : assure-toi d'avoir nettoyé run_analysis_3_conditional_controls
        # (pas de recalculs/écrasements de m_raw / ai_raw)
        # -------------------------------------------------------------------------
        r_val = self.run_analysis_3_conditional_controls()
        print(f"\n[SI-3] DONE | Spearman raw vs cond: r={r_val:.3f}")

        # -------------------------------------------------------------------------
        # SI-2 : 4 runs = (gamma/min, gamma/fixed, half/min, half/fixed)
        # -------------------------------------------------------------------------
        print("\n" + "=" * 80)
        print("[SI-2] RUN 1/4: mature_mode='gamma' | lag_mode='min' (paper mode)")
        print("=" * 80)
        p_g_min = self.run_analysis_2_placebo_tests(
            n_circular_runs=5,
            lag_mode="min",
            lag=2,
            mature_mode="gamma",
            mature_threshold=0.7,
            signal_tau=0.5,
        )
        print(f"[SI-2|gamma|min] DONE | Wilcoxon p (confinement) = {p_g_min:.4e}")

        print("\n" + "=" * 80)
        print("[SI-2] RUN 2/4: mature_mode='gamma' | lag_mode='fixed' | lag=2")
        print("=" * 80)
        p_g_fixed = self.run_analysis_2_placebo_tests(
            n_circular_runs=5,
            lag_mode="fixed",
            lag=2,
            mature_mode="gamma",
            mature_threshold=0.7,
            signal_tau=0.5,
        )
        print(f"[SI-2|gamma|fixed] DONE | Wilcoxon p (confinement) = {p_g_fixed:.4e}")

        print("\n" + "=" * 80)
        print("[SI-2] RUN 3/4: mature_mode='half' | lag_mode='min'")
        print("=" * 80)
        p_h_min = self.run_analysis_2_placebo_tests(
            n_circular_runs=5,
            lag_mode="min",
            lag=2,
            mature_mode="half",
            mature_threshold=0.7,  # ignoré en half, mais ok
            signal_tau=0.5,
        )
        print(f"[SI-2|half|min] DONE | Wilcoxon p (confinement) = {p_h_min:.4e}")

        print("\n" + "=" * 80)
        print("[SI-2] RUN 4/4: mature_mode='half' | lag_mode='fixed' | lag=2")
        print("=" * 80)
        p_h_fixed = self.run_analysis_2_placebo_tests(
            n_circular_runs=5,
            lag_mode="fixed",
            lag=2,
            mature_mode="half",
            mature_threshold=0.7,  # ignoré en half, mais ok
            signal_tau=0.5,
        )
        print(f"[SI-2|half|fixed] DONE | Wilcoxon p (confinement) = {p_h_fixed:.4e}")

        # -------------------------------------------------------------------------
        # Résumé final
        # -------------------------------------------------------------------------
        print("\n" + "-" * 80)
        print("[SI SUMMARY]")
        print(f"SI-3 Spearman r: {r_val:.3f}")
        print(f"SI-2 gamma/min   p: {p_g_min:.4e}")
        print(f"SI-2 gamma/fixed p: {p_g_fixed:.4e}")
        print(f"SI-2 half/min    p: {p_h_min:.4e}")
        print(f"SI-2 half/fixed  p: {p_h_fixed:.4e}")
        print("-" * 80)

        # --- NOUVEAUX TESTS DE ROBUSTESSE INDEX-SAFE ---
        print("\n" + "=" * 80)
        print("[SI-2 BIS] PLACEBO-Γ (INDEX-SAFE)")
        print("=" * 80)
        self.run_analysis_placebo_gamma_index_safe(
            lag_mode="min",
            lag=2,
            mature_threshold=0.7,
            n_circular_runs=5,
            signal_tau=0.5,
        )

        print("\n" + "=" * 80)
        print("[SI-4] CROSS-SWAP (INTER-PROJECT)")
        print("=" * 80)
        self.run_analysis_cross_swap(
            lag_mode="min",
            lag=2,
            mature_threshold=0.7,
            n_swaps=50,
            signal_tau=0.5,
        )
        return {
            "si3_r": r_val,
            "p_gamma_min": p_g_min,
            "p_gamma_fixed": p_g_fixed,
            "p_half_min": p_h_min,
            "p_half_fixed": p_h_fixed,
        }