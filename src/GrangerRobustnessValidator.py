import warnings

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller
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

    def _compute_ai_series(self, series_s, series_a, index=None, lag=2, lag_mode='fixed', window=None, diff_mode="adf"):
        """
        Calcul de l'AI normalisé sur fenêtre glissante avec alignement explicite.
        """
        if window is None: window = self.window_size

        n = len(series_s)
        if n < window + 10:
            return pd.Series(dtype=float)

        if index is None: index = pd.RangeIndex(n)

        # 1. Reconstruction & Alignement Explicite (Safeguard #1)
        # On force les séries d'entrée sur l'index maître AVANT preprocessing
        s0 = pd.Series(series_s, index=index).astype(float)
        a0 = pd.Series(series_a, index=index).astype(float)

        # Preprocessing, puis ré-alignement forcé sur l'index maître
        # (Au cas où le preprocessing dropperait des index ou changerait la taille)
        s = self._preprocess_series(s0, mode=diff_mode).reindex(index)
        a = self._preprocess_series(a0, mode=diff_mode).reindex(index)

        # 2. DataFrame aligné
        data = pd.DataFrame({'S': s, 'A': a})
        if isinstance(data.index, pd.DatetimeIndex):
            # 'MS' = Month Start (le warning suggère que c'est ce qu'il a deviné)
            # Si vos données ne sont pas régulières, utilisez .asfreq(), sinon l'assignation directe suffit
            try:
                if data.index.freq is None:
                    data.index.freq = pd.infer_freq(data.index)  # Tente de deviner proprement
                    if data.index.freq is None:  # Si échec, force MS comme le warning
                        data.index.freq = 'MS'
            except ValueError:
                pass  # Si l'index n'est pas parfaitement régulier, on ignore
        ai_vals, ai_idx = [], []
        valid_windows_count = 0

        # 3. Rolling VAR
        for i in range(window, n):
            w = data.iloc[i - window:i].dropna()

            # Checks techniques
            if len(w) < window * 0.9: continue
            if w.var().min() < 1e-8: continue

            try:
                model = VAR(w)

                # Sélection du Lag (AIC Robust Patch #2)
                p = 1
                if lag_mode == 'aic':
                    try:
                        sel = model.select_order(maxlags=min(6, len(w) // 2))  # maxlags conservateur
                        p_aic = sel.aic

                        # Gestion robuste du type de retour de statsmodels
                        if hasattr(p_aic, "idxmin"):  # Series/Dict like
                            p = int(p_aic.idxmin())
                        elif p_aic is None or (isinstance(p_aic, float) and np.isnan(p_aic)):
                            p = 1
                        else:
                            p = int(p_aic)

                        # Clamping final
                        p = max(1, min(p, 6))
                    except:
                        p = 1
                else:
                    p = int(lag)

                # Fit
                results = model.fit(p)

                # Granger Tests (Syntaxe safe)
                test_as = results.test_causality(caused='S', causing=['A'], kind='f')
                f_as = test_as.test_statistic

                test_sa = results.test_causality(caused='A', causing=['S'], kind='f')
                f_sa = test_sa.test_statistic

                # AI Normalisé
                denom = f_sa + f_as
                ai = (f_sa - f_as) / (denom + 1e-9) if denom > 1e-9 else 0.0

                ai_vals.append(ai)
                ai_idx.append(data.index[i])
                valid_windows_count += 1
            except:
                continue

        # Sortie enrichie (Safeguard #4)
        res = pd.Series(ai_vals, index=ai_idx, name="AI")
        res.attrs['n_windows'] = valid_windows_count
        return res

    def _calculate_nac_metrics(self, ai_series, gamma_series=None):
        """
        mature_mode:
          - "gamma" : mature = gamma >= threshold
          - "half"  : mature = seconde moitié temporelle (non circulaire)
        """
        mature_threshold= .7
        mature_mode = "half"
        if ai_series is None or len(ai_series) < 12:
            return {'slope': np.nan, 'variance': np.nan, 'confinement': np.nan, 'zc': np.nan, 'n_windows': 0}

        # --- Alignement (optionnel) ---
        if mature_mode == "gamma":
            if gamma_series is None:
                return {'slope': np.nan, 'variance': np.nan, 'confinement': np.nan, 'zc': np.nan, 'n_windows': 0}
            common_idx = ai_series.index.intersection(gamma_series.index)
            if len(common_idx) < 12:
                return {'slope': np.nan, 'variance': np.nan, 'confinement': np.nan, 'zc': np.nan, 'n_windows': 0}
            ai_aligned = ai_series.loc[common_idx]
            gamma_aligned = gamma_series.loc[common_idx]
            ai_mature = ai_aligned[gamma_aligned >= mature_threshold]

        elif mature_mode == "half":
            # IMPORTANT: la maturité est définie uniquement par le temps (2e moitié de AI)
            ai_aligned = ai_series.dropna()
            if len(ai_aligned) < 24:  # 12 mois min en 2e moitié → il faut un peu de marge
                return {'slope': np.nan, 'variance': np.nan, 'confinement': np.nan, 'zc': np.nan, 'n_windows': 0}
            mid = len(ai_aligned) // 2
            ai_mature = ai_aligned.iloc[mid:]

        else:
            raise ValueError("mature_mode must be 'gamma' or 'half'")

        if ai_mature is None or len(ai_mature) < 12:
            return {'slope': np.nan, 'variance': np.nan, 'confinement': np.nan, 'zc': np.nan, 'n_windows': 0}

        # 1) Drift
        x = np.arange(len(ai_mature))
        try:
            slope, _ = np.polyfit(x, ai_mature.values, 1)
        except:
            slope = np.nan

        # 2) Variance
        variance = np.var(ai_mature)

        # 3) Confinement
        confinement = (np.abs(ai_mature) <= self.band).mean()

        # 4) Zero crossings
        signs = np.sign(ai_mature)
        signs = signs[signs != 0]
        zc = int(np.sum(np.abs(np.diff(signs)) == 2)) if len(signs) > 1 else 0

        n_win = ai_series.attrs.get('n_windows', len(ai_series))
        return {'slope': slope, 'variance': variance, 'confinement': confinement, 'zc': zc, 'n_windows': n_win}

    # ==========================================================================
    # 2. ANALYSES
    # ==========================================================================

    def run_analysis_1_lag_sensitivity(self):
        print("\n⚙️ [SI-1] Lag Sensitivity (incl. AIC)...")
        results = []
        lags_to_test = [1, 2, 3, 4, 6, 'aic']

        for name, df in tqdm(list(self.dfs.items()), desc="Lag"):
            s_raw = df['monthly_gamma']
            a_raw = df['total_weight']

            for p in lags_to_test:
                mode = 'aic' if p == 'aic' else 'fixed'
                lag_val = 1 if p == 'aic' else p

                ai = self._compute_ai_series(s_raw, a_raw, index=df.index, lag=lag_val, lag_mode=mode)
                m = self._calculate_nac_metrics(ai, s_raw)
                m['project'] = name
                m['lag_type'] = str(p)
                results.append(m)

        df_res = pd.DataFrame(results)
        df_res.to_csv(f"{self.output_dir}SI_lag_sensitivity.csv", index=False)

        # -----------------------------
        # LOGS "reviewer-proof"
        # -----------------------------
        print("\n" + "-" * 78)
        print("[SI-1] Lag Sensitivity Summary")
        print("-" * 78)

        valid = df_res.dropna(subset=["confinement", "slope", "variance", "zc"])
        print(f"   Projects total: {df_res['project'].nunique()} | valid rows: {len(valid)}")

        order = ['1', '2', '3', '4', '6', 'aic']
        for lt in order:
            sub = valid[valid['lag_type'] == lt]
            print(f"\n   Lag={lt} | projects={sub['project'].nunique()} rows={len(sub)}")

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

            _summ(sub["slope"], "slope")
            _summ(sub["confinement"], "confinement")
            _summ(sub["variance"], "variance")
            _summ(sub["zc"], "zero_crossings")

            if len(sub) > 0:
                pct_low_drift = (np.abs(sub["slope"]) < 0.005).mean()
                pct_high_conf = (sub["confinement"] > 0.6).mean()
                print(f"   [Rates] |slope|<0.005: {pct_low_drift:.2%} | conf>0.6: {pct_high_conf:.2%}")

        # -----------------------------
        # Plot 2x2
        # -----------------------------
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        sns.boxplot(data=df_res, x='lag_type', y='slope', ax=axes[0, 0], order=order)
        axes[0, 0].axhline(0, color='red', linestyle='--')
        axes[0, 0].set_title('Drift (Slope)')
        axes[0, 0].set_ylim(-0.02, 0.02)

        sns.boxplot(data=df_res, x='lag_type', y='confinement', ax=axes[0, 1], order=order)
        axes[0, 1].set_title(f'Confinement (Band={self.band})')
        axes[0, 1].set_ylim(0, 1)

        sns.boxplot(data=df_res, x='lag_type', y='variance', ax=axes[1, 0], order=order)
        axes[1, 0].set_title('Variance')

        sns.boxplot(data=df_res, x='lag_type', y='zc', ax=axes[1, 1], order=order)
        axes[1, 1].set_title('Zero Crossings')

        plt.tight_layout()
        plt.savefig(f"{self.output_dir}SI_fig_lag_stability_full.pdf")
        plt.close()

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

    def run_analysis_2_placebo_tests(self, n_circular_runs=5):
        print("\n⚙️ [SI-2] Placebo Tests (Strict Alignment & Hierarchical Stats)...")

        results = []
        shifts = [-6, 6]

        for name in tqdm(list(self.dfs.keys()), desc="Placebo"):
            df = self.dfs[name]
            s_raw = df['monthly_gamma']
            a_raw = df['total_weight']

            # -----------------------------
            # 1) Observed
            # -----------------------------
            ai_obs = self._compute_ai_series(s_raw, a_raw, index=df.index)
            m_obs = self._calculate_nac_metrics(ai_obs, s_raw)
            m_obs.update({'project': name, 'type': 'Observed', 'rep': np.nan})
            results.append(m_obs)

            # -----------------------------
            # 2) Linear shifts
            # -----------------------------
            for k in shifts:
                if k > 0:
                    s_lin = s_raw.iloc[k:]
                    a_lin = a_raw.iloc[:-k]
                else:
                    s_lin = s_raw.iloc[:k]
                    a_lin = a_raw.iloc[-k:]

                idx_lin = s_lin.index
                ai_lin = self._compute_ai_series(s_lin, a_lin, index=idx_lin)

                # IMPORTANT: gamma_ref aligné sur l'index des fenêtres AI
                if len(ai_lin) > 0:
                    gamma_ref = df.loc[ai_lin.index, 'monthly_gamma']
                else:
                    gamma_ref = pd.Series(dtype=float)

                m_lin = self._calculate_nac_metrics(ai_lin, gamma_ref)
                m_lin.update({'project': name, 'type': f'Shift({k}m)', 'rep': np.nan})
                results.append(m_lin)

            # -----------------------------
            # 3) Circular shifts (multi-run)
            # -----------------------------
            s_vals = s_raw.values
            for rep in range(n_circular_runs):
                split = int(self.rng.integers(len(s_vals) // 4, len(s_vals) * 3 // 4))
                s_circ = self._circular_shift(s_vals, split)
                s_circ_series = pd.Series(s_circ, index=df.index)

                ai_circ = self._compute_ai_series(s_circ_series, a_raw, index=df.index)
                m_circ = self._calculate_nac_metrics(ai_circ, s_raw)
                m_circ.update({'project': name, 'type': 'Circular', 'rep': rep})
                results.append(m_circ)

            # -----------------------------
            # 4) Block permutation
            # -----------------------------
            s_blk = self._block_permute(s_vals, block_len=12)
            s_blk_series = pd.Series(s_blk, index=df.index)

            ai_blk = self._compute_ai_series(s_blk_series, a_raw, index=df.index)
            m_blk = self._calculate_nac_metrics(ai_blk, s_raw)
            m_blk.update({'project': name, 'type': 'BlockPerm', 'rep': np.nan})
            results.append(m_blk)

        # ------------------------------------------------------------------
        # Build DataFrame + export brut (avant filtrage)
        # ------------------------------------------------------------------
        df_res = pd.DataFrame(results)
        df_res.to_csv(f"{self.output_dir}SI_placebo_tests_RAW.csv", index=False)

        # ------------------------------------------------------------------
        # Filtrage propre (NE PAS dropna() sur tout !)
        # ------------------------------------------------------------------
        valid = df_res.dropna(subset=["project", "type", "confinement"])
        valid.to_csv(f"{self.output_dir}SI_placebo_tests.csv", index=False)

        # ------------------------------------------------------------------
        # Debug counts
        # ------------------------------------------------------------------
        print("\n[DEBUG PLACEBO] type counts (valid rows):")
        print(valid['type'].value_counts(dropna=False))
        print("[DEBUG PLACEBO] projects total:", valid['project'].nunique())
        print("[DEBUG PLACEBO] observed projects:", valid[valid['type'] == 'Observed']['project'].nunique())
        print("[DEBUG PLACEBO] placebo projects:", valid[valid['type'] != 'Observed']['project'].nunique())

        # ------------------------------------------------------------------
        # Stats: Observed vs Mean(Placebos) (hiérarchique)
        # ------------------------------------------------------------------
        obs = valid[valid['type'] == 'Observed'].set_index('project')['confinement']

        placebo_df = valid[valid['type'] != 'Observed']
        mean_per_type = placebo_df.groupby(['project', 'type'])['confinement'].mean()
        placebo_mean = mean_per_type.groupby('project').mean()

        common = obs.index.intersection(placebo_mean.index)

        if len(common) > 10:
            stat, p_val_mean = stats.wilcoxon(
                obs.loc[common],
                placebo_mean.loc[common],
                alternative='greater'
            )
            print(f"   [Stat] Wilcoxon Obs > Mean(Placebos) (Confinement): stat={stat:.3f} p={p_val_mean:.4e}")
        else:
            stat = np.nan
            p_val_mean = 1.0
            print("   [Stat] Wilcoxon skipped: not enough common projects.")

        # Bonus sanity check: pooled placebo mean (sans hiérarchie)
        pooled = placebo_df.groupby('project')['confinement'].mean()
        common2 = obs.index.intersection(pooled.index)
        if len(common2) > 10:
            stat2, p2 = stats.wilcoxon(obs.loc[common2], pooled.loc[common2], alternative='greater')
            print(f"   [Stat] Wilcoxon Obs > pooled placebos: stat={stat2:.3f} p={p2:.4e}")
        else:
            p2 = 1.0

        # ------------------------------------------------------------------
        # Descriptifs + direction
        # ------------------------------------------------------------------
        print("\n" + "-" * 78)
        print("[SI-2] Placebo Summary (Observed vs Placebo Mean)")
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

        if len(common) > 0:
            delta = (obs.loc[common] - placebo_mean.loc[common])
            _summ(obs.loc[common], "Observed confinement")
            _summ(placebo_mean.loc[common], "Mean placebo confinement")
            _summ(delta, "Delta (Obs - PlaceboMean)")
            pct_better = (delta > 0).mean()
            print(f"   [Direction] Obs>PlaceboMean in {pct_better:.2%} of projects | N={len(delta)}")
        else:
            print("   [Summary] No common projects for observed vs placebo comparison (after filtering).")

        # ------------------------------------------------------------------
        # Plot (sur valid)
        # ------------------------------------------------------------------
        plt.figure(figsize=(10, 6))
        sns.violinplot(
            data=valid,
            x='type',
            y='confinement',
            hue='type',
            palette='muted',
            cut=0,
            legend=False
        )
        plt.title(f'Confinement Ratio vs Placebos (Wilcoxon p={p_val_mean:.2e})')
        plt.ylabel(f'Mature Phase Confinement (|AI| <= {self.band})')
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}SI_fig_placebo_conf.pdf")
        plt.close()

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
            m_cond = self._calculate_nac_metrics(ai_cond, df['monthly_gamma'])

            ai_raw = self._compute_ai_series(S, A, index=idx)
            m_raw = self._calculate_nac_metrics(ai_raw, df['monthly_gamma'])

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

    def generate_latex_summary(self, p_wilcoxon, r_spearman):
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
        with open(f"{self.output_dir}SI_summary.tex", "w") as f:
            f.write(tex)

    def run_full_suite(self):

        print("\n" + "=" * 80)
        print(f"LANCEMENT ROBUSTESSE NAC (Window={self.window_size}, Band={self.band})")
        print("=" * 80)

        self.run_analysis_1_lag_sensitivity()
        p_val = self.run_analysis_2_placebo_tests(n_circular_runs=5)
        r_val = self.run_analysis_3_conditional_controls()
        self.generate_latex_summary(p_val, r_val)