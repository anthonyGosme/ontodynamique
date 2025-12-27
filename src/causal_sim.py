"""
DIAGNOSTIC MODULE: Is Causal Symmetry Real or Artifact?
========================================================

This module implements four critical tests to distinguish between:
- Hypothesis A: Methodological circularity (Gamma and Activity are derived from same source)
- Hypothesis B: Statistical ceiling effect (both p-values → 0, so ratio → 1 mechanically)
- Hypothesis C: Real signal (genuine bidirectional constraint emerges)


"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.tsa.stattools import grangercausalitytests
from collections import defaultdict
import warnings

warnings.filterwarnings("ignore")


class CausalSymmetryDiagnostic:
    """
    Diagnostic suite to test whether coupling_ratio = 1.00 is artifact or signal.

    Four tests:
    1. Temporal Permutation: Does shuffling Gamma destroy the coupling?
    2. Phase Shift: Does coupling peak at lag=0?
    3. P-value Saturation: Are both p-values << 0.01 (ceiling effect)?
    4. Residual Independence: Does coupling survive after partialing out common source?
    """

    def __init__(self, all_dataframes, crossover_results):
        # Keep only projects with sufficient data
        self.dfs = {k: v for k, v in all_dataframes.items() if v is not None and len(v) > 40}
        self.crossover = crossover_results
        self.results = {}

    def run_full_diagnostic(self):
        """Execute all diagnostic tests."""
        print("\n" + "=" * 80)
        print("DIAGNOSTIC: IS CAUSAL SYMMETRY REAL OR ARTIFACT?")
        print("=" * 80)

        # Test 1: Temporal Permutation
        self.results['permutation'] = self.test_temporal_permutation()

        # Test 2: Phase Shift
        self.results['phase_shift'] = self.test_phase_shift()

        # Test 3: P-value Saturation Analysis
        self.results['saturation'] = self.test_pvalue_saturation()

        # Test 4: Independence Check (Partial Correlation)
        self.results['independence'] = self.test_residual_independence()

        # Final Verdict
        self._render_verdict()

        # Visualization
        self._plot_diagnostic_summary()

        return self.results

    # =========================================================================
    # TEST 1: TEMPORAL PERMUTATION (Null Model)
    # =========================================================================
    def test_temporal_permutation(self, n_permutations=200):
        """
        Shuffle Gamma temporally while keeping Activity intact.
        If coupling ratio remains high, it's an artifact (shared source).
        If it collapses, the temporal structure matters (real signal).

        Logic:
        - Real causal relationship depends on temporal ordering
        - If Gamma(t) causes Activity(t+1), shuffling Gamma breaks this
        - Artifacts (same-source correlation) survive shuffling
        """
        print("\n" + "-" * 60)
        print("TEST 1: TEMPORAL PERMUTATION (Null Model)")
        print("-" * 60)
        print("H0: Coupling ratio is independent of temporal ordering")
        print("If H0 rejected: Signal is real (temporal structure matters)\n")

        real_ratios = []
        null_ratios_all = []

        for name, df in self.dfs.items():
            if name not in self.crossover:
                continue

            gamma = df['monthly_gamma'].values
            activity = df['total_weight'].values

            if len(gamma) < 30:
                continue

            # Real coupling ratio
            real_ratio = self._compute_coupling_ratio(activity, gamma)
            if real_ratio is None:
                continue
            real_ratios.append(real_ratio)

            # Null distribution: shuffle gamma
            null_ratios = []
            for _ in range(n_permutations):
                gamma_shuffled = np.random.permutation(gamma)
                null_ratio = self._compute_coupling_ratio(activity, gamma_shuffled)
                if null_ratio is not None:
                    null_ratios.append(null_ratio)

            if null_ratios:
                null_ratios_all.extend(null_ratios)

        if not real_ratios or not null_ratios_all:
            print("⚠️ Insufficient data for permutation test")
            return {'verdict': 'INCONCLUSIVE'}

        # Statistics
        real_mean = np.mean(real_ratios)
        null_mean = np.mean(null_ratios_all)
        null_std = np.std(null_ratios_all)

        z_score = (real_mean - null_mean) / (null_std + 1e-9)

        # P-value (one-tailed: real > null)
        p_value = 1 - stats.norm.cdf(z_score)

        print(f"Real Coupling Ratio (mean):  {real_mean:.4f}")
        print(f"Null Coupling Ratio (mean):  {null_mean:.4f}")
        print(f"Null Std Dev:                {null_std:.4f}")
        print(f"Z-Score:                     {z_score:.2f}")
        print(f"P-value (real > null):       {p_value:.4e}")

        if z_score > 3 and p_value < 0.001:
            verdict = "SIGNAL_CONFIRMED"
            print("\n✅ TEMPORAL STRUCTURE MATTERS: Real > Null (z > 3)")
        elif z_score > 2:
            verdict = "SIGNAL_LIKELY"
            print("\n⚠️ SIGNAL LIKELY but not definitive (2 < z < 3)")
        else:
            verdict = "ARTIFACT_SUSPECTED"
            print("\n❌ ARTIFACT SUSPECTED: Shuffling doesn't destroy the pattern")

        return {
            'real_mean': real_mean,
            'null_mean': null_mean,
            'null_std': null_std,
            'z_score': z_score,
            'p_value': p_value,
            'verdict': verdict
        }

    # =========================================================================
    # TEST 2: PHASE SHIFT (Causality Direction Check)
    # =========================================================================
    def test_phase_shift(self, shifts=[-12, -6, -3, 0, 3, 6, 12]):
        """
        Shift Gamma relative to Activity by various lags.
        Real causal coupling should peak at lag=0 and decay symmetrically.
        Artifact would show flat profile across shifts.

        Logic:
        - If Gamma and Activity are causally linked, alignment matters
        - Shifting one series relative to the other breaks alignment
        - Artifacts (spurious correlation) are shift-invariant
        """
        print("\n" + "-" * 60)
        print("TEST 2: PHASE SHIFT (Temporal Alignment)")
        print("-" * 60)
        print("Real causality peaks at lag=0; artifacts are shift-invariant\n")

        shift_results = defaultdict(list)

        for name, df in self.dfs.items():
            gamma = df['monthly_gamma'].values
            activity = df['total_weight'].values

            if len(gamma) < 40:
                continue

            for shift in shifts:
                # Shift gamma relative to activity
                if shift > 0:
                    gamma_shifted = gamma[shift:]
                    activity_aligned = activity[:-shift]
                elif shift < 0:
                    gamma_shifted = gamma[:shift]
                    activity_aligned = activity[-shift:]
                else:
                    gamma_shifted = gamma
                    activity_aligned = activity

                if len(gamma_shifted) < 20:
                    continue

                ratio = self._compute_coupling_ratio(activity_aligned, gamma_shifted)
                if ratio is not None:
                    shift_results[shift].append(ratio)

        # Compute means per shift
        shift_means = {}
        for shift in shifts:
            if shift_results[shift]:
                shift_means[shift] = np.mean(shift_results[shift])

        if not shift_means:
            print("⚠️ Insufficient data for phase shift test")
            return {'verdict': 'INCONCLUSIVE'}

        # Analysis
        ratio_at_zero = shift_means.get(0, 0)
        ratio_at_plus6 = shift_means.get(6, 0)
        ratio_at_minus6 = shift_means.get(-6, 0)

        # Peak should be at 0
        is_peaked_at_zero = ratio_at_zero > ratio_at_plus6 and ratio_at_zero > ratio_at_minus6

        # Decay rate (how much does coupling drop when misaligned)
        avg_decay = ((ratio_at_zero - ratio_at_plus6) + (ratio_at_zero - ratio_at_minus6)) / 2

        print(f"{'Shift (months)':<15} | {'Coupling Ratio':<15}")
        print("-" * 35)
        for shift in sorted(shift_means.keys()):
            marker = " ← PEAK" if shift == 0 else ""
            print(f"{shift:<15} | {shift_means[shift]:<15.4f}{marker}")

        print(f"\nPeaked at lag=0: {is_peaked_at_zero}")
        print(f"Average decay from peak: {avg_decay:.4f}")

        if is_peaked_at_zero and avg_decay > 0.05:
            verdict = "CAUSAL_STRUCTURE_CONFIRMED"
            print("\n✅ CAUSAL STRUCTURE: Coupling peaks at correct alignment")
        elif is_peaked_at_zero:
            verdict = "WEAK_CAUSAL_STRUCTURE"
            print("\n⚠️ WEAK STRUCTURE: Peak at 0 but shallow decay")
        else:
            verdict = "NO_CAUSAL_STRUCTURE"
            print("\n❌ NO CAUSAL STRUCTURE: Pattern is shift-invariant (artifact)")

        return {
            'shift_means': shift_means,
            'peaked_at_zero': is_peaked_at_zero,
            'decay_rate': avg_decay,
            'verdict': verdict
        }

    # =========================================================================
    # TEST 3: P-VALUE SATURATION ANALYSIS
    # =========================================================================
    def test_pvalue_saturation(self):
        """
        Examine raw p-values from Granger tests.
        If both p-values are << 0.01, ratio → 1 is ceiling effect, not symmetry.
        True symmetry: p_ag ≈ p_ga at moderate values (e.g., both around 0.03).

        Logic:
        - Coupling ratio = min(1-p_ag, 1-p_ga) / max(1-p_ag, 1-p_ga)
        - If p_ag = 0.001 and p_ga = 0.002, ratio ≈ 0.999/0.998 ≈ 1.00
        - This is statistical saturation, not meaningful symmetry
        - True symmetry: both p-values are similar AND in discriminative range
        """
        print("\n" + "-" * 60)
        print("TEST 3: P-VALUE SATURATION ANALYSIS")
        print("-" * 60)
        print("Ceiling effect: both p << 0.01 → ratio ≈ 1 mechanically")
        print("True symmetry: p_ag ≈ p_ga at moderate levels\n")

        p_values_ag = []
        p_values_ga = []
        ratios = []

        for name, df in self.dfs.items():
            gamma = df['monthly_gamma'].values
            activity = df['total_weight'].values

            if len(gamma) < 30:
                continue

            p_ag, p_ga = self._get_granger_pvalues(activity, gamma)
            if p_ag is not None and p_ga is not None:
                p_values_ag.append(p_ag)
                p_values_ga.append(p_ga)

                # Compute ratio from strengths
                s_ag = 1 - p_ag
                s_ga = 1 - p_ga
                ratio = min(s_ag, s_ga) / (max(s_ag, s_ga) + 1e-9)
                ratios.append(ratio)

        if not p_values_ag:
            print("⚠️ Insufficient data for saturation analysis")
            return {'verdict': 'INCONCLUSIVE'}

        p_ag_arr = np.array(p_values_ag)
        p_ga_arr = np.array(p_values_ga)

        # Saturation metrics
        pct_saturated_ag = np.mean(p_ag_arr < 0.01) * 100
        pct_saturated_ga = np.mean(p_ga_arr < 0.01) * 100
        pct_both_saturated = np.mean((p_ag_arr < 0.01) & (p_ga_arr < 0.01)) * 100

        # True symmetry metric: |log(p_ag) - log(p_ga)| when both are moderate
        moderate_mask = (p_ag_arr > 0.01) & (p_ga_arr > 0.01) & (p_ag_arr < 0.5) & (p_ga_arr < 0.5)
        if np.sum(moderate_mask) > 5:
            log_diff = np.abs(np.log10(p_ag_arr[moderate_mask]) - np.log10(p_ga_arr[moderate_mask]))
            symmetry_at_moderate = np.mean(log_diff)
        else:
            symmetry_at_moderate = np.nan

        print(f"P-values saturated (< 0.01):")
        print(f"  Activity → Gamma: {pct_saturated_ag:.1f}%")
        print(f"  Gamma → Activity: {pct_saturated_ga:.1f}%")
        print(f"  BOTH saturated:   {pct_both_saturated:.1f}%")
        print(f"\nMean coupling ratio:     {np.mean(ratios):.4f}")
        print(f"Mean ratio when BOTH saturated: ", end="")

        both_sat_mask = (p_ag_arr < 0.01) & (p_ga_arr < 0.01)
        if np.sum(both_sat_mask) > 0:
            sat_ratios = np.array(ratios)[both_sat_mask]
            print(f"{np.mean(sat_ratios):.4f}")
        else:
            print("N/A")

        # Verdict
        if pct_both_saturated > 70:
            verdict = "CEILING_EFFECT_DOMINANT"
            print("\n❌ CEILING EFFECT: Most observations have both p << 0.01")
            print("   The ratio → 1 is mechanical, not meaningful symmetry")
        elif pct_both_saturated > 40:
            verdict = "CEILING_EFFECT_PARTIAL"
            print("\n⚠️ PARTIAL CEILING: Many observations saturated")
            print("   Interpret ratio = 1 with caution")
        else:
            verdict = "NO_CEILING_EFFECT"
            print("\n✅ NO CEILING EFFECT: P-values are in discriminative range")

        return {
            'pct_saturated_ag': pct_saturated_ag,
            'pct_saturated_ga': pct_saturated_ga,
            'pct_both_saturated': pct_both_saturated,
            'mean_ratio': np.mean(ratios),
            'symmetry_at_moderate': symmetry_at_moderate,
            'verdict': verdict
        }

    # =========================================================================
    # TEST 4: RESIDUAL INDEPENDENCE (Partial Correlation)
    # =========================================================================
    def test_residual_independence(self):
        """
        Regress out the common commit-count from both Gamma and Activity.
        If coupling persists in residuals, it's not purely driven by shared source.

        Logic:
        - Gamma is computed from commits (files survived, lines retained)
        - Activity is computed from commits (total weight)
        - Both share 'files_touched' as common factor
        - If coupling is due to this shared source, residualizing removes it
        - If coupling persists, there's genuine bidirectional constraint
        """
        print("\n" + "-" * 60)
        print("TEST 4: RESIDUAL INDEPENDENCE (Partial Correlation)")
        print("-" * 60)
        print("Regress out files_touched (common source) from both series")
        print("If coupling persists in residuals → not purely circular\n")

        raw_ratios = []
        residual_ratios = []

        for name, df in self.dfs.items():
            gamma = df['monthly_gamma'].values
            activity = df['total_weight'].values

            # Common source: files_touched (proxy for commit intensity)
            if 'files_touched' not in df.columns:
                continue
            common = df['files_touched'].values

            if len(gamma) < 30 or np.std(common) == 0:
                continue

            # Raw ratio
            raw_ratio = self._compute_coupling_ratio(activity, gamma)
            if raw_ratio is None:
                continue
            raw_ratios.append(raw_ratio)

            # Residualize both series w.r.t. common source
            gamma_resid = self._residualize(gamma, common)
            activity_resid = self._residualize(activity, common)

            # Residual ratio
            resid_ratio = self._compute_coupling_ratio(activity_resid, gamma_resid)
            if resid_ratio is not None:
                residual_ratios.append(resid_ratio)

        if not raw_ratios or not residual_ratios:
            print("⚠️ Insufficient data for residual test")
            return {'verdict': 'INCONCLUSIVE'}

        raw_mean = np.mean(raw_ratios)
        resid_mean = np.mean(residual_ratios)
        retention = resid_mean / raw_mean if raw_mean > 0 else 0

        print(f"Raw Coupling Ratio (mean):      {raw_mean:.4f}")
        print(f"Residual Coupling Ratio (mean): {resid_mean:.4f}")
        print(f"Retention after partialing:     {retention:.1%}")

        # Statistical test: paired comparison
        if len(raw_ratios) == len(residual_ratios) and len(raw_ratios) >= 5:
            t_stat, p_val = stats.ttest_rel(raw_ratios, residual_ratios)
            print(f"Paired t-test (raw vs resid):   t={t_stat:.2f}, p={p_val:.4f}")

        if retention > 0.7:
            verdict = "COUPLING_INDEPENDENT_OF_SOURCE"
            print("\n✅ COUPLING PERSISTS: Not driven by shared commit source")
        elif retention > 0.4:
            verdict = "COUPLING_PARTIALLY_INDEPENDENT"
            print("\n⚠️ PARTIAL INDEPENDENCE: Some coupling is source-driven")
        else:
            verdict = "COUPLING_SOURCE_DEPENDENT"
            print("\n❌ SOURCE-DEPENDENT: Coupling mostly from shared commits")

        return {
            'raw_mean': raw_mean,
            'residual_mean': resid_mean,
            'retention': retention,
            'verdict': verdict
        }

    # =========================================================================
    # HELPER METHODS
    # =========================================================================

    def _compute_coupling_ratio(self, activity, gamma, max_lag=3):
        """Compute coupling ratio from Granger test."""
        p_ag, p_ga = self._get_granger_pvalues(activity, gamma, max_lag)
        if p_ag is None or p_ga is None:
            return None

        s_ag = 1 - p_ag
        s_ga = 1 - p_ga

        return min(s_ag, s_ga) / (max(s_ag, s_ga) + 1e-9)

    def _get_granger_pvalues(self, activity, gamma, max_lag=3):
        """Get raw p-values from Granger causality tests."""
        try:
            # Normalize
            if np.std(activity) == 0 or np.std(gamma) == 0:
                return None, None

            act_norm = (activity - np.mean(activity)) / np.std(activity)
            gam_norm = (gamma - np.mean(gamma)) / np.std(gamma)

            data = pd.DataFrame({'act': act_norm, 'gam': gam_norm})

            # Test 1: Act → Gamma
            g1 = grangercausalitytests(data[['gam', 'act']], maxlag=max_lag, verbose=False)
            p_ag = min([g1[lag][0]['ssr_ftest'][1] for lag in range(1, max_lag + 1)])

            # Test 2: Gamma → Act
            g2 = grangercausalitytests(data[['act', 'gam']], maxlag=max_lag, verbose=False)
            p_ga = min([g2[lag][0]['ssr_ftest'][1] for lag in range(1, max_lag + 1)])

            return p_ag, p_ga

        except Exception:
            return None, None

    def _residualize(self, y, x):
        """Regress y on x and return residuals."""
        x = np.array(x).reshape(-1, 1)
        y = np.array(y)

        # Add constant
        X = np.column_stack([np.ones(len(x)), x])

        try:
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
            y_pred = X @ beta
            residuals = y - y_pred
            return residuals
        except Exception:
            return y

    # =========================================================================
    # VERDICT & VISUALIZATION
    # =========================================================================

    def _render_verdict(self):
        """Synthesize all tests into final verdict."""
        print("\n" + "=" * 80)
        print("FINAL DIAGNOSTIC VERDICT")
        print("=" * 80)

        verdicts = {
            'permutation': self.results.get('permutation', {}).get('verdict', 'N/A'),
            'phase_shift': self.results.get('phase_shift', {}).get('verdict', 'N/A'),
            'saturation': self.results.get('saturation', {}).get('verdict', 'N/A'),
            'independence': self.results.get('independence', {}).get('verdict', 'N/A'),
        }

        print("\nTest Results Summary:")
        print(f"  1. Temporal Permutation: {verdicts['permutation']}")
        print(f"  2. Phase Shift:          {verdicts['phase_shift']}")
        print(f"  3. P-value Saturation:   {verdicts['saturation']}")
        print(f"  4. Residual Independence:{verdicts['independence']}")

        # Scoring
        score = 0
        max_score = 4

        if 'CONFIRMED' in verdicts['permutation'] or 'LIKELY' in verdicts['permutation']:
            score += 1
        if 'CONFIRMED' in verdicts['phase_shift']:
            score += 1
        if 'NO_CEILING' in verdicts['saturation']:
            score += 1
        if 'INDEPENDENT' in verdicts['independence']:
            score += 1

        print(f"\nOVERALL SCORE: {score}/{max_score}")

        if score >= 3:
            print("\n✅ VERDICT: CAUSAL SYMMETRY IS LIKELY REAL")
            print("   The coupling ratio reflects genuine bidirectional constraint.")
            print("   However, consider reporting as 'near-symmetric' rather than 'exact'.")
        elif score >= 2:
            print("\n⚠️ VERDICT: SIGNAL IS MIXED")
            print("   Some evidence for real coupling, but artifacts contribute.")
            print("   Recommend: Report coupling trend, not exact ratio value.")
        else:
            print("\n❌ VERDICT: ARTIFACT SUSPECTED")
            print("   The ratio = 1.00 is likely driven by:")
            if 'CEILING' in verdicts['saturation']:
                print("   - Statistical ceiling effect (both p-values saturated)")
            if 'DEPENDENT' in verdicts['independence']:
                print("   - Circularity (both derived from same commit stream)")
            print("   Recommend: Do not report exact ratio; describe trend only.")

        self.results['final_score'] = score
        self.results['final_verdict'] = 'REAL' if score >= 3 else ('MIXED' if score >= 2 else 'ARTIFACT')

    def _plot_diagnostic_summary(self):
        """Create diagnostic visualization."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # ===== Plot 1: Permutation Test =====
        ax1 = axes[0, 0]
        perm = self.results.get('permutation', {})
        if 'real_mean' in perm:
            # Simulate null distribution for visualization
            null_mean = perm['null_mean']
            null_std = perm['null_std']
            x = np.linspace(null_mean - 4 * null_std, null_mean + 4 * null_std, 100)
            y = stats.norm.pdf(x, null_mean, null_std)
            ax1.fill_between(x, y, alpha=0.3, color='gray', label='Null Distribution')
            ax1.axvline(perm['real_mean'], color='red', linewidth=2, label=f"Real: {perm['real_mean']:.3f}")
            ax1.axvline(null_mean, color='gray', linestyle='--', label=f"Null Mean: {null_mean:.3f}")
            ax1.set_xlabel('Coupling Ratio')
            ax1.set_ylabel('Density')
            ax1.set_title(f"Test 1: Temporal Permutation\nZ = {perm.get('z_score', 0):.2f}")
            ax1.legend()
        else:
            ax1.text(0.5, 0.5, 'Insufficient Data', ha='center', va='center', transform=ax1.transAxes)
            ax1.set_title('Test 1: Temporal Permutation')

        # ===== Plot 2: Phase Shift =====
        ax2 = axes[0, 1]
        shift = self.results.get('phase_shift', {})
        if 'shift_means' in shift:
            shifts = sorted(shift['shift_means'].keys())
            ratios = [shift['shift_means'][s] for s in shifts]
            colors = ['red' if s == 0 else 'steelblue' for s in shifts]
            ax2.bar(shifts, ratios, color=colors, edgecolor='black')
            ax2.axhline(shift['shift_means'].get(0, 0), color='red', linestyle='--', alpha=0.5)
            ax2.set_xlabel('Lag (months)')
            ax2.set_ylabel('Coupling Ratio')
            ax2.set_title(f"Test 2: Phase Shift\nPeaked at 0: {shift.get('peaked_at_zero', False)}")
        else:
            ax2.text(0.5, 0.5, 'Insufficient Data', ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Test 2: Phase Shift')

        # ===== Plot 3: P-value Saturation =====
        ax3 = axes[1, 0]
        sat = self.results.get('saturation', {})
        if 'pct_both_saturated' in sat:
            categories = ['Act→Γ\nSaturated', 'Γ→Act\nSaturated', 'BOTH\nSaturated']
            values = [sat['pct_saturated_ag'], sat['pct_saturated_ga'], sat['pct_both_saturated']]
            colors = ['#3498db', '#e74c3c', '#9b59b6']
            ax3.bar(categories, values, color=colors, edgecolor='black')
            ax3.axhline(50, color='red', linestyle='--', label='50% threshold')
            ax3.set_ylabel('Percentage (%)')
            ax3.set_title(f"Test 3: P-value Saturation\n(< 0.01 = saturated)")
            ax3.set_ylim(0, 100)
            ax3.legend()
        else:
            ax3.text(0.5, 0.5, 'Insufficient Data', ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Test 3: P-value Saturation')

        # ===== Plot 4: Residual Independence =====
        ax4 = axes[1, 1]
        ind = self.results.get('independence', {})
        if 'raw_mean' in ind:
            categories = ['Raw\nCoupling', 'Residual\nCoupling']
            values = [ind['raw_mean'], ind['residual_mean']]
            colors = ['#3498db', '#27ae60']
            bars = ax4.bar(categories, values, color=colors, edgecolor='black')
            ax4.set_ylabel('Coupling Ratio')
            ax4.set_title(f"Test 4: Residual Independence\nRetention: {ind['retention']:.1%}")
            # Add retention arrow
            ax4.annotate('', xy=(1, ind['residual_mean']), xytext=(0, ind['raw_mean']),
                         arrowprops=dict(arrowstyle='->', color='black', lw=2))
        else:
            ax4.text(0.5, 0.5, 'Insufficient Data', ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Test 4: Residual Independence')

        # Overall title
        final_verdict = self.results.get('final_verdict', 'N/A')
        final_score = self.results.get('final_score', 0)
        fig.suptitle(f'CAUSAL SYMMETRY DIAGNOSTIC\nVerdict: {final_verdict} (Score: {final_score}/4)',
                     fontsize=14, fontweight='bold')

        plt.tight_layout()
        plt.savefig('causal_symmetry_diagnostic.png', dpi=150)
        print(f"\n✅ Diagnostic plot saved: causal_symmetry_diagnostic.png")
        plt.close()


# =============================================================================
# STANDALONE RUNNER
# =============================================================================

def run_causal_symmetry_diagnostic(all_dataframes, crossover_results):
    """
    Main entry point for diagnostic.
    Call this from your main script after Phase 7.

    Parameters:
    -----------
    all_dataframes : dict
        Dictionary of {project_name: DataFrame} from omega_v36 analysis
    crossover_results : dict
        Dictionary of {project_name: rolling_granger_results} from Phase 7

    Returns:
    --------
    dict with keys:
        - 'permutation': Test 1 results
        - 'phase_shift': Test 2 results
        - 'saturation': Test 3 results
        - 'independence': Test 4 results
        - 'final_score': 0-4
        - 'final_verdict': 'REAL', 'MIXED', or 'ARTIFACT'
    """
    diagnostic = CausalSymmetryDiagnostic(all_dataframes, crossover_results)
    results = diagnostic.run_full_diagnostic()
    return results


# =============================================================================
# USAGE EXAMPLE
# =============================================================================

if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    CAUSAL SYMMETRY DIAGNOSTIC MODULE                         ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  This module tests whether coupling_ratio = 1.00 is real or artifact.        ║
║                                                                              ║                                                                 ║
║                                                                              ║
║  INTERPRETATION:                                                             ║
║  ---------------                                                             ║
║  Score 3-4/4 : Signal is real. Report "near-symmetric coupling"              ║
║  Score 2/4   : Signal is mixed. Report "convergence toward symmetry"         ║
║  Score 0-1/4 : Artifact suspected. Do not report exact ratio                 ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)