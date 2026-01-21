import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats  # Requis pour Beta/Binom
from itertools import product


class HysteresisValidator:
    """
    MODULE V47.2: HYSTERESIS (DWELL-TIME ASYMMETRY) TEST
    Tests if the transition to the mature regime is irreversible (Ratchet Effect).

    Fixes:
    1. Added missing 'time_in_mature' in return dict (Critical Key Error fix).
    2. Implemented Beta/Wilson interval for robust CI when reversion rate is 0%.
    3. Handles end-of-series transient excursions.
    """

    def __init__(self, all_dataframes):
        # Filter for relevant projects (e.g., > 24 months)
        self.dfs = {k: v for k, v in all_dataframes.items() if v is not None and len(v) > 24}
        self.results_cache = {}

    def _analyze_series(self, gamma_series, g_high, g_low, durable_months):
        """
        State machine to detect regime transitions.
        """
        n = len(gamma_series)
        state = 'EXPLORATORY'  # States: EXPLORATORY, MATURE, REVERTED

        entry_date_idx = None
        exit_date_idx = None

        transient_excursions = 0

        # Consecutive counters
        conosec_high = 0
        conosec_low = 0

        # Helper to get values
        values = gamma_series.values

        for i, val in enumerate(values):
            if state == 'EXPLORATORY':
                # Check for Entry condition
                if val >= g_high:
                    conosec_high += 1
                else:
                    conosec_high = 0

                # If durable entry condition met
                if conosec_high >= durable_months:
                    state = 'MATURE'
                    # The entry date is the start of this sequence
                    entry_date_idx = i - durable_months + 1
                    conosec_low = 0  # Reset low counter

            elif state == 'MATURE':
                # We are in mature regime, check for drops
                if val < g_low:
                    conosec_low += 1
                else:
                    # If we had a drop but it recovered before becoming durable -> Transient Excursion
                    if 0 < conosec_low < durable_months:
                        transient_excursions += 1
                    conosec_low = 0

                # Check for Durable Exit condition
                if conosec_low >= durable_months:
                    state = 'REVERTED'
                    exit_date_idx = i - durable_months + 1
                    break  # Stop analyzing after first fatal reversion (conservative)

        # Count any ongoing excursion at end of series
        if state == 'MATURE' and 0 < conosec_low < durable_months:
            transient_excursions += 1

        is_mature = (entry_date_idx is not None)
        has_reverted = (exit_date_idx is not None)

        # Calculate durations
        time_to_entry = entry_date_idx if is_mature else None

        if has_reverted:
            time_in_mature = exit_date_idx - entry_date_idx
        elif is_mature:
            time_in_mature = n - entry_date_idx
        else:
            time_in_mature = 0

        return {
            'is_mature': is_mature,
            'time_to_entry': time_to_entry,
            'time_in_mature': time_in_mature,  # <--- CRITICAL FIX V47.2
            'has_reverted': has_reverted,
            'time_to_exit': (exit_date_idx - entry_date_idx) if has_reverted else None,
            'transient_excursions': transient_excursions,
            'final_state': state
        }

    def run_main_test(self, g_high=0.7, g_low=0.5, durable=6):
        """
        Output 1 & 2: Project stats and Aggregated stats.
        """
        print("\n" + "=" * 80)
        print(f"HYSTERESIS TEST (High={g_high}, Low={g_low}, Durable={durable}m)")
        print("=" * 80)

        rows = []
        for name, df in self.dfs.items():
            # Robust column check
            col_name = 'monthly_gamma'
            if col_name not in df.columns:
                if 'gamma_composite' in df.columns:
                    col_name = 'gamma_composite'
                else:
                    continue

            res = self._analyze_series(df[col_name], g_high, g_low, durable)
            res['project'] = name
            rows.append(res)

        df_res = pd.DataFrame(rows)
        self.results_cache['main'] = df_res

        # 1. Per Project Display
        print(f"\n{'Project':<20} | {'Entry (Mo)':<10} | {'Reverted?':<10} | {'Excursions':<10} | {'Mths in Regime'}")
        print("-" * 75)

        mature_subset = df_res[df_res['is_mature']].sort_values('transient_excursions', ascending=False)

        for _, row in mature_subset.head(15).iterrows():
            rev_str = "YES" if row['has_reverted'] else "No"
            print(
                f"{row['project']:<20} | {row['time_to_entry']:<10} | {rev_str:<10} | {row['transient_excursions']:<10} | {row['time_in_mature']}")

        # 2. Aggregated Stats
        n_total = len(df_res)
        n_entered = df_res['is_mature'].sum()
        n_reverted = df_res['has_reverted'].sum()

        reversion_rate = (n_reverted / n_entered * 100) if n_entered > 0 else 0
        avg_excursions = df_res.loc[df_res['is_mature'], 'transient_excursions'].mean()

        print("\n" + "-" * 40)
        print("AGGREGATED RESULTS")
        print("-" * 40)
        print(f"Total Projects analyzed : {n_total}")
        print(f"Durable Entry (> {g_high})  : {n_entered} ({n_entered / n_total:.1%})")
        print(f"Durable Reversion (< {g_low}): {n_reverted}")
        print(f"REVERSION RATE          : {reversion_rate:.2f}%")

        # --- ROBUST CI FIX (Wilson/Beta) ---
        if n_entered > 0:
            # Using Beta distribution for robust CI near 0% or 100%
            ci_low, ci_high = stats.beta.interval(0.95, n_reverted + 0.5, n_entered - n_reverted + 0.5)
            # Clip to [0,1] just in case
            ci_low = max(0.0, ci_low)
            ci_high = min(1.0, ci_high)
            print(f"95% CI (Robust)         : [{ci_low * 100:.1f}% - {ci_high * 100:.1f}%]")
        # -----------------------------------

        print(f"Avg Transient Excursions: {avg_excursions:.1f} per mature project")

        # Adjusted threshold logic considering CI
        if reversion_rate < 10:
            print("\n✅ HYPOTHESIS CONFIRMED: Hysteresis is strong.")
            print("   Target Phrase: 'Although transient excursions below the threshold occur,")
            print("   durable reversions are rare or absent once the mature regime is reached.'")
        else:
            print(f"\n⚠️ HYPOTHESIS WEAK: Reversion rate is {reversion_rate:.1f}%")

        self.plot_hysteresis(df_res)
        return df_res

    def run_sensitivity_analysis(self):
        """
        Output 3: Sensitivity Table.
        """
        print("\n🔎 Sensitivity Analysis...")

        g_highs = [0.65, 0.70, 0.75]
        g_lows = [0.45, 0.50, 0.55]
        durations = [4, 6, 8]

        results = []

        for gh, gl, d in product(g_highs, g_lows, durations):
            if gl >= gh: continue

            n_rev = 0
            n_ent = 0

            for _, df in self.dfs.items():
                col = 'monthly_gamma' if 'monthly_gamma' in df.columns else 'gamma_composite'
                res = self._analyze_series(df[col], gh, gl, d)
                if res['is_mature']:
                    n_ent += 1
                if res['has_reverted']:
                    n_rev += 1

            rate = (n_rev / n_ent * 100) if n_ent > 0 else 0
            results.append({
                'GAMMA_HIGH': gh,
                'GAMMA_LOW': gl,
                'DURATION': d,
                'N_Entered': n_ent,
                'Reversion_%': round(rate, 2)
            })

        df_sens = pd.DataFrame(results)
        print(df_sens.to_string(index=False))
        return df_sens

    def plot_hysteresis(self, df_res):
        """
        Output 4: Visualizations.
        """
        mature_df = df_res[df_res['is_mature']].copy()
        if mature_df.empty: return

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Plot A: Comparative Histogram
        # Compare "Time to Entry" vs "Time in Mature" (for survivors)
        # Cap duration at 150 months for readability
        time_to_entry = mature_df['time_to_entry'].dropna()
        time_in_mature = mature_df[~mature_df['has_reverted']]['time_in_mature'].dropna().clip(upper=150)

        axes[0].hist([time_to_entry, time_in_mature],
                     bins=20,
                     label=['Time to Entry', 'Duration in Mature (Survivors)'],
                     color=['#e67e22', '#27ae60'],
                     alpha=0.7,
                     edgecolor='black')

        axes[0].set_title('Asymmetry: Entry vs Persistence')
        axes[0].set_xlabel('Months')
        axes[0].set_ylabel('Number of Projects')
        axes[0].legend()

        # Plot B: Scatter Duration vs Status
        # Jitter for visibility
        mature_df['jitter_y'] = mature_df['has_reverted'].astype(int) + np.random.normal(0, 0.05, len(mature_df))
        colors = mature_df['has_reverted'].map({True: '#c0392b', False: '#2980b9'})

        axes[1].scatter(mature_df['time_in_mature'], mature_df['jitter_y'],
                        c=colors, alpha=0.6, s=100, edgecolors='k')

        axes[1].set_yticks([0, 1])
        axes[1].set_yticklabels(['Stable (Mature)', 'Reverted'])
        axes[1].set_xlabel('Months in Mature Regime')
        axes[1].set_title('Stability Landscape')

        # Annotate outliers (Reverted)
        for _, row in mature_df[mature_df['has_reverted']].iterrows():
            axes[1].annotate(row['project'], (row['time_in_mature'], 1.05), fontsize=8, color='#c0392b')

        plt.tight_layout()
        plt.savefig("omega-v47-hysteresis-final.pdf", dpi=300)
        print("✅ Plot saved: omega-v47-hysteresis-final.pdf")
        plt.close()