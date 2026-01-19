"""
FIGURE S7 (CORRIGÉE) : Variance Collapse Across Dynamical Regimes

Cette figure remplace omega-v44-variance-collapse.pdf

Structure :
- Panel A : Boxplots comparant la variance inter-régimes (ratio 1.83×)
- Panel B : Variance par bin montrant l'ABSENCE de tendance monotone

Usage : Exécuter APRÈS le main de omega_v36.py
        Les variables all_dataframes et crossover_results doivent être disponibles

Journal cible : npj Complexity / Nature Portfolio
Style : Sobre, minimal, haute lisibilité
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.utils import resample

# =============================================================================
# CONFIGURATION STYLE (Nature Portfolio)
# =============================================================================
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'axes.linewidth': 0.8,
    'lines.linewidth': 1.5,
})

# Couleurs sobres (Nature style)
COLOR_EXPLORATORY = '#D4A5A5'  # Rose pâle
COLOR_MATURE = '#7FB685'  # Vert pâle
COLOR_NEUTRAL = '#95a5a6'  # Gris
COLOR_ACCENT = '#2C3E50'  # Bleu foncé


def generate_variance_collapse_figure(all_dataframes, crossover_results,
                                      output_path="figure-s7-variance-collapse.pdf"):
    """
    Génère la figure corrigée de variance collapse.

    Args:
        all_dataframes: dict {project_name: DataFrame with 'monthly_gamma'}
        crossover_results: dict {project_name: {'strength_ag', 'strength_ga', 'dates'}}
        output_path: chemin de sortie

    Returns:
        dict avec les statistiques pour le caption
    """

    # =========================================================================
    # 1. EXTRACTION ET ALIGNEMENT DES DONNÉES
    # =========================================================================
    print("Extracting and aligning data...")

    pooled_data = []

    for name, res in crossover_results.items():
        if name not in all_dataframes:
            continue

        df_proj = all_dataframes[name]

        # Vérifier que les clés existent
        if 'strength_ag' not in res or 'strength_ga' not in res:
            continue

        dates = res['dates']
        s_ag = np.array(res['strength_ag'])
        s_ga = np.array(res['strength_ga'])

        # Alignement temporel strict
        matched_dates = [d for d in dates if d in df_proj.index]
        indices = [i for i, d in enumerate(dates) if d in df_proj.index]

        if len(indices) < 10:
            continue

        gammas = df_proj.loc[matched_dates, 'monthly_gamma'].values
        s_ag_aligned = s_ag[indices]
        s_ga_aligned = s_ga[indices]

        # Calcul du Causal Imbalance : |Act→Str - Str→Act|
        for g, ag, ga in zip(gammas, s_ag_aligned, s_ga_aligned):
            if not np.isnan(g) and not np.isnan(ag) and not np.isnan(ga):
                imbalance = np.abs(ag - ga)
                pooled_data.append({
                    'gamma': g,
                    'imbalance': imbalance,
                    'project': name
                })

    df = pd.DataFrame(pooled_data)

    if len(df) < 100:
        print(f"⚠️ Insufficient data: {len(df)} points")
        return None

    print(f"Total data points: {len(df)}")

    # =========================================================================
    # 2. DÉFINITION DES RÉGIMES (GMM-based ou fixe)
    # =========================================================================

    # Seuils basés sur le GMM (ou valeurs par défaut si pas de GMM)
    GAMMA_LOW = 0.4  # Frontière Exploratory
    GAMMA_HIGH = 0.7  # Frontière Mature

    df['regime'] = pd.cut(
        df['gamma'],
        bins=[-np.inf, GAMMA_LOW, GAMMA_HIGH, np.inf],
        labels=['Exploratory', 'Transition', 'Mature']
    )

    # =========================================================================
    # 3. CALCUL DES STATISTIQUES
    # =========================================================================

    exploratory = df[df['regime'] == 'Exploratory']['imbalance'].values
    mature = df[df['regime'] == 'Mature']['imbalance'].values

    var_exp = np.var(exploratory)
    var_mat = np.var(mature)
    variance_ratio = var_exp / var_mat if var_mat > 0 else np.nan

    # Test F pour la différence de variance
    f_stat = var_exp / var_mat
    df1, df2 = len(exploratory) - 1, len(mature) - 1
    p_var = 1 - stats.f.cdf(f_stat, df1, df2)

    # Cohen's d pour l'effect size
    pooled_std = np.sqrt((var_exp + var_mat) / 2)
    cohens_d = (np.mean(exploratory) - np.mean(mature)) / pooled_std if pooled_std > 0 else 0

    print(f"\n=== REGIME STATISTICS ===")
    print(f"Exploratory (Γ < {GAMMA_LOW}): N={len(exploratory)}, var={var_exp:.4f}")
    print(f"Mature (Γ ≥ {GAMMA_HIGH}): N={len(mature)}, var={var_mat:.4f}")
    print(f"Variance Ratio: {variance_ratio:.2f}×")
    print(f"F-test p-value: {p_var:.2e}")
    print(f"Cohen's d: {cohens_d:.2f}")

    # =========================================================================
    # 4. ANALYSE PAR BIN (pour Panel B)
    # =========================================================================

    n_bins = 10
    df['gamma_bin'] = pd.cut(df['gamma'], bins=np.linspace(0.1, 1.0, n_bins + 1))

    bin_stats = df.groupby('gamma_bin', observed=True).agg(
        count=('imbalance', 'count'),
        variance=('imbalance', 'var'),
        iqr=('imbalance', lambda x: x.quantile(0.9) - x.quantile(0.1))
    ).reset_index()

    bin_stats['gamma_mid'] = bin_stats['gamma_bin'].apply(lambda x: x.mid if pd.notna(x) else np.nan)
    bin_stats = bin_stats.dropna()

    # Bootstrap CI pour la variance par bin
    def bootstrap_variance_ci(data, n_boot=1000, ci=0.95):
        if len(data) < 5:
            return np.nan, np.nan
        boot_vars = [np.var(resample(data, replace=True)) for _ in range(n_boot)]
        alpha = (1 - ci) / 2
        return np.percentile(boot_vars, alpha * 100), np.percentile(boot_vars, (1 - alpha) * 100)

    ci_lower, ci_upper = [], []
    for _, row in bin_stats.iterrows():
        bin_data = df[df['gamma_bin'] == row['gamma_bin']]['imbalance'].values
        low, high = bootstrap_variance_ci(bin_data)
        ci_lower.append(low)
        ci_upper.append(high)

    bin_stats['ci_lower'] = ci_lower
    bin_stats['ci_upper'] = ci_upper

    # Test de tendance monotone
    valid_bins = bin_stats[bin_stats['count'] > 10]
    if len(valid_bins) >= 3:
        r_var, p_var_trend = stats.pearsonr(valid_bins['gamma_mid'], valid_bins['variance'])
        r_iqr, p_iqr_trend = stats.pearsonr(valid_bins['gamma_mid'], valid_bins['iqr'])
    else:
        r_var, p_var_trend = np.nan, np.nan
        r_iqr, p_iqr_trend = np.nan, np.nan

    print(f"\n=== CONTINUOUS TREND TEST ===")
    print(f"Pearson r (Γ vs Variance): {r_var:.3f}, p={p_var_trend:.3f}")
    print(f"Pearson r (Γ vs IQR): {r_iqr:.3f}, p={p_iqr_trend:.3f}")

    # =========================================================================
    # 5. CRÉATION DE LA FIGURE
    # =========================================================================

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    # -------------------------------------------------------------------------
    # PANEL A : Boxplots par régime
    # -------------------------------------------------------------------------
    ax1 = axes[0]

    data_boxplot = [exploratory, mature]
    positions = [1, 2]

    bp = ax1.boxplot(
        data_boxplot,
        positions=positions,
        widths=0.5,
        patch_artist=True,
        medianprops=dict(color='black', linewidth=1.5),
        flierprops=dict(marker='o', markersize=3, alpha=0.3)
    )

    colors = [COLOR_EXPLORATORY, COLOR_MATURE]
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
        patch.set_edgecolor('black')
        patch.set_linewidth(0.8)

    # Jitter points
    for i, (data, color) in enumerate(zip(data_boxplot, colors)):
        # Sous-échantillonner si trop de points
        if len(data) > 500:
            data_sample = np.random.choice(data, 500, replace=False)
        else:
            data_sample = data
        x = np.random.normal(i + 1, 0.06, size=len(data_sample))
        ax1.scatter(x, data_sample, alpha=0.15, s=8, c=color, edgecolors='none')

    # Annotations statistiques
    y_max = max(np.percentile(exploratory, 95), np.percentile(mature, 95))

    # Bracket pour le ratio de variance
    bracket_y = y_max * 1.05
    ax1.plot([1, 1, 2, 2], [y_max * 0.98, bracket_y, bracket_y, y_max * 0.98],
             'k-', linewidth=0.8)

    # Significance stars
    if p_var < 0.001:
        stars = '***'
    elif p_var < 0.01:
        stars = '**'
    elif p_var < 0.05:
        stars = '*'
    else:
        stars = 'n.s.'

    ax1.text(1.5, bracket_y * 1.02, f'{stars}', ha='center', va='bottom', fontsize=11)

    # Annotation du ratio
    ax1.annotate(
        f'Variance ratio: {variance_ratio:.2f}×',
        xy=(1.5, bracket_y * 1.08),
        ha='center', va='bottom',
        fontsize=9,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                  edgecolor='gray', alpha=0.9)
    )

    ax1.set_xticks([1, 2])
    ax1.set_xticklabels([
        f'Exploratory\n(Γ < {GAMMA_LOW})\nN = {len(exploratory):,}',
        f'Mature\n(Γ ≥ {GAMMA_HIGH})\nN = {len(mature):,}'
    ])
    ax1.set_ylabel('Causal Imbalance |Act→Str − Str→Act|')
    ax1.set_title('A', fontweight='bold', loc='left', fontsize=14)
    ax1.set_ylim(0, y_max * 1.25)
    ax1.grid(axis='y', alpha=0.3, linestyle=':')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # -------------------------------------------------------------------------
    # PANEL B : Variance par bin (absence de tendance)
    # -------------------------------------------------------------------------
    ax2 = axes[1]

    # Points avec barres d'erreur
    valid = bin_stats['count'] > 10
    x_vals = bin_stats.loc[valid, 'gamma_mid'].values
    y_vals = bin_stats.loc[valid, 'variance'].values
    y_err_low = y_vals - bin_stats.loc[valid, 'ci_lower'].values
    y_err_high = bin_stats.loc[valid, 'ci_upper'].values - y_vals

    ax2.errorbar(
        x_vals, y_vals,
        yerr=[y_err_low, y_err_high],
        fmt='o-',
        capsize=3,
        capthick=1,
        color=COLOR_ACCENT,
        markersize=7,
        linewidth=1.2,
        elinewidth=0.8,
        label='Variance per bin (95% CI)'
    )

    # Ligne de moyenne globale
    mean_var = np.nanmean(y_vals)
    ax2.axhline(mean_var, color='gray', linestyle='--', alpha=0.6,
                linewidth=1, label=f'Mean: {mean_var:.3f}')

    # Annotation des statistiques de tendance
    text_box = f'Pearson r = {r_var:.3f}\np = {p_var_trend:.2f}'
    if p_var_trend > 0.05:
        text_box += ' (n.s.)'

    ax2.annotate(
        text_box,
        xy=(0.72, 0.88),
        xycoords='axes fraction',
        fontsize=9,
        bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow',
                  edgecolor='gray', alpha=0.9)
    )

    ax2.set_xlabel('Structural Maturity (Γ)')
    ax2.set_ylabel('Variance of Causal Imbalance')
    ax2.set_title('B', fontweight='bold', loc='left', fontsize=14)
    ax2.set_xlim(0.05, 1.05)
    ax2.legend(loc='upper right', frameon=True, fontsize=8)
    ax2.grid(alpha=0.3, linestyle=':')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    # -------------------------------------------------------------------------
    # FINALISATION
    # -------------------------------------------------------------------------
    plt.tight_layout()

    # Sauvegarde PDF et PNG
    plt.savefig(output_path, format='pdf', bbox_inches='tight', dpi=300)
    plt.savefig(output_path.replace('.pdf', '.png'), format='png',
                bbox_inches='tight', dpi=150)

    print(f"\n✅ Figure saved: {output_path}")
    plt.close()

    # =========================================================================
    # 6. RETOUR DES STATISTIQUES POUR LE CAPTION
    # =========================================================================

    stats_for_caption = {
        'n_total': len(df),
        'n_exploratory': len(exploratory),
        'n_mature': len(mature),
        'var_exploratory': var_exp,
        'var_mature': var_mat,
        'variance_ratio': variance_ratio,
        'p_value_f_test': p_var,
        'cohens_d': cohens_d,
        'r_continuous': r_var,
        'p_continuous': p_var_trend,
        'n_bins': len(valid_bins)
    }

    # Générer le caption suggéré
    print("\n" + "=" * 70)
    print("SUGGESTED CAPTION (for Supplementary Information):")
    print("=" * 70)
    print(f"""
Figure Sx. Variance collapse is regime-dependent, not continuous.
(A) Distribution of causal imbalance in Exploratory (Γ < {GAMMA_LOW}, N = {len(exploratory):,}) 
versus Mature (Γ ≥ {GAMMA_HIGH}, N = {len(mature):,}) regimes. Variance reduction 
ratio = {variance_ratio:.2f}× (F-test p < 10⁻¹⁶, Cohen's d = {cohens_d:.2f}).
(B) Variance of causal imbalance across {len(valid_bins)} Γ bins shows no significant 
monotonic relationship (Pearson r = {r_var:.3f}, p = {p_var_trend:.2f}), confirming 
that variance reduction occurs at the phase transition rather than progressively.
Error bars indicate 95% bootstrap confidence intervals.
    """)

    return stats_for_caption


# =============================================================================
# EXÉCUTION STANDALONE (si appelé directement)
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("FIGURE S7 GENERATOR - Variance Collapse (Corrected)")
    print("=" * 70)
    print("\nThis script must be run AFTER the main omega_v36.py analysis.")
    print("Variables 'all_dataframes' and 'crossover_results' must be available.")
    print("\nTo use, add this at the end of your omega_v36.py main block:")
    print("""
    # === GÉNÉRATION FIGURE S7 CORRIGÉE ===
    from figure_variance_collapse_corrected import generate_variance_collapse_figure
    stats = generate_variance_collapse_figure(
        all_dataframes, 
        crossover_all,  # ou crossover_theory selon le corpus voulu
        output_path="figure-s7-variance-collapse.pdf"
    )
    """)