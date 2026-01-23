import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings

# Configuration Globale
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
MIN_MONTHS = 12
N_SIMULATIONS = 1000
BURN_IN_PERIOD = 50  # Période de chauffe pour détacher la dépendance initiale

# Style des graphiques
sns.set_style("whitegrid")
plt.rcParams.update({'figure.max_open_warning': 0})


# ==============================================================================
# 1. MOTEUR AR(1) : CALIBRATION & SIMULATION
# ==============================================================================

def calibrate_ar1(series):
    """
    Calibration AR(1) via OLS sur données centrées.
    Retourne: mu (moyenne), phi (coeff auto-reg), sigma (bruit)
    CORRECTION : Accepte les tableaux NumPy directement.
    """
    # Sécurité : on s'assure que c'est un array numpy
    series = np.array(series)

    mu = np.mean(series)
    y_centered = series - mu

    # Correction ici : on enlève .values car c'est déjà un array
    y_t = y_centered[1:]
    y_t_minus_1 = y_centered[:-1]

    # Estimation OLS
    # Sécurité division par zéro
    denominator = np.sum(y_t_minus_1 ** 2)
    if denominator == 0:
        return mu, 0, 0

    phi = np.sum(y_t * y_t_minus_1) / denominator

    # Résidus et Sigma
    residuals = y_t - phi * y_t_minus_1
    sigma_epsilon = np.std(residuals, ddof=1)

    return mu, phi, sigma_epsilon
def simulate_ar1_robust(mu, phi, sigma, n_steps, n_sims=1000, burn_in=50):
    """
    Simulation Monte Carlo avec période de chauffe (Burn-in).
    Assure que la distribution simulée reflète les propriétés structurelles de l'AR(1)
    et non la position du dernier point observé.
    """
    total_steps = n_steps + burn_in

    # Initialisation vectorielle
    simulations = np.zeros((total_steps, n_sims))
    epsilon = np.random.normal(0, sigma, size=(total_steps, n_sims))

    # Démarrage à l'équilibre (moyenne)
    current_val = np.full(n_sims, mu)

    for t in range(total_steps):
        next_val = mu + phi * (current_val - mu) + epsilon[t, :]
        simulations[t, :] = next_val
        current_val = next_val

    # Rejet du burn-in
    return simulations[burn_in:, :]


# ==============================================================================
# 2. MÉTRIQUES (CALCUL STRICT)
# ==============================================================================

def calculate_metrics_strict(series_array):
    """
    Calcul vectorisé des métriques.
    Input: Array shape (Temps, Simulations) ou (Temps, 1)
    """
    if series_array.ndim == 1:
        series_array = series_array.reshape(-1, 1)

    T, N = series_array.shape

    # 1. Zero-Crossing Strict
    # Détecte un changement de signe franc (produit < 0)
    # Ignore les cas où la valeur touche 0 sans traverser
    crossings = np.sum((series_array[:-1] * series_array[1:]) < 0, axis=0)
    freq_zc = crossings / (T - 1)

    # 2. Variance
    variance = np.var(series_array, axis=0)

    # 3. Confinement Ratio (Zone [-0.3, +0.3])
    confinement = np.mean((series_array >= -0.3) & (series_array <= 0.3), axis=0)

    return freq_zc, variance, confinement


def compute_stats_row(observed_val, simulated_dist, project_name, metric_name):
    """
    Helper: Calcule p-values, quantiles, direction pour une ligne CSV.
    """
    median_sim = np.median(simulated_dist)

    # P-value bilatérale
    percentile_rank = np.mean(simulated_dist <= observed_val)
    p_value = 2 * min(percentile_rank, 1 - percentile_rank)

    # Direction de l'effet
    direction = "higher" if observed_val > median_sim else "lower"

    return {
        'project': project_name,
        'metric': metric_name,
        'observed_value': observed_val,
        'simulated_median': median_sim,
        'simulated_q025': np.percentile(simulated_dist, 2.5),
        'simulated_q975': np.percentile(simulated_dist, 97.5),
        'percentile': percentile_rank,
        'pvalue': p_value,
        'direction': direction
    }


# ==============================================================================
# 3. PIPELINE D'ANALYSE
# ==============================================================================

def run_null_model_analysis(df):
    results_long = []
    diagnostic_data = {}  # Pour les spaghetti plots

    projects = df['project'].unique()
    print(f"Starting Null Model Analysis on {len(projects)} projects...")

    for proj in projects:
        sub = df[df['project'] == proj].sort_values('month')

        # Extraction des phases
        early = sub[sub['regime'] == 'early']['AI'].values
        mature = sub[sub['regime'] == 'mature']['AI'].values

        # Filtres
        if len(early) < MIN_MONTHS or len(mature) < MIN_MONTHS:
            continue

        # 1. Calibration (Phase Early)
        mu, phi, sigma = calibrate_ar1(early)

        # Filtre stationnarité AR(1)
        if np.abs(phi) >= 1.0:
            print(f"⚠️ [Excluded] {proj}: Non-stationary (phi={phi:.2f})")
            continue

        # 2. Simulation (Null Model)
        sim_paths = simulate_ar1_robust(mu, phi, sigma, len(mature), N_SIMULATIONS, BURN_IN_PERIOD)

        # Stockage pour diagnostic (quelques projets représentatifs)
        if len(diagnostic_data) < 4:
            diagnostic_data[proj] = {'obs': mature, 'sim': sim_paths, 'params': (mu, phi)}

        # 3. Calculs Métriques
        obs_zc, obs_var, obs_conf = calculate_metrics_strict(mature)
        sim_zc, sim_var, sim_conf = calculate_metrics_strict(sim_paths)

        # 4. Enregistrement (Format Long)
        results_long.append(compute_stats_row(obs_zc[0], sim_zc, proj, 'zero_crossing_freq'))
        results_long.append(compute_stats_row(obs_var[0], sim_var, proj, 'variance'))
        results_long.append(compute_stats_row(obs_conf[0], sim_conf, proj, 'confinement_ratio'))

    return pd.DataFrame(results_long), diagnostic_data


# ==============================================================================
# 4. VISUALISATION (FIGURES CLÉS)
# ==============================================================================

def plot_main_comparison_panel(df):
    """
    FIGURE PRINCIPALE : Panel 3 histogrammes (Observed vs Simulated).
    C'est la preuve visuelle centrale de l'article.
    """
    metrics_map = {
        'zero_crossing_freq': 'Zero-Crossing Frequency',
        'variance': 'Variance',
        'confinement_ratio': 'Confinement Ratio'
    }

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    plt.subplots_adjust(wspace=0.3)

    for i, (metric_key, metric_label) in enumerate(metrics_map.items()):
        ax = axes[i]
        sub = df[df['metric'] == metric_key]

        if sub.empty: continue

        # Données
        obs_vals = sub['observed_value']
        sim_vals = sub['simulated_median']

        # Test Binomial pour le titre
        n_sig = np.sum(sub['pvalue'] < 0.05)
        n_tot = len(sub)
        binom = stats.binomtest(n_sig, n_tot, p=0.05, alternative='greater')

        # Histogrammes
        sns.histplot(obs_vals, color='#1f77b4', label='Observed (Mature)', ax=ax, kde=True, element="step", alpha=0.6)
        sns.histplot(sim_vals, color='#7f7f7f', label='Simulated Median (Null)', ax=ax, kde=True, element="step",
                     alpha=0.3)

        # Lignes verticales (Médianes globales)
        ax.axvline(obs_vals.median(), color='#1f77b4', linestyle='-', linewidth=2)
        ax.axvline(sim_vals.median(), color='#4d4d4d', linestyle='--', linewidth=2)

        # Titres et Labels
        title = f"{metric_label}\nBinomial p-value: {binom.pvalue:.1e}"
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel("Value")
        if i == 0: ax.legend()

    plt.suptitle("Observed Mature Regime vs. Null Model (AR1)", fontsize=14, y=1.05)

    # Sauvegarde
    plt.savefig("null_model_comparison_panel.pdf", bbox_inches='tight')
    plt.savefig("null_model_comparison_panel.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Figure Principale sauvegardée : null_model_comparison_panel.pdf/png")


def plot_diagnostic_spaghetti(diagnostic_data):
    """
    FIGURE SI : Trajectoires individuelles pour comprendre la dynamique locale.
    """
    n = len(diagnostic_data)
    if n == 0: return

    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), sharey=True)
    if n == 1: axes = [axes]

    for ax, (proj, data) in zip(axes, diagnostic_data.items()):
        # Simulations (Gris léger)
        ax.plot(data['sim'][:, :50], color='gray', alpha=0.1, linewidth=0.5)
        # Observé (Bleu)
        ax.plot(data['obs'], color='blue', linewidth=1.5, label='Observed')

        # Limites Confinement
        ax.axhline(0.3, color='red', linestyle=':', alpha=0.5)
        ax.axhline(-0.3, color='red', linestyle=':', alpha=0.5)

        ax.set_title(f"{proj} (φ={data['params'][1]:.2f})")
        ax.set_xlabel("Time (Mature Phase)")

    axes[0].set_ylabel("Authority Index")
    plt.suptitle("Diagnostic: Observed Trajectory vs. Null Model Bundle", y=1.05)

    plt.tight_layout()
    plt.savefig("null_model_spaghetti_diagnostic.png", dpi=200, bbox_inches='tight')
    plt.close()
    print("✅ Figure Diagnostic sauvegardée : null_model_spaghetti_diagnostic.png")


# ==============================================================================
# 5. EXPORTS ET WRAPPER
# ==============================================================================

def generate_reports(results_df):
    if results_df.empty: return

    # 1. Export CSV Long (Tidy Data)
    cols = ['project', 'metric', 'observed_value', 'simulated_median',
            'simulated_q025', 'simulated_q975', 'percentile', 'pvalue', 'direction']
    results_df[cols].to_csv("null_model_results_long.csv", index=False)
    print("✅ CSV exporté : null_model_results_long.csv")

    # 2. Tableau Récapitulatif
    summary_data = []
    metrics_order = ['zero_crossing_freq', 'variance', 'confinement_ratio']

    for metric in metrics_order:
        sub = results_df[results_df['metric'] == metric]
        if sub.empty: continue

        n_sig = np.sum(sub['pvalue'] < 0.05)
        n_tot = len(sub)
        binom = stats.binomtest(n_sig, n_tot, p=0.05, alternative='greater')

        summary_data.append({
            'Metric': metric,
            'Obs. Median': f"{sub['observed_value'].median():.4f}",
            'Sim. Median': f"{sub['simulated_median'].median():.4f}",
            'Rejection Rate': f"{n_sig}/{n_tot} ({100 * n_sig / n_tot:.1f}%)",
            'Binom. p': f"{binom.pvalue:.2e}"
        })

    summ_df = pd.DataFrame(summary_data)
    print("\n" + summ_df.to_markdown(index=False))

    # 3. Code LaTeX
    print("\n--- LaTeX Code ---")
    print(summ_df.to_latex(index=False,
                           caption="Statistical comparison of observed mature regime vs. stochastic null model (AR1).",
                           label="tab:null_model"))


# ==============================================================================
# MAIN (EXECUTION TEST)
# ==============================================================================

if __name__ == "__main__":
    # Génération données synthétiques pour valider le pipeline
    print("Generating synthetic test data...")
    dummy_data = []

    # Scénario : Le régime mature est un attracteur (variance faible)
    # Le null model (AR1 calibré sur early) devrait surestimer la variance mature
    for i in range(50):
        # Phase Early: Haute variance, marche aléatoire bornée
        early = np.random.normal(0, 0.6, 36)
        # Phase Mature: Confinement fort (Sigma très faible)
        mature = np.random.normal(0, 0.15, 48)

        proj = f"PROJ_{i:02d}"
        for t, v in enumerate(early):
            dummy_data.append({'project': proj, 'month': t, 'AI': v, 'regime': 'early'})
        for t, v in enumerate(mature):
            dummy_data.append({'project': proj, 'month': t + 36, 'AI': v, 'regime': 'mature'})

    df_input = pd.DataFrame(dummy_data)

    print("-" * 60)
    print("RUNNING NULL MODEL VALIDATION SUITE")
    print("-" * 60)

    # 1. Analyse
    results, diag_data = run_null_model_analysis(df_input)

    # 2. Visualisations
    plot_main_comparison_panel(results)  #
    plot_diagnostic_spaghetti(diag_data)  #

    # 3. Rapports
    generate_reports(results)

    print("-" * 60)
    print("PIPELINE COMPLETED SUCCESSFULLY")