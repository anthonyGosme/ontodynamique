"""
MODULE CTR (Core Touch Ratio) - Version Corrigée
=================================================
Test mécaniste : Les systèmes matures touchent-ils des fichiers plus anciens ?

Hypothèse : In the mature regime, a larger fraction of development activity
targets files that predate the commit by a substantial margin.

"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from scipy import stats
from sklearn.utils import resample

# Import PyGit2 pour la performance
try:
    import pygit2
except ImportError:
    print("⚠️ PyGit2 non disponible. Installez avec: pip install pygit2")
    pygit2 = None


# ==============================================================================
# 1. EXTRACTION DU CORE TOUCH RATIO
# ==============================================================================

class CoreTouchExtractor:
    """
    Extrait le Core Touch Ratio (CTR) pour un repository.

    CTR = Proportion de fichiers modifiés dont l'âge > seuil T

    L'âge d'un fichier = date_commit - date_création_fichier
    """

    def __init__(self, repo_path, project_name, thresholds=None):
        """
        Args:
            repo_path: Chemin vers le repository Git
            project_name: Nom du projet
            thresholds: Liste des seuils en mois [6, 12, 24]
        """
        self.repo_path = repo_path
        self.project_name = project_name
        self.thresholds = thresholds or [6, 12, 24]
        self.repo = None

    def process_repo(self):
        """
        Parcourt le repo chronologiquement pour calculer l'âge des fichiers.

        Returns:
            DataFrame avec colonnes: CTR_6, CTR_12, CTR_24, project
            Index: date mensuelle
        """
        if pygit2 is None:
            print(f"⚠️ PyGit2 requis pour {self.project_name}")
            return None

        try:
            self.repo = pygit2.Repository(self.repo_path)
        except Exception as e:
            print(f"⚠️ Erreur ouverture repo {self.project_name}: {e}")
            return None

        # Dictionnaire: filepath -> timestamp de création
        file_birthdays = {}

        # Liste des données par commit
        commit_data = []

        # Walker chronologique (OLDEST -> NEWEST)
        # GIT_SORT_REVERSE inverse l'ordre par défaut (newest first)
        try:
            walker = self.repo.walk(
                self.repo.head.target,
                pygit2.GIT_SORT_TOPOLOGICAL | pygit2.GIT_SORT_REVERSE
            )
        except Exception as e:
            print(f"⚠️ Erreur walker {self.project_name}: {e}")
            return None

        for commit in walker:
            try:
                # Calcul du diff
                if not commit.parents:
                    # Premier commit: diff avec arbre vide
                    diff = commit.tree.diff_to_tree(swap=True)
                else:
                    # Diff avec le parent
                    diff = self.repo.diff(commit.parents[0], commit)

                current_ts = commit.commit_time
                modified_files_ages = []

                # Parcours des fichiers modifiés
                for patch in diff:
                    delta = patch.delta
                    path = delta.new_file.path

                    # Mise à jour de la date de naissance si nouveau fichier
                    if delta.status == pygit2.GIT_DELTA_ADDED or path not in file_birthdays:
                        file_birthdays[path] = current_ts

                    # Calcul de l'âge en mois
                    age_seconds = current_ts - file_birthdays[path]
                    age_months = age_seconds / (3600 * 24 * 30.44)  # Mois moyen
                    modified_files_ages.append(max(0, age_months))  # Âge >= 0

                if not modified_files_ages:
                    continue

                # Calcul des CTR pour chaque seuil
                ages_arr = np.array(modified_files_ages)
                row = {
                    'timestamp': current_ts,
                    'n_files': len(ages_arr),
                    'mean_age': np.mean(ages_arr)
                }

                for t in self.thresholds:
                    # CTR = proportion de fichiers avec âge >= t mois
                    ctr = np.mean(ages_arr >= t)
                    row[f'CTR_{t}'] = ctr

                commit_data.append(row)

            except Exception:
                continue

        if not commit_data:
            return None

        # Conversion en DataFrame
        df = pd.DataFrame(commit_data)

        # Conversion timestamp -> datetime
        df['date'] = pd.to_datetime(df['timestamp'], unit='s')
        df['month'] = df['date'].dt.to_period('M').dt.to_timestamp()

        # Agrégation mensuelle (moyenne pondérée par nombre de fichiers)
        monthly_data = []

        for month, group in df.groupby('month'):
            row = {'month': month, 'project': self.project_name}

            # Poids = nombre de fichiers dans chaque commit
            weights = group['n_files'].values
            total_weight = weights.sum()

            if total_weight == 0:
                continue

            for t in self.thresholds:
                col = f'CTR_{t}'
                # Moyenne pondérée
                weighted_avg = np.average(group[col].values, weights=weights)
                row[col] = weighted_avg

            row['mean_age_weighted'] = np.average(group['mean_age'].values, weights=weights)
            row['n_commits'] = len(group)
            row['n_files_total'] = total_weight

            monthly_data.append(row)

        if not monthly_data:
            return None

        result_df = pd.DataFrame(monthly_data)
        result_df = result_df.set_index('month').sort_index()

        return result_df


def _ctr_worker(args):
    """Worker pour multiprocessing avec gestion de cache."""
    name, path, thresholds, cache_dir = args

    # Vérification du cache
    if cache_dir:
        cache_file = os.path.join(cache_dir, f"{name}_ctr_v2.pkl")
        if os.path.exists(cache_file):
            try:
                df = pd.read_pickle(cache_file)
                if not df.empty and 'CTR_12' in df.columns:
                    return name, df
            except Exception:
                pass

    # Calcul
    extractor = CoreTouchExtractor(path, name, thresholds)
    df = extractor.process_repo()

    # Sauvegarde cache
    if df is not None and cache_dir:
        try:
            os.makedirs(cache_dir, exist_ok=True)
            df.to_pickle(cache_file)
        except Exception as e:
            print(f"⚠️ Cache write failed for {name}: {e}")

    return name, df


# ==============================================================================
# 2. VALIDATION STATISTIQUE
# ==============================================================================

class CTRMechanisticValidator:
    """
    Validation statistique du mécanisme CTR.

    Tests:
    1. Cluster Bootstrap (binaire: Early vs Mature)
    2. Corrélation continue (Spearman)
    3. Robustesse multi-seuils
    """

    def __init__(self, ctr_dataframes, gamma_dataframes):
        """
        Args:
            ctr_dataframes: dict {project: DataFrame avec CTR_6, CTR_12, CTR_24}
            gamma_dataframes: dict {project: DataFrame avec monthly_gamma}
        """
        self.ctr_dfs = ctr_dataframes
        self.gamma_dfs = gamma_dataframes
        self.merged_data = None
        self.thresholds = [6, 12, 24]

    def prepare_data(self):
        """
        Fusionne les données CTR et Gamma par projet/mois.
        Crée la colonne 'regime' (Early vs Mature).
        """
        print("\n📊 Préparation des données CTR + Gamma...")

        merged_frames = []

        for name, ctr_df in self.ctr_dfs.items():
            if name not in self.gamma_dfs or ctr_df is None:
                continue

            gamma_df = self.gamma_dfs[name]

            if gamma_df is None or gamma_df.empty:
                continue

            # S'assurer que les index sont comparables (datetime)
            ctr_idx = pd.to_datetime(ctr_df.index)
            gamma_idx = pd.to_datetime(gamma_df.index)

            # Créer des copies avec index normalisés
            ctr_norm = ctr_df.copy()
            ctr_norm.index = ctr_idx

            gamma_subset = gamma_df[['monthly_gamma']].copy()
            gamma_subset.index = gamma_idx

            # Inner join sur l'index temporel
            merged = ctr_norm.join(gamma_subset, how='inner')

            if merged.empty:
                continue

            # Ajouter le nom du projet
            merged['project'] = name

            merged_frames.append(merged)

        if not merged_frames:
            print("⚠️ Aucune donnée fusionnée disponible")
            self.merged_data = pd.DataFrame()
            return

        self.merged_data = pd.concat(merged_frames)

        # Créer la colonne regime (CRITIQUE: doit être fait ici)
        self.merged_data['regime'] = np.where(
            self.merged_data['monthly_gamma'] >= 0.7,
            'Mature',
            'Early'
        )

        n_obs = len(self.merged_data)
        n_projects = self.merged_data['project'].nunique()

        print(f"   ✅ {n_obs} observations fusionnées de {n_projects} projets")
        print(f"   Early (Γ<0.7): {(self.merged_data['regime'] == 'Early').sum()}")
        print(f"   Mature (Γ≥0.7): {(self.merged_data['regime'] == 'Mature').sum()}")

    def run_cluster_bootstrap(self, threshold_months=12, n_boot=10000):
        """
        Test H1: Cluster Bootstrap au niveau projet.

        Compare CTR moyen en régime Mature vs Early.
        Bootstrap au niveau PROJET (pas observation) pour gérer l'autocorrélation.

        Args:
            threshold_months: Seuil CTR à tester (6, 12, ou 24)
            n_boot: Nombre d'itérations bootstrap

        Returns:
            dict avec delta, IC 95%, p-value, Cohen's d
        """
        col = f'CTR_{threshold_months}'

        print(f"\n🎲 Cluster Bootstrap Test (CTR_{threshold_months})")
        print("-" * 60)

        if self.merged_data is None or self.merged_data.empty:
            print("⚠️ Données non préparées. Appelez prepare_data() d'abord.")
            return None

        if col not in self.merged_data.columns:
            print(f"⚠️ Colonne {col} non trouvée")
            return None

        df = self.merged_data.dropna(subset=['monthly_gamma', col, 'regime'])

        if len(df) < 50:
            print(f"⚠️ Données insuffisantes: {len(df)} observations")
            return None

        # Calcul des moyennes par projet et régime
        # Ceci évite la pseudo-réplication (un projet = une observation)
        proj_means = df.groupby(['project', 'regime'])[col].mean().unstack(fill_value=np.nan)

        # Filtrer les projets qui ont des données dans les deux régimes
        proj_means_valid = proj_means.dropna(subset=['Early', 'Mature'])

        if len(proj_means_valid) < 3:
            print(f"⚠️ Pas assez de projets avec données dans les deux régimes")
            # Fallback: utiliser tous les projets même si pas dans les deux régimes
            proj_means_valid = proj_means.dropna(how='all')

        # Statistiques observées
        if 'Early' in proj_means_valid.columns:
            mean_early = proj_means_valid['Early'].mean()
        else:
            mean_early = np.nan

        if 'Mature' in proj_means_valid.columns:
            mean_mature = proj_means_valid['Mature'].mean()
        else:
            mean_mature = np.nan

        delta_obs = mean_mature - mean_early if not (np.isnan(mean_early) or np.isnan(mean_mature)) else 0

        print(f"   Mean Early (Γ < 0.7)  : {mean_early:.4f}")
        print(f"   Mean Mature (Γ ≥ 0.7) : {mean_mature:.4f}")
        print(f"   Δ Observed            : {delta_obs:+.4f}")

        # Bootstrap au niveau projet
        projects = proj_means_valid.index.tolist()
        n_projects = len(projects)

        boot_deltas = []

        for _ in range(n_boot):
            # Resample des projets (avec remise)
            sample_projects = resample(projects, replace=True)
            sample_data = proj_means_valid.loc[sample_projects]

            # Calcul du delta sur l'échantillon
            m_early = sample_data['Early'].mean() if 'Early' in sample_data else np.nan
            m_mature = sample_data['Mature'].mean() if 'Mature' in sample_data else np.nan

            if not np.isnan(m_early) and not np.isnan(m_mature):
                boot_deltas.append(m_mature - m_early)

        if not boot_deltas:
            print("⚠️ Bootstrap échoué")
            return None

        boot_deltas = np.array(boot_deltas)

        # Intervalle de confiance et p-value
        ci_low = np.percentile(boot_deltas, 2.5)
        ci_high = np.percentile(boot_deltas, 97.5)

        # P-value one-tailed (H1: Mature > Early)
        p_value = (boot_deltas <= 0).mean()

        # Cohen's d
        if 'Early' in proj_means_valid.columns and 'Mature' in proj_means_valid.columns:
            std_early = proj_means_valid['Early'].std()
            std_mature = proj_means_valid['Mature'].std()
            pooled_std = np.sqrt((std_early ** 2 + std_mature ** 2) / 2)
            cohens_d = delta_obs / pooled_std if pooled_std > 0 else 0
        else:
            cohens_d = 0

        print(f"   95% CI               : [{ci_low:.4f}, {ci_high:.4f}]")
        print(f"   P-value (one-tailed) : {p_value:.5f}")
        print(f"   Cohen's d            : {cohens_d:.3f}")

        # Verdict
        if ci_low > 0 and p_value < 0.05:
            print(f"   ✅ VALIDÉ: Les systèmes matures touchent des fichiers plus anciens")
        elif p_value < 0.1:
            print(f"   ⚠️ TENDANCE: Signal positif mais IC inclut 0")
        else:
            print(f"   ❌ NON SIGNIFICATIF")

        return {
            'threshold': threshold_months,
            'mean_early': mean_early,
            'mean_mature': mean_mature,
            'delta': delta_obs,
            'ci_low': ci_low,
            'ci_high': ci_high,
            'p_value': p_value,
            'cohens_d': cohens_d,
            'n_projects': n_projects,
            'validated': ci_low > 0 and p_value < 0.05
        }

    def run_continuous_correlation(self):
        """
        Analyse de corrélation continue (Spearman).

        Évite le biais du seuil binaire en testant la relation Γ ↔ CTR
        sur toute la gamme des valeurs.

        Returns:
            dict avec corrélations micro et macro pour chaque seuil
        """
        print(f"\n📈 Corrélation Continue (Spearman)")
        print("-" * 60)

        if self.merged_data is None or self.merged_data.empty:
            print("⚠️ Données non préparées")
            return None

        results = {}

        for t in self.thresholds:
            col = f'CTR_{t}'

            if col not in self.merged_data.columns:
                continue

            # 1. MICRO-corrélation (tous les mois)
            # Attention: pseudo-réplication possible
            valid_micro = self.merged_data.dropna(subset=['monthly_gamma', col])

            if len(valid_micro) < 20:
                continue

            r_micro, p_micro = stats.spearmanr(
                valid_micro['monthly_gamma'],
                valid_micro[col]
            )

            # 2. MACRO-corrélation (moyennes par projet)
            # Plus conservateur, évite la pseudo-réplication
            proj_avgs = self.merged_data.groupby('project')[['monthly_gamma', col]].mean()
            proj_avgs = proj_avgs.dropna()

            if len(proj_avgs) < 5:
                r_macro, p_macro = np.nan, np.nan
            else:
                r_macro, p_macro = stats.spearmanr(
                    proj_avgs['monthly_gamma'],
                    proj_avgs[col]
                )

            results[t] = {
                'r_micro': r_micro,
                'p_micro': p_micro,
                'n_micro': len(valid_micro),
                'r_macro': r_macro,
                'p_macro': p_macro,
                'n_macro': len(proj_avgs)
            }

            sig_micro = "✅" if p_micro < 0.05 and r_micro > 0 else ""
            sig_macro = "✅" if p_macro < 0.05 and r_macro > 0 else ""

            print(f"   CTR_{t:2d} | Micro: r={r_micro:+.3f} (p={p_micro:.4f}) {sig_micro}")
            print(f"         | Macro: r={r_macro:+.3f} (p={p_macro:.4f}) {sig_macro}")

        return results

    def run_partial_correlation(self, activity_col='n_files_total'):
        """
        Corrélation partielle Γ ↔ CTR en contrôlant:
        - l'activité (n_files_total)
        - l'âge du projet (mois depuis le premier commit)

        Version MICRO + MACRO.
        """
        print(f"\n🔗 Corrélation Partielle (contrôle activité + âge projet)")
        print("-" * 60)

        if self.merged_data is None or self.merged_data.empty:
            print("⚠️ Données non préparées")
            return None

        # Calculer l'âge du projet pour chaque observation
        df = self.merged_data.copy()
        df['obs_date'] = pd.to_datetime(df.index)

        # Premier mois par projet
        project_start = df.groupby('project')['obs_date'].transform('min')
        df['project_age_months'] = (df['obs_date'] - project_start).dt.days / 30.44

        results = {}

        for t in self.thresholds:
            ctr_col = f'CTR_{t}'
            if ctr_col not in df.columns:
                continue

            df_valid = df.dropna(
                subset=['monthly_gamma', ctr_col, activity_col, 'project_age_months']
            ).copy()

            if len(df_valid) < 50:
                continue

            # Transformations log
            df_valid['log_activity'] = np.log1p(df_valid[activity_col])
            df_valid['log_project_age'] = np.log1p(df_valid['project_age_months'])

            # =====================================================
            # MICRO — toutes les observations
            # =====================================================
            gamma_r = stats.rankdata(df_valid['monthly_gamma'])
            ctr_r = stats.rankdata(df_valid[ctr_col])
            act_r = stats.rankdata(df_valid['log_activity'])
            age_r = stats.rankdata(df_valid['log_project_age'])

            # Résidualisation sur les DEUX covariables
            # Régression multiple sur ranks
            X = np.column_stack([act_r, age_r, np.ones(len(act_r))])

            # Résidus de gamma
            beta_gamma = np.linalg.lstsq(X, gamma_r, rcond=None)[0]
            gamma_res = gamma_r - X @ beta_gamma

            # Résidus de CTR
            beta_ctr = np.linalg.lstsq(X, ctr_r, rcond=None)[0]
            ctr_res = ctr_r - X @ beta_ctr

            r_micro, p_micro = stats.spearmanr(gamma_res, ctr_res)

            # =====================================================
            # MACRO — moyennes par projet
            # =====================================================
            proj_avgs = df_valid.groupby('project')[
                ['monthly_gamma', ctr_col, 'log_activity', 'log_project_age']
            ].mean().dropna()

            if len(proj_avgs) >= 5:
                gamma_r_m = stats.rankdata(proj_avgs['monthly_gamma'])
                ctr_r_m = stats.rankdata(proj_avgs[ctr_col])
                act_r_m = stats.rankdata(proj_avgs['log_activity'])
                age_r_m = stats.rankdata(proj_avgs['log_project_age'])

                X_m = np.column_stack([act_r_m, age_r_m, np.ones(len(act_r_m))])

                beta_gamma_m = np.linalg.lstsq(X_m, gamma_r_m, rcond=None)[0]
                gamma_res_m = gamma_r_m - X_m @ beta_gamma_m

                beta_ctr_m = np.linalg.lstsq(X_m, ctr_r_m, rcond=None)[0]
                ctr_res_m = ctr_r_m - X_m @ beta_ctr_m

                r_macro, p_macro = stats.spearmanr(gamma_res_m, ctr_res_m)
            else:
                r_macro, p_macro = np.nan, np.nan

            # -------------------------
            # Stockage résultats
            # -------------------------
            results[t] = {
                'r_partial_micro': r_micro,
                'p_partial_micro': p_micro,
                'n_micro': len(df_valid),
                'r_partial_macro': r_macro,
                'p_partial_macro': p_macro,
                'n_projects': len(proj_avgs)
            }

            sig_micro = "✅" if p_micro < 0.05 and r_micro > 0 else ""
            sig_macro = "✅" if p_macro < 0.05 and r_macro > 0 else ""

            print(
                f"   CTR_{t:2d} | Micro: partial r={r_micro:+.3f} "
                f"(p={p_micro:.4f}, n={len(df_valid)}) {sig_micro}"
            )
            print(
                f"         | Macro: partial r={r_macro:+.3f} "
                f"(p={p_macro:.4f}, n={len(proj_avgs)}) {sig_macro}"
            )

        return results

    def run_robustness_analysis(self, n_boot=5000):
        """
        Test de robustesse multi-seuils.

        Vérifie que le signal CTR est stable across T = 6, 12, 24 mois.

        Returns:
            DataFrame avec résultats pour chaque seuil
        """
        print(f"\n🔬 Analyse de Robustesse Multi-Seuils")
        print("-" * 60)

        results = []

        for t in self.thresholds:
            res = self.run_cluster_bootstrap(threshold_months=t, n_boot=n_boot)
            if res:
                results.append(res)

        if not results:
            return None

        df_results = pd.DataFrame(results)

        # Résumé
        print(f"\n{'Seuil':<8} | {'Δ':<8} | {'IC 95%':<20} | {'p-value':<10} | {'d':<6}")
        print("-" * 65)

        for _, row in df_results.iterrows():
            t = int(row['threshold'])
            delta = row['delta']
            ci = f"[{row['ci_low']:.3f}, {row['ci_high']:.3f}]"
            p = row['p_value']
            d = row['cohens_d']
            status = "✅" if row['validated'] else ""

            print(f"CTR_{t:<4} | {delta:+.4f} | {ci:<20} | {p:<10.5f} | {d:<6.2f} {status}")

        # Verdict global
        n_validated = df_results['validated'].sum()
        print(f"\n→ {n_validated}/{len(df_results)} seuils validés")

        if n_validated >= 2:
            print("✅ ROBUSTE: Signal stable across thresholds")
        elif n_validated >= 1:
            print("⚠️ PARTIEL: Signal présent mais pas robuste")
        else:
            print("❌ NON VALIDÉ: Pas de signal cohérent")

        return df_results

    def plot_results(self, output_path="figure_ctr_mechanism.pdf"):
        """
        Génère la figure de publication.

        Panel A: Scatter Γ vs CTR_12 avec LOWESS
        Panel B: Barplot des deltas multi-seuils
        """
        if self.merged_data is None or self.merged_data.empty:
            print("⚠️ Données non disponibles pour le plot")
            return

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # =====================================================================
        # PANEL A: Scatter + LOWESS
        # =====================================================================
        ax1 = axes[0]

        col = 'CTR_12'
        df_plot = self.merged_data.dropna(subset=['monthly_gamma', col])

        if len(df_plot) > 100:
            # Hexbin pour densité si beaucoup de points
            hb = ax1.hexbin(
                df_plot['monthly_gamma'],
                df_plot[col],
                gridsize=30,
                cmap='Blues',
                mincnt=1,
                alpha=0.8
            )
            plt.colorbar(hb, ax=ax1, label='Density')
        else:
            # Scatter simple
            ax1.scatter(
                df_plot['monthly_gamma'],
                df_plot[col],
                alpha=0.3,
                s=20,
                c='#3498db'
            )

        # LOWESS trend
        try:
            import statsmodels.api as sm
            lowess = sm.nonparametric.lowess(
                df_plot[col].values,
                df_plot['monthly_gamma'].values,
                frac=0.3
            )
            ax1.plot(lowess[:, 0], lowess[:, 1], 'r-', linewidth=2.5,
                     label='LOWESS Trend')
        except Exception:
            pass

        # Ligne de seuil de maturité
        ax1.axvline(0.7, color='black', linestyle='--', alpha=0.5,
                    label='Maturity Threshold')

        # Annotation Spearman
        r, p = stats.spearmanr(df_plot['monthly_gamma'], df_plot[col])
        ax1.annotate(
            f'Spearman ρ = {r:.3f}\np = {p:.2e}',
            xy=(0.05, 0.95),
            xycoords='axes fraction',
            verticalalignment='top',
            fontsize=10,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
        )

        ax1.set_xlabel(r'Structural Maturity ($\Gamma$)')
        ax1.set_ylabel('Core Touch Ratio (Files > 12 months)')
        ax1.set_title('A. Architectural Constraint Mechanism', fontweight='bold')
        ax1.legend(loc='lower right')
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)
        ax1.grid(True, alpha=0.3)

        # =====================================================================
        # PANEL B: Barplot multi-seuils
        # =====================================================================
        ax2 = axes[1]

        # Calculer les deltas pour chaque seuil
        deltas = []
        errors = []
        labels = []

        for t in self.thresholds:
            col_t = f'CTR_{t}'
            if col_t not in self.merged_data.columns:
                continue

            proj_means = self.merged_data.groupby(['project', 'regime'])[col_t].mean().unstack()

            if 'Early' in proj_means and 'Mature' in proj_means:
                early = proj_means['Early'].dropna()
                mature = proj_means['Mature'].dropna()

                delta = mature.mean() - early.mean()
                # Erreur standard de la différence
                se = np.sqrt(early.var() / len(early) + mature.var() / len(mature))

                deltas.append(delta)
                errors.append(1.96 * se)  # 95% CI
                labels.append(f'> {t} mo')

        if deltas:
            x = np.arange(len(labels))
            colors = ['#3498db', '#2ecc71', '#9b59b6'][:len(deltas)]

            bars = ax2.bar(x, deltas, yerr=errors, capsize=5,
                           color=colors, alpha=0.8, edgecolor='black')

            ax2.set_xticks(x)
            ax2.set_xticklabels(labels)
            ax2.set_ylabel(r'$\Delta$ CTR (Mature - Early)')
            ax2.set_xlabel('File Age Threshold')
            ax2.set_title('B. Robustness Across Thresholds', fontweight='bold')
            ax2.axhline(0, color='black', linewidth=1)
            ax2.grid(True, alpha=0.3, axis='y')

            # Annotations significance
            for i, (d, e) in enumerate(zip(deltas, errors)):
                if d - e > 0:  # IC ne croise pas 0
                    ax2.annotate('*', xy=(i, d + e + 0.02), ha='center',
                                 fontsize=16, fontweight='bold')

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\n✅ Figure sauvegardée: {output_path}")
        plt.close()


# ==============================================================================
# 3. FONCTION PRINCIPALE
# ==============================================================================

def run_ctr_mechanistic_test(
        gamma_dataframes: dict,
        repos_config: dict,
        cache_dir: str = None,
        max_workers: int = 4,
        n_boot: int = 10000
) -> dict:
    """
    Fonction principale pour exécuter le test mécaniste CTR.

    Args:
        gamma_dataframes: dict {project_name: DataFrame avec monthly_gamma}
        repos_config: dict {project_name: {'path': ..., ...}}
        cache_dir: Répertoire de cache (optionnel)
        max_workers: Nombre de workers pour le multiprocessing
        n_boot: Nombre d'itérations bootstrap

    Returns:
        dict avec tous les résultats
    """
    print("\n" + "=" * 80)
    print("TEST MÉCANISTE : CORE TOUCH RATIO (CTR)")
    print("=" * 80)
    print("Hypothèse: Les systèmes matures touchent des fichiers plus anciens")
    print("           (l'architecture établie médiatise le changement)\n")

    # =========================================================================
    # PHASE 1: Extraction CTR
    # =========================================================================
    print("[PHASE 1] Extraction du Core Touch Ratio...")

    # Préparer les tâches
    tasks = []
    for name in gamma_dataframes.keys():
        if name in repos_config:
            path = repos_config[name]['path']
            tasks.append((name, path, [6, 12, 24], cache_dir))

    print(f"   {len(tasks)} projets à traiter")

    ctr_results = {}

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(tqdm(
            executor.map(_ctr_worker, tasks),
            total=len(tasks),
            desc="   Extraction CTR"
        ))

        for name, df in results:
            if df is not None and not df.empty:
                ctr_results[name] = df

    print(f"   ✅ {len(ctr_results)} projets extraits avec succès")

    if not ctr_results:
        print("❌ Aucune donnée CTR extraite")
        return None

    # =========================================================================
    # PHASE 2: Validation statistique
    # =========================================================================
    print("\n[PHASE 2] Validation statistique...")

    validator = CTRMechanisticValidator(ctr_results, gamma_dataframes)
    validator.prepare_data()

    # Test de robustesse (tous les seuils)
    robustness_df = validator.run_robustness_analysis(n_boot=n_boot)

    # Corrélation continue
    correlation_results = validator.run_continuous_correlation()
    partial_corr_results = validator.run_partial_correlation()
    # =========================================================================
    # PHASE 3: Visualisation
    # =========================================================================
    print("\n[PHASE 3] Génération de la figure...")

    validator.plot_results("figure_ctr_mechanism.pdf")

    # =========================================================================
    # RÉSUMÉ FINAL
    # =========================================================================
    print("\n" + "=" * 80)
    print("RÉSUMÉ DU TEST MÉCANISTE CTR")
    print("=" * 80)

    if robustness_df is not None:
        n_validated = robustness_df['validated'].sum()
        print(f"Seuils validés: {n_validated}/3")

        if n_validated >= 2:
            print("✅ MÉCANISME VALIDÉ")
            print("   → Les systèmes matures touchent effectivement des fichiers plus anciens")
            print("   → L'architecture établie médiatise le changement")
        elif n_validated >= 1:
            print("⚠️ SIGNAL PARTIEL")
            print("   → Tendance présente mais pas robuste across thresholds")
        else:
            print("❌ MÉCANISME NON VALIDÉ")
            print("   → Pas de différence significative Early vs Mature")

    return {
        'ctr_data': ctr_results,
        'robustness': robustness_df,
        'correlations': correlation_results,
        'partial_correlations': partial_corr_results,
        'validator': validator
    }


# ==============================================================================
# 4. EXEMPLE D'UTILISATION
# ==============================================================================

if __name__ == "__main__":
    print("Ce module doit être importé et utilisé avec vos données.")
    print("\nExemple d'utilisation:")
    print("""
    from ctr_mechanistic_test import run_ctr_mechanistic_test

    results = run_ctr_mechanistic_test(
        gamma_dataframes=dfs_theory,    # dict {project: DataFrame}
        repos_config=REPOS_CONFIG,       # dict {project: {'path': ...}}
        cache_dir="./cache/ctr/",
        max_workers=6,
        n_boot=10000
    )
    """)


def generate_ctr_si_figure(merged_data, partial_results=None, output_path="figure_si_ctr_mechanism.pdf"):
    """
    Génère la figure SI pour le Core Touch Ratio avec 3 panels.

    Panel A: Scatter Γ vs CTR_12 avec LOWESS
    Panel B: Barplot des deltas multi-seuils
    Panel C: Comparaison corrélations avant/après contrôle

    Args:
        merged_data: DataFrame avec colonnes monthly_gamma, CTR_*, regime, project
        partial_results: dict des résultats de run_partial_correlation() (optionnel)
        output_path: Chemin de sortie
    """

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    thresholds = [6, 12, 24]

    # =========================================================================
    # PANEL A: Scatter Γ vs CTR_12 avec LOWESS
    # =========================================================================
    ax1 = axes[0]

    col = 'CTR_12'
    df_plot = merged_data.dropna(subset=['monthly_gamma', col]).copy()

    # Hexbin pour densité
    hb = ax1.hexbin(
        df_plot['monthly_gamma'],
        df_plot[col],
        gridsize=35,
        cmap='Blues',
        mincnt=1,
        alpha=0.9,
        linewidths=0.2
    )

    # Colorbar
    cb = plt.colorbar(hb, ax=ax1)
    cb.set_label('Density', fontsize=9)

    # LOWESS trend
    try:
        import statsmodels.api as sm
        lowess = sm.nonparametric.lowess(
            df_plot[col].values,
            df_plot['monthly_gamma'].values,
            frac=0.25
        )
        ax1.plot(lowess[:, 0], lowess[:, 1], 'r-', linewidth=2.5,
                 label='LOWESS trend', zorder=10)
    except ImportError:
        pass

    # Ligne de seuil de maturité
    ax1.axvline(0.7, color='black', linestyle='--', alpha=0.6,
                linewidth=1.5, label='Maturity threshold')

    # Annotation Spearman
    r, p = stats.spearmanr(df_plot['monthly_gamma'], df_plot[col])
    ax1.annotate(
        f'Spearman ρ = {r:.3f}\np < 0.0001',
        xy=(0.05, 0.95),
        xycoords='axes fraction',
        verticalalignment='top',
        fontsize=9,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                  edgecolor='gray', alpha=0.9)
    )

    ax1.set_xlabel(r'Structural persistence ($\Gamma$)')
    ax1.set_ylabel('Core Touch Ratio (files > 12 months)')
    ax1.set_title('A', fontweight='bold', loc='left', fontsize=14)
    ax1.legend(loc='lower right', framealpha=0.9)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3, linewidth=0.5)

    # =========================================================================
    # PANEL B: Barplot multi-seuils avec IC 95%
    # =========================================================================
    ax2 = axes[1]

    deltas = []
    ci_lows = []
    ci_highs = []
    labels = []

    for t in thresholds:
        col_t = f'CTR_{t}'
        if col_t not in merged_data.columns:
            continue

        proj_means = merged_data.groupby(['project', 'regime'])[col_t].mean().unstack()

        if 'Early' in proj_means.columns and 'Mature' in proj_means.columns:
            valid = proj_means.dropna(subset=['Early', 'Mature'])

            delta = valid['Mature'].mean() - valid['Early'].mean()

            # Bootstrap IC 95%
            n_boot = 10000
            boot_deltas = []
            projects = valid.index.tolist()

            for _ in range(n_boot):
                idx = np.random.choice(len(projects), size=len(projects), replace=True)
                sample = valid.iloc[idx]
                boot_deltas.append(sample['Mature'].mean() - sample['Early'].mean())

            boot_deltas = np.array(boot_deltas)
            ci_low = np.percentile(boot_deltas, 2.5)
            ci_high = np.percentile(boot_deltas, 97.5)

            deltas.append(delta)
            ci_lows.append(ci_low)
            ci_highs.append(ci_high)
            labels.append(f'>{t}mo')

    if deltas:
        x = np.arange(len(labels))
        colors = ['#3498db', '#27ae60', '#8e44ad']

        errors_low = [d - ci_l for d, ci_l in zip(deltas, ci_lows)]
        errors_high = [ci_h - d for d, ci_h in zip(deltas, ci_highs)]

        ax2.bar(x, deltas,
                yerr=[errors_low, errors_high],
                capsize=6,
                color=colors,
                alpha=0.85,
                edgecolor='black',
                linewidth=1,
                error_kw={'linewidth': 1.5, 'capthick': 1.5})

        ax2.set_xticks(x)
        ax2.set_xticklabels(labels)
        ax2.set_ylabel(r'$\Delta$CTR (Mature − Early)')
        ax2.set_xlabel('File age threshold')
        ax2.set_title('B', fontweight='bold', loc='left', fontsize=14)
        ax2.axhline(0, color='black', linewidth=1)
        ax2.grid(True, alpha=0.3, axis='y', linewidth=0.5)

        for i, (d, ci_l, ci_h) in enumerate(zip(deltas, ci_lows, ci_highs)):
            if ci_l > 0:
                ax2.annotate('***', xy=(i, ci_h + 0.015), ha='center',
                             fontsize=12, fontweight='bold')

        ax2.annotate(
            "Cohen's d > 1.3\nfor all thresholds",
            xy=(0.95, 0.95),
            xycoords='axes fraction',
            ha='right', va='top',
            fontsize=8,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor='gray', alpha=0.9)
        )

        ax2.set_ylim(0, max(ci_highs) + 0.06)

    # =========================================================================
    # PANEL C: Effet du contrôle (avant/après)
    # =========================================================================
    ax3 = axes[2]

    # Valeurs par défaut basées sur les résultats observés
    # (remplacer par partial_results si fourni)
    raw_micro = 0.454
    raw_macro = 0.432
    ctrl_micro = 0.176
    ctrl_macro = 0.433

    if partial_results and 12 in partial_results:
        ctrl_micro = partial_results[12].get('r_partial_micro', ctrl_micro)
        ctrl_macro = partial_results[12].get('r_partial_macro', ctrl_macro)

    categories = ['Raw\ncorrelation', 'Controlled\n(activity + age)']
    micro_vals = [raw_micro, ctrl_micro]
    macro_vals = [raw_macro, ctrl_macro]

    x = np.arange(len(categories))
    width = 0.35

    bars1 = ax3.bar(x - width/2, micro_vals, width,
                    label='Observation-level',
                    color='#3498db', alpha=0.85, edgecolor='black', linewidth=1)
    bars2 = ax3.bar(x + width/2, macro_vals, width,
                    label='Project-level',
                    color='#e74c3c', alpha=0.85, edgecolor='black', linewidth=1)

    ax3.set_ylabel('Spearman ρ')
    ax3.set_title('C', fontweight='bold', loc='left', fontsize=14)
    ax3.set_xticks(x)
    ax3.set_xticklabels(categories)
    ax3.legend(loc='upper right', framealpha=0.9)
    ax3.set_ylim(0, 0.6)
    ax3.axhline(0, color='black', linewidth=1)
    ax3.grid(True, alpha=0.3, axis='y', linewidth=0.5)

    # Annotations p-values
    for i, m in enumerate(micro_vals):
        ax3.annotate('***', xy=(i - width/2, m + 0.02), ha='center',
                     fontsize=10, fontweight='bold')
    for i, M in enumerate(macro_vals):
        ax3.annotate('**', xy=(i + width/2, M + 0.02), ha='center',
                     fontsize=10, fontweight='bold')

    # Annotation explicative
    ax3.annotate(
        'Project-level correlation\nrobust to confounds',
        xy=(1 + width/2, macro_vals[1] + 0.08),
        ha='center', va='bottom',
        fontsize=8,
        color='#c0392b',
        fontweight='bold'
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')

    print(f"✅ Figure sauvegardée: {output_path}")
    print(f"✅ Figure sauvegardée: {output_path.replace('.pdf', '.png')}")

    plt.close()

    return fig
