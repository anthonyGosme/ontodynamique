import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import git
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Tuple, Optional

# Constantes de configuration
GAMMA_THRESHOLD = 0.7
WINDOW_PRE = 6
WINDOW_POST = 12
BOOTSTRAP_ITERATIONS = 1000


class CrashTestAnalyzer:
    """
    Analyseur de perturbation longitudinale (Crash Test / Superimposed Epoch Analysis).

    Version : V2 (Scientifiquement Robustifiée)
    - Gestion des trous d'historique (Mois sans commits)
    - Alignement strict des dates (Month Start)
    - Dé-clustering des événements (Séparation min. 6 mois)
    """

    def __init__(self, repo_path: str, time_series_df: pd.DataFrame):
        """
        :param repo_path: Chemin vers le clone git local.
        :param time_series_df: DataFrame contenant 'gamma' et 'authority_index',
                               indexé par Datetime (au 1er du mois).
        """
        self.repo_path = repo_path
        # On s'assure que l'index est trié
        self.metrics = time_series_df.sort_index()

        # CORRECTION ICI : Initialisation Git tolérante aux erreurs
        # Permet de créer une instance "Dummy" pour l'agrégation sans repo valide
        try:
            self.repo = git.Repo(repo_path)
        except (git.exc.InvalidGitRepositoryError, git.exc.NoSuchPathError):
            self.repo = None

    def detect_crashes_by_regime(
            self,
            churn_df: pd.DataFrame,
            min_separation_months: int = 6,
            q: float = 0.95
    ) -> Dict[str, List[pd.Timestamp]]:
        """
        Détecte des chocs de churn de façon conditionnelle au régime (exploratory vs mature).
        Retourne un dict: {'exploratory': [...], 'mature': [...]}.
        """
        if churn_df.empty or self.metrics.empty:
            return {'exploratory': [], 'mature': []}

        # Aligner churn et metrics sur l'index commun
        aligned = churn_df.join(self.metrics[['gamma']], how='inner').dropna()
        if aligned.empty:
            return {'exploratory': [], 'mature': []}

        results = {}

        for regime_name, mask in [
            ('exploratory', aligned['gamma'] < GAMMA_THRESHOLD),
            ('mature', aligned['gamma'] >= GAMMA_THRESHOLD),
        ]:
            sub = aligned[mask]
            if len(sub) < 12:
                results[regime_name] = []
                continue

            thr = sub['churn'].quantile(q)
            candidates = sub[sub['churn'] >= thr][['churn']].sort_values('churn', ascending=False)

            # déclustering greedy (identique à ton detect_crashes)
            final = []
            for date in candidates.index:
                too_close = False
                for acc in final:
                    diff_months = abs((date.year - acc.year) * 12 + (date.month - acc.month))
                    if diff_months < min_separation_months:
                        too_close = True
                        break
                if not too_close:
                    final.append(date)

            results[regime_name] = sorted(final)

        return results

    def compute_contributor_sets(self) -> Dict[pd.Timestamp, set]:
        """
        Extrait l'ensemble des contributeurs actifs A(t) sur une grille mensuelle CONTINUE.

        CORRECTION CRITIQUE :
        Si aucun commit n'est fait un mois donné, ce mois doit exister dans le dictionnaire
        avec un ensemble vide A(t) = set(), sinon le calcul du churn (différence) sera faux.
        """
        print(f"[{self.repo_path}] Extraction des ensembles de contributeurs...")

        # 1. Récupération brute des auteurs par date
        raw_commits = []
        # Optimisation : on ne lit que les headers nécessaires
        for commit in self.repo.iter_commits():
            # Conversion Timestamp Git -> Datetime -> Timestamp Pandas (Month Start)
            dt = datetime.fromtimestamp(commit.committed_date)
            month_start = pd.Timestamp(dt).to_period('M').to_timestamp()

            email = commit.author.email.lower().strip()
            raw_commits.append((month_start, email))

        if not raw_commits:
            return {}

        # 2. Définition de la plage temporelle complète (min -> max)
        # On prend le min des commits et le max des métriques pour couvrir toute la période d'analyse
        start_date = min(c[0] for c in raw_commits)
        end_date = max(c[0] for c in raw_commits)

        if not self.metrics.empty:
            start_date = min(start_date, self.metrics.index.min())
            end_date = max(end_date, self.metrics.index.max())

        # Création de la grille mensuelle stricte (MS = Month Start)
        full_range = pd.date_range(start=start_date, end=end_date, freq='MS')

        # 3. Remplissage des ensembles (avec gestion des vides)
        monthly_contributors = {date: set() for date in full_range}

        for date, email in raw_commits:
            if date in monthly_contributors:
                monthly_contributors[date].add(email)

        return monthly_contributors

    def compute_churn_metrics(self, active_sets: Dict[pd.Timestamp, set]) -> pd.DataFrame:
        """
        Calcule le Turnover/Churn : |In(t)| + |Out(t)|.
        Restreint le résultat aux mois présents dans self.metrics pour garantir l'analyse.
        """
        sorted_dates = sorted(active_sets.keys())
        churn_data = []

        for i in range(1, len(sorted_dates)):
            t_curr = sorted_dates[i]
            t_prev = sorted_dates[i - 1]

            # Si ce mois n'est pas dans nos métriques (Gamma/AI), on ne le calcule pas
            # car on ne pourra pas extraire la fenêtre d'analyse [-6, +12] autour.
            if t_curr not in self.metrics.index:
                continue

            set_curr = active_sets[t_curr]
            set_prev = active_sets[t_prev]

            # Calcul ensembliste
            in_t = len(set_curr - set_prev)
            out_t = len(set_prev - set_curr)

            churn_val = in_t + out_t

            churn_data.append({
                'month': t_curr,
                'churn': churn_val
            })

        return pd.DataFrame(churn_data).set_index('month')

    def detect_crashes(self, churn_df: pd.DataFrame, min_separation_months=6) -> List[pd.Timestamp]:
        """
        Identifie les mois Q95 avec DÉ-CLUSTERING.

        Problème résolu : Une crise dure souvent 2-3 mois. On ne veut pas déclencher
        3 événements séparés. On garde le pic maximal dans une fenêtre glissante.
        """
        if churn_df.empty:
            return []

        q95 = churn_df['churn'].quantile(0.95)

        # Candidats au-dessus du seuil
        candidates = churn_df[churn_df['churn'] >= q95].copy()

        if candidates.empty:
            return []

        # Tri par intensité de churn décroissante (Greedy approach : on prend le pire d'abord)
        candidates = candidates.sort_values('churn', ascending=False)

        final_crashes = []

        for date in candidates.index:
            # Vérifier si ce candidat est trop proche d'un crash déjà validé
            is_too_close = False
            for accepted_date in final_crashes:
                # Calcul de distance en mois approximatif mais suffisant
                diff_months = abs((date.year - accepted_date.year) * 12 + (date.month - accepted_date.month))

                if diff_months < min_separation_months:
                    is_too_close = True
                    break

            if not is_too_close:
                final_crashes.append(date)

        # On retourne la liste triée chronologiquement pour l'extraction
        return sorted(final_crashes)

    def extract_event_windows(self, crash_months: List[pd.Timestamp]) -> pd.DataFrame:
        """
        Extrait les fenêtres [-6, +12] autour des crashs.
        Calcule les DELTAS (Différence par rapport à t=0).
        """
        extracted_data = []

        for t0 in crash_months:
            if t0 not in self.metrics.index:
                continue

            # 1. Détermination du Régime à l'instant t0
            gamma_t0 = self.metrics.loc[t0, 'gamma']
            regime = "mature" if gamma_t0 >= GAMMA_THRESHOLD else "exploratory"

            # Valeurs de base à t0 (pour le calcul des deltas)
            base_gamma = self.metrics.loc[t0, 'gamma']
            base_auth = self.metrics.loc[t0, 'authority_index']

            # 2. Extraction de la fenêtre
            window_valid = True
            event_traces = []

            for offset in range(-WINDOW_PRE, WINDOW_POST + 1):
                # Utilisation robuste de DateOffset sur Timestamp
                target_month = t0 + pd.DateOffset(months=offset)

                # Si le mois cible sort de l'historique dispo, la fenêtre est invalide
                if target_month not in self.metrics.index:
                    window_valid = False
                    break

                curr_gamma = self.metrics.loc[target_month, 'gamma']
                curr_auth = self.metrics.loc[target_month, 'authority_index']

                # Calcul des Deltas (Réponse Impulse-Response)
                event_traces.append({
                    'regime': regime,
                    'event_month': t0,
                    'delta_t': offset,
                    'delta_gamma': curr_gamma - base_gamma,
                    'delta_auth': curr_auth - base_auth
                })

            if window_valid:
                extracted_data.extend(event_traces)

        return pd.DataFrame(extracted_data)

    def _get_bootstrap_ci(self, data: np.array) -> Tuple[float, float]:
        """Calcule l'Intervalle de Confiance à 95% par Bootstrap."""
        if len(data) < 2:
            return np.nan, np.nan

        means = []
        # Vectorisation simple si possible, sinon boucle
        for _ in range(BOOTSTRAP_ITERATIONS):
            sample = np.random.choice(data, size=len(data), replace=True)
            means.append(np.mean(sample))

        return np.percentile(means, 2.5), np.percentile(means, 97.5)

    def aggregate_events(self, traces_df: pd.DataFrame) -> pd.DataFrame:
        """
        Agrège les traces par Régime et par Delta_T.
        Calcule Moyenne, Médiane et IC 95%.
        """
        if traces_df.empty:
            return pd.DataFrame()

        results = []

        # Groupement par Régime et Décalage Temporel
        grouped = traces_df.groupby(['regime', 'delta_t'])

        for (regime, dt), group in grouped:
            # Métriques
            d_gamma = group['delta_gamma'].dropna().values
            d_auth = group['delta_auth'].dropna().values

            if len(d_gamma) == 0: continue

            # Statistiques Gamma
            mean_dg = np.mean(d_gamma)
            ci_dg = self._get_bootstrap_ci(d_gamma)

            # Statistiques Authority (La métrique clé)
            mean_da = np.mean(d_auth)
            ci_da = self._get_bootstrap_ci(d_auth)

            results.append({
                'regime': regime,
                'delta_t': dt,
                'n_events': len(group),

                'mean_delta_gamma': mean_dg,
                'ci_lower_gamma': ci_dg[0],
                'ci_upper_gamma': ci_dg[1],

                'mean_delta_auth': mean_da,
                'ci_lower_auth': ci_da[0],
                'ci_upper_auth': ci_da[1]
            })

        return pd.DataFrame(results)

    def plot_crash_test_response(self, agg_df: pd.DataFrame):
        """
        Génère les graphiques SEA (Superimposed Epoch Analysis).
        Panel A : Réponse Structurelle (Gamma)
        Panel B : Réponse Causale (Authority Index) - Prioritaire
        """
        if agg_df.empty:
            print("⚠️ Pas de données à tracer.")
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharex=True)

        # Couleurs cohérentes avec le reste du papier
        styles = {
            'exploratory': {'color': '#3498db', 'label': 'Exploratory (Γ < 0.7)'},
            'mature': {'color': '#e74c3c', 'label': 'Mature (Γ ≥ 0.7)'}
        }

        # --- PANEL A: Réponse Gamma ---
        for regime in ['exploratory', 'mature']:
            subset = agg_df[agg_df['regime'] == regime].sort_values('delta_t')
            if subset.empty: continue

            s = styles[regime]
            ax1.plot(subset['delta_t'], subset['mean_delta_gamma'],
                     color=s['color'], label=s['label'], linewidth=2.5)
            ax1.fill_between(subset['delta_t'],
                             subset['ci_lower_gamma'], subset['ci_upper_gamma'],
                             color=s['color'], alpha=0.15)

        ax1.axvline(0, color='black', linestyle='--', linewidth=1)
        ax1.axhline(0, color='gray', linewidth=0.8)
        ax1.set_title("Panel A: Structural Response ($\Delta \Gamma$)", fontweight='bold')
        ax1.set_xlabel("Months from Turnover Event")
        ax1.set_ylabel("Change in Structure")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # --- PANEL B: Réponse Authority (Causale) ---
        for regime in ['exploratory', 'mature']:
            subset = agg_df[agg_df['regime'] == regime].sort_values('delta_t')
            if subset.empty: continue

            s = styles[regime]
            ax2.plot(subset['delta_t'], subset['mean_delta_auth'],
                     color=s['color'], label=s['label'], linewidth=2.5)
            ax2.fill_between(subset['delta_t'],
                             subset['ci_lower_auth'], subset['ci_upper_auth'],
                             color=s['color'], alpha=0.15)

        ax2.axvline(0, color='black', linestyle='--', linewidth=1)
        ax2.axhline(0, color='gray', linewidth=0.8)
        ax2.set_title("Panel B: Causal Response ($\Delta AI$)", fontweight='bold')
        ax2.set_xlabel("Months from Turnover Event")
        ax2.set_ylabel("Change in Authority Index")
        # Zone de dominance (pour interprétation)
        ax2.text(8, 0.1, "Structure Gains Control", fontsize=8, alpha=0.5)
        ax2.text(8, -0.1, "Activity Gains Control", fontsize=8, alpha=0.5)

        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        filename = "omega_v36_crash_test_SEA.pdf"
        plt.savefig(filename, format="pdf", dpi=300)
        print(f"✅ Graphique SEA sauvegardé : {filename}")
        #plt.show()
        plt.close()