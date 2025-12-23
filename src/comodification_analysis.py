"""
================================================================================
MODULE V42: CO-MODIFICATION COUPLING ANALYSIS (WITH BIAS CONTROL)
================================================================================
Validates the structural constraint mechanism by analyzing file co-modification
patterns. Includes normalization by commit size to control for commit style bias.

Mechanism: Γ ↑ → fan-in ↑ → Granger S→A ↑
================================================================================
"""

import os
import pickle
import subprocess
from collections import defaultdict
from datetime import datetime
from itertools import combinations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats, sparse
from tqdm import tqdm

try:
    import pygit2
except ImportError:
    print("⚠️ pygit2 required: pip install pygit2")

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False
    print("⚠️ networkx optional but recommended: pip install networkx")

# ==============================================================================
# CONFIGURATION
# ==============================================================================

CODE_EXTENSIONS = {
    '.py', '.js', '.ts', '.jsx', '.tsx', '.c', '.h', '.cpp', '.hpp',
    '.java', '.go', '.rs', '.rb', '.php', '.vue', '.sql', '.swift',
    '.kt', '.scala', '.sh', '.pl', '.r', '.m', '.cs', '.vb', '.lua',
    '.hs', '.ml', '.clj', '.ex', '.erl', '.f90', '.f', '.asm', '.s'
}

CONFIG_EXTENSIONS = {
    '.json', '.yaml', '.yml', '.toml', '.xml', '.gradle', '.cmake',
    '.makefile', '.mk', '.lock', '.mod', '.sum'
}

CONFIG_FILES = {
    'package.json', 'cargo.toml', 'go.mod', 'go.sum', 'pom.xml',
    'build.gradle', 'cmakelists.txt', 'makefile', 'dockerfile',
    '.gitignore', '.eslintrc', 'tsconfig.json', 'pyproject.toml',
    'setup.py', 'setup.cfg', 'requirements.txt', 'gemfile',
    'podfile', 'pubspec.yaml', 'composer.json'
}

EXCLUDE_PATTERNS = [
    '/node_modules/', '/vendor/', '/dist/', '/build/', '/target/',
    '/__pycache__/', '/.git/', '/coverage/', '/tmp/', '/.cache/',
    '/generated/', '/gen/', '/out/', '/bin/', '/obj/',
    '.min.js', '.min.css', '.bundle.', '.map',
    '.png', '.jpg', '.jpeg', '.gif', '.ico', '.svg', '.webp',
    '.mp3', '.mp4', '.wav', '.avi', '.mov',
    '.pdf', '.doc', '.docx', '.xls', '.xlsx',
    '.zip', '.tar', '.gz', '.rar', '.7z',
    '.exe', '.dll', '.so', '.dylib', '.a', '.o',
    '.woff', '.woff2', '.ttf', '.eot',
    'package-lock.json', 'yarn.lock', 'poetry.lock',
]


# ==============================================================================
# CO-MODIFICATION ANALYZER
# ==============================================================================

class CoModificationAnalyzer:
    """
    Analyzes file co-modification patterns to detect structural coupling.
    """

    def __init__(self, name, config, cache_dir=None):
        self.name = name
        self.config = config
        self.repo = None

        # Sparse matrix storage: file_idx -> {other_file_idx: count}
        self.comod_counts = defaultdict(lambda: defaultdict(int))

        # File indexing for memory efficiency
        self.file_to_idx = {}
        self.idx_to_file = {}
        self.next_idx = 0

        # Commit metadata for temporal analysis
        self.commit_dates = []
        self.commit_files = []  # List of sets of file indices
        self.commit_sizes = []  # Number of files per commit (for bias control)

        # Computed metrics
        self.temporal_metrics = []

        # Caching
        self.cache_dir = cache_dir or "./cache_comod/"
        self.cache_file = os.path.join(self.cache_dir, f"{name}_comod_v42.pkl")

    def load_repo(self):
        """Load the git repository using pygit2."""
        try:
            self.repo = pygit2.Repository(self.config['path'])
            return True
        except Exception as e:
            print(f"[{self.name}] ❌ Error loading repo: {e}")
            return False

    def _get_file_idx(self, filepath):
        """Get or create index for a file path."""
        if filepath not in self.file_to_idx:
            self.file_to_idx[filepath] = self.next_idx
            self.idx_to_file[self.next_idx] = filepath
            self.next_idx += 1
        return self.file_to_idx[filepath]

    def _should_include_file(self, filepath):
        """Filter files using same logic as Γ analysis."""
        filepath_lower = filepath.lower()

        for pattern in EXCLUDE_PATTERNS:
            if pattern in filepath_lower:
                return False

        _, ext = os.path.splitext(filepath_lower)
        basename = os.path.basename(filepath_lower)

        if ext in CODE_EXTENSIONS:
            return True
        if ext in CONFIG_EXTENSIONS:
            return True
        if basename in CONFIG_FILES:
            return True

        return False

    def _is_core_file(self, filepath):
        """Check if file is in core paths."""
        core_paths = self.config.get('core_paths', [])
        return any(filepath.startswith(cp) for cp in core_paths)

    def _get_commit_files(self, commit):
        """Extract modified files from a commit using pygit2."""
        files = set()

        try:
            if not commit.parents:
                diff = commit.tree.diff_to_tree(swap=True)
            else:
                diff = self.repo.diff(commit.parents[0], commit)

            for delta in diff.deltas:
                for path in [delta.old_file.path, delta.new_file.path]:
                    if path and self._should_include_file(path):
                        files.add(path)

        except Exception:
            pass

        return files

    def get_commit_style_metrics(self):
        """Compute commit style metrics for bias control."""
        if not self.commit_sizes:
            return {}

        sizes = np.array(self.commit_sizes)
        return {
            'mean_commit_size': np.mean(sizes),
            'median_commit_size': np.median(sizes),
            'std_commit_size': np.std(sizes),
            'pct_large_commits': np.sum(sizes > 10) / len(sizes) * 100
        }

    import subprocess

    def extract_comodification_data(self):
        """
        MODE TURBO: Utilise 'git log' directement au lieu de pygit2.
        Beaucoup plus rapide pour les gros dépôts.
        """
        print(f"[{self.name}] 🚀 Mode TURBO activé (git log stream)...")

        # Commande git pour avoir Hash, Timestamp et les fichiers modifiés
        # --no-merges : On ignore les merges pour éviter le bruit et accélérer
        # Commande optimisée et sécurisée
        cmd = [
            "git", "-C", self.config['path'], "log",
            "--name-only",  # Liste juste les noms de fichiers
            "--pretty=format:===%H|%ct",  # Marqueur de début de commit clair
            "--no-merges",  # Exclut les fusions (bruit statistique)
            "--diff-filter=ACMR",  # Added, Copied, Modified, Renamed (Ignore les suppressions pures 'D')
            "HEAD"
        ]

        try:
            process = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, encoding='utf-8', errors='replace', bufsize=1024 * 1024
            )
        except Exception as e:
            print(f"[{self.name}] ❌ Erreur lancement git log: {e}")
            return False

        current_files = set()
        current_date = None
        commits_processed = 0

        # On utilise tqdm sans total (car on stream) mais avec une barre de vitesse
        pbar = tqdm(desc=f"[{self.name[:10]}]", unit=" com")

        for line in process.stdout:
            line = line.strip()
            if not line:
                continue

            if line.startswith("==="):
                # -- Fin du commit précédent, traitement --
                if len(current_files) >= 2 and len(current_files) <= 100:
                    self.commit_sizes.append(len(current_files))
                    file_indices = {self._get_file_idx(f) for f in current_files}

                    self.commit_dates.append(current_date)
                    self.commit_files.append(file_indices)

                    # Combinatoire (la partie mathématique lourde)
                    # Astuce: Trier les indices accélère l'accès mémoire
                    sorted_indices = sorted(file_indices)
                    for i in range(len(sorted_indices)):
                        idx_a = sorted_indices[i]
                        for j in range(i + 1, len(sorted_indices)):
                            idx_b = sorted_indices[j]
                            self.comod_counts[idx_a][idx_b] += 1
                            self.comod_counts[idx_b][idx_a] += 1

                # -- Nouveau commit --
                try:
                    _, timestamp = line[3:].split('|')
                    current_date = datetime.fromtimestamp(int(timestamp))
                    current_files = set()
                    commits_processed += 1
                    if commits_processed % 1000 == 0:
                        pbar.update(1000)
                except ValueError:
                    pass
            else:
                # C'est un nom de fichier
                if self._should_include_file(line):
                    current_files.add(line)

        pbar.close()
        process.wait()

        if commits_processed == 0:
            print(f"[{self.name}] ⚠️ Aucun commit trouvé ou erreur de lecture.")
            return False

        print(f"[{self.name}] ✅ Fin du Turbo : {commits_processed} commits traités.")
        return True
    def compute_coupling_metrics(self, threshold=3):
        """Compute coupling metrics for all files."""
        metrics = []

        for idx in range(self.next_idx):
            filepath = self.idx_to_file[idx]
            neighbors = self.comod_counts[idx]

            coupled_files = {
                other_idx: count
                for other_idx, count in neighbors.items()
                if count >= threshold
            }

            if not coupled_files:
                continue

            fan_in = len(coupled_files)
            weighted_degree = sum(coupled_files.values())
            max_coupling = max(coupled_files.values())
            is_core = self._is_core_file(filepath)

            metrics.append({
                'file_idx': idx,
                'filepath': filepath,
                'fan_in': fan_in,
                'weighted_degree': weighted_degree,
                'max_coupling': max_coupling,
                'mean_coupling': weighted_degree / fan_in,
                'is_core': is_core
            })

        return pd.DataFrame(metrics)

    def compute_temporal_metrics(self, window_months=3, threshold=3):
        """Compute coupling metrics per time window with bias control."""
        if not self.commit_dates:
            print(f"[{self.name}] ⚠️ No commit data. Run extract_comodification_data first.")
            return []

        commit_df = pd.DataFrame({
            'date': self.commit_dates,
            'files': self.commit_files
        })
        commit_df['date'] = pd.to_datetime(commit_df['date'])
        commit_df = commit_df.sort_values('date')

        start_date = commit_df['date'].min()
        end_date = commit_df['date'].max()

        windows = pd.date_range(
            start=start_date,
            end=end_date,
            freq=f'{window_months}MS'
        )

        temporal_results = []

        for i in range(len(windows) - 1):
            window_start = windows[i]
            window_end = windows[i + 1]

            mask = (commit_df['date'] >= window_start) & (commit_df['date'] < window_end)
            window_commits = commit_df[mask]

            if len(window_commits) < 10:
                continue

            # Build co-modification matrix for this window
            window_comod = defaultdict(lambda: defaultdict(int))
            window_files = set()

            # Track commit sizes in this window for normalization
            window_commit_sizes = []

            for _, row in window_commits.iterrows():
                file_indices = row['files']
                window_files.update(file_indices)
                window_commit_sizes.append(len(file_indices))

                for idx_a, idx_b in combinations(sorted(file_indices), 2):
                    window_comod[idx_a][idx_b] += 1
                    window_comod[idx_b][idx_a] += 1

            # Compute metrics for this window
            fan_ins = []
            weighted_degrees = []
            core_fan_ins = []
            num_edges = 0

            for idx in window_files:
                neighbors = window_comod[idx]
                coupled = {k: v for k, v in neighbors.items() if v >= threshold}

                if coupled:
                    fi = len(coupled)
                    wd = sum(coupled.values())
                    fan_ins.append(fi)
                    weighted_degrees.append(wd)
                    num_edges += fi

                    if self._is_core_file(self.idx_to_file.get(idx, '')):
                        core_fan_ins.append(fi)

            num_edges //= 2

            # Bias control: commit size in this window
            mean_commit_size = np.mean(window_commit_sizes) if window_commit_sizes else 1
            median_commit_size = np.median(window_commit_sizes) if window_commit_sizes else 1

            # Normalized fan-in (controls for commit style)
            mean_fan_in = np.mean(fan_ins) if fan_ins else 0
            normalized_fan_in = mean_fan_in / mean_commit_size if mean_commit_size > 0 else 0

            quarter = (window_start.month - 1) // 3 + 1
            window_label = f"{window_start.year}-Q{quarter}"

            temporal_results.append({
                'project': self.name,
                'window': window_label,
                'window_start': window_start,
                'window_end': window_end,
                'num_commits': len(window_commits),
                'num_nodes': len(window_files),
                'num_edges': num_edges,
                'mean_fan_in': mean_fan_in,
                'median_fan_in': np.median(fan_ins) if fan_ins else 0,
                'max_fan_in': max(fan_ins) if fan_ins else 0,
                'std_fan_in': np.std(fan_ins) if fan_ins else 0,
                'mean_weighted_degree': np.mean(weighted_degrees) if weighted_degrees else 0,
                'core_fan_in': np.mean(core_fan_ins) if core_fan_ins else 0,
                'density': (2 * num_edges) / (len(window_files) * (len(window_files) - 1) + 1e-9),
                # Bias control metrics
                'mean_commit_size': mean_commit_size,
                'median_commit_size': median_commit_size,
                'normalized_fan_in': normalized_fan_in,
            })

        self.temporal_metrics = temporal_results
        return temporal_results

    def compute_betweenness_centrality(self, threshold=3, sample_size=1000):
        """Compute betweenness centrality using networkx."""
        if not HAS_NETWORKX:
            print("⚠️ networkx required for betweenness centrality")
            return {}

        G = nx.Graph()

        for idx_a in self.comod_counts:
            for idx_b, count in self.comod_counts[idx_a].items():
                if count >= threshold and idx_a < idx_b:
                    G.add_edge(idx_a, idx_b, weight=count)

        if G.number_of_nodes() == 0:
            return {}

        print(f"[{self.name}] 🔗 Computing betweenness on {G.number_of_nodes()} nodes, "
              f"{G.number_of_edges()} edges...")

        if G.number_of_nodes() > sample_size:
            bc = nx.betweenness_centrality(G, k=sample_size, weight='weight')
        else:
            bc = nx.betweenness_centrality(G, weight='weight')

        return {
            self.idx_to_file[idx]: score
            for idx, score in bc.items()
        }

    def save_cache(self):
        """Save extracted data to cache."""
        os.makedirs(self.cache_dir, exist_ok=True)

        cache_data = {
            'comod_counts': dict(self.comod_counts),
            'file_to_idx': self.file_to_idx,
            'idx_to_file': self.idx_to_file,
            'next_idx': self.next_idx,
            'commit_dates': self.commit_dates,
            'commit_files': self.commit_files,
            'commit_sizes': self.commit_sizes,
            'temporal_metrics': self.temporal_metrics
        }

        with open(self.cache_file, 'wb') as f:
            pickle.dump(cache_data, f)

        print(f"[{self.name}] 💾 Cache saved")

    def load_cache(self):
        """Load from cache if available."""
        if not os.path.exists(self.cache_file):
            return False

        try:
            with open(self.cache_file, 'rb') as f:
                cache_data = pickle.load(f)

            self.comod_counts = defaultdict(
                lambda: defaultdict(int),
                {k: defaultdict(int, v) for k, v in cache_data['comod_counts'].items()}
            )
            self.file_to_idx = cache_data['file_to_idx']
            self.idx_to_file = cache_data['idx_to_file']
            self.next_idx = cache_data['next_idx']
            self.commit_dates = cache_data['commit_dates']
            self.commit_files = cache_data['commit_files']
            self.commit_sizes = cache_data.get('commit_sizes', [])
            self.temporal_metrics = cache_data.get('temporal_metrics', [])

            print(f"[{self.name}] ✅ Cache loaded ({len(self.file_to_idx)} files)")
            return True

        except Exception as e:
            print(f"[{self.name}] ⚠️ Cache load failed: {e}")
            return False

    def run_analysis(self, use_cache=True):
        """Run complete co-modification analysis."""
        if use_cache and self.load_cache():
            if not self.temporal_metrics:
                self.compute_temporal_metrics()
            global_metrics = self.compute_coupling_metrics()
            return global_metrics, self.temporal_metrics

        if not self.load_repo():
            return None, None

        if not self.extract_comodification_data():
            return None, None

        global_metrics = self.compute_coupling_metrics()
        temporal_metrics = self.compute_temporal_metrics()

        self.save_cache()

        return global_metrics, temporal_metrics


# ==============================================================================
# CORRELATION WITH Γ METRICS
# ==============================================================================

def correlate_with_gamma(
        comod_temporal: list,
        gamma_df: pd.DataFrame,
        project_name: str
) -> dict:
    """Correlate co-modification metrics with Γ (Gamma) evolution."""
    if not comod_temporal or gamma_df is None or gamma_df.empty:
        return None

    gamma_df = gamma_df.copy()
    gamma_df.index = pd.to_datetime(gamma_df.index)

    gamma_df['quarter'] = gamma_df.index.to_period('Q')
    gamma_quarterly = gamma_df.groupby('quarter').agg({
        'monthly_gamma': 'mean',
        'total_weight': 'sum'
    }).reset_index()
    gamma_quarterly['quarter_str'] = gamma_quarterly['quarter'].astype(str)

    comod_df = pd.DataFrame(comod_temporal)
    comod_df['quarter'] = pd.to_datetime(comod_df['window_start']).dt.to_period('Q')
    comod_df['quarter_str'] = comod_df['quarter'].astype(str)

    merged = pd.merge(
        comod_df,
        gamma_quarterly,
        on='quarter_str',
        how='inner',
        suffixes=('_comod', '_gamma')
    )

    if len(merged) < 5:
        return {'error': 'Insufficient overlapping data points'}

    results = {
        'project': project_name,
        'n_points': len(merged)
    }

    # Raw correlation: Γ vs fan-in
    r_gamma_fanin, p_gamma_fanin = stats.spearmanr(
        merged['monthly_gamma'],
        merged['mean_fan_in']
    )
    results['r_gamma_fanin'] = r_gamma_fanin
    results['p_gamma_fanin'] = p_gamma_fanin

    # BIAS CONTROL: Γ vs NORMALIZED fan-in
    if 'normalized_fan_in' in merged.columns:
        r_gamma_norm, p_gamma_norm = stats.spearmanr(
            merged['monthly_gamma'],
            merged['normalized_fan_in']
        )
        results['r_gamma_fanin_normalized'] = r_gamma_norm
        results['p_gamma_fanin_normalized'] = p_gamma_norm

    # Correlation: Γ vs core fan-in
    r_gamma_core, p_gamma_core = stats.spearmanr(
        merged['monthly_gamma'],
        merged['core_fan_in']
    )
    results['r_gamma_core_fanin'] = r_gamma_core
    results['p_gamma_core_fanin'] = p_gamma_core

    # Correlation: Γ vs graph density
    r_gamma_density, p_gamma_density = stats.spearmanr(
        merged['monthly_gamma'],
        merged['density']
    )
    results['r_gamma_density'] = r_gamma_density
    results['p_gamma_density'] = p_gamma_density

    # Control: Activity vs fan-in
    r_act_fanin, p_act_fanin = stats.spearmanr(
        merged['total_weight'],
        merged['mean_fan_in']
    )
    results['r_activity_fanin'] = r_act_fanin
    results['p_activity_fanin'] = p_act_fanin

    # Control: Commit size vs fan-in (to check bias)
    if 'mean_commit_size' in merged.columns:
        r_commit_fanin, p_commit_fanin = stats.spearmanr(
            merged['mean_commit_size'],
            merged['mean_fan_in']
        )
        results['r_commitsize_fanin'] = r_commit_fanin
        results['p_commitsize_fanin'] = p_commit_fanin

    results['merged_data'] = merged

    return results


# ==============================================================================
# VISUALIZATION
# ==============================================================================

def plot_comodification_evolution(
        temporal_metrics: list,
        project_name: str,
        gamma_df: pd.DataFrame = None
):
    """Plot the evolution of co-modification coupling over time."""
    if not temporal_metrics:
        print(f"[{project_name}] No temporal data to plot")
        return

    df = pd.DataFrame(temporal_metrics)
    df['date'] = pd.to_datetime(df['window_start'])
    df = df.sort_values('date')

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # === Plot 1: Fan-in evolution (raw vs normalized) ===
    ax1 = axes[0, 0]
    ax1.plot(df['date'], df['mean_fan_in'], 'b-', linewidth=2, label='Raw Fan-in')
    if 'normalized_fan_in' in df.columns:
        ax1_twin = ax1.twinx()
        ax1_twin.plot(df['date'], df['normalized_fan_in'], 'r--', linewidth=2, label='Normalized')
        ax1_twin.set_ylabel('Normalized Fan-in', color='red')
        ax1_twin.legend(loc='upper right')
    ax1.set_ylabel('Mean Fan-in', color='blue')
    ax1.set_title(f'{project_name}: Fan-in Evolution (Raw vs Normalized)')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)

    # === Plot 2: Commit size evolution (bias indicator) ===
    ax2 = axes[0, 1]
    if 'mean_commit_size' in df.columns:
        ax2.plot(df['date'], df['mean_commit_size'], 'g-', linewidth=2, label='Mean Commit Size')
        ax2.fill_between(df['date'], 0, df['mean_commit_size'], alpha=0.2)
    ax2.set_ylabel('Files per Commit')
    ax2.set_title('Commit Style Evolution (Bias Control)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # === Plot 3: Correlation with Γ (if available) ===
    ax3 = axes[1, 0]

    if gamma_df is not None and not gamma_df.empty:
        gdf = gamma_df.copy()
        gdf.index = pd.to_datetime(gdf.index)
        gdf['quarter'] = gdf.index.to_period('Q')
        gq = gdf.groupby('quarter')['monthly_gamma'].mean().reset_index()
        gq['date'] = gq['quarter'].dt.to_timestamp()

        ax3_twin = ax3.twinx()

        line1 = ax3.plot(df['date'], df['mean_fan_in'], 'b-', linewidth=2, label='Fan-in')
        ax3.set_ylabel('Mean Fan-in', color='blue')

        line2 = ax3_twin.plot(gq['date'], gq['monthly_gamma'], 'green', linewidth=2, label='Γ')
        ax3_twin.set_ylabel('Metabolic Efficiency (Γ)', color='green')
        ax3_twin.set_ylim(0, 1.1)

        ax3.set_title('Fan-in vs Γ Co-evolution')
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax3.legend(lines, labels, loc='upper left')
    else:
        ax3.text(0.5, 0.5, 'Γ data not available', ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('Fan-in vs Γ Co-evolution')

    ax3.grid(True, alpha=0.3)

    # === Plot 4: Scatter Fan-in vs Γ ===
    ax4 = axes[1, 1]

    if gamma_df is not None and not gamma_df.empty:
        corr_results = correlate_with_gamma(temporal_metrics, gamma_df, project_name)

        if corr_results and 'merged_data' in corr_results:
            merged = corr_results['merged_data']

            ax4.scatter(
                merged['monthly_gamma'],
                merged['mean_fan_in'],
                c=range(len(merged)),
                cmap='viridis',
                s=100,
                alpha=0.7
            )

            z = np.polyfit(merged['monthly_gamma'], merged['mean_fan_in'], 1)
            p = np.poly1d(z)
            x_line = np.linspace(merged['monthly_gamma'].min(), merged['monthly_gamma'].max(), 100)
            ax4.plot(x_line, p(x_line), 'r--', linewidth=2, alpha=0.7)

            r = corr_results['r_gamma_fanin']
            p_val = corr_results['p_gamma_fanin']
            r_norm = corr_results.get('r_gamma_fanin_normalized', np.nan)

            title = f'Γ vs Fan-in\nRaw: r={r:.3f}, p={p_val:.4f}'
            if not np.isnan(r_norm):
                title += f'\nNormalized: r={r_norm:.3f}'
            ax4.set_title(title)
            ax4.set_xlabel('Metabolic Efficiency (Γ)')
            ax4.set_ylabel('Mean Fan-in')

            cbar = plt.colorbar(ax4.collections[0], ax=ax4)
            cbar.set_label('Time (early → late)')
    else:
        ax4.text(0.5, 0.5, 'Correlation not available', ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Γ vs Fan-in Correlation')

    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    filename = f"comod_v42_{project_name.lower()}_evolution.png"
    plt.savefig(filename, dpi=150)
    print(f"✅ Plot saved: {filename}")
    plt.close(fig)


def plot_global_correlation_summary(all_correlations: list):
    """Plot summary of Γ-FanIn correlations across all projects."""
    if not all_correlations:
        print("No correlation data to summarize")
        return

    valid = [c for c in all_correlations if c and 'r_gamma_fanin' in c]

    if not valid:
        print("No valid correlation results")
        return

    df = pd.DataFrame(valid)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # === Plot 1: Distribution of RAW correlations ===
    ax1 = axes[0, 0]
    ax1.hist(df['r_gamma_fanin'], bins=15, color='#3498db', edgecolor='black', alpha=0.7)
    ax1.axvline(x=0, color='red', linestyle='--', linewidth=2)
    ax1.axvline(x=df['r_gamma_fanin'].mean(), color='green', linestyle='-', linewidth=2,
                label=f"Mean: {df['r_gamma_fanin'].mean():.3f}")
    ax1.set_xlabel('Spearman r (Γ vs Fan-in)')
    ax1.set_ylabel('Number of Projects')
    ax1.set_title('RAW Correlations')
    ax1.legend()

    # === Plot 2: Distribution of NORMALIZED correlations ===
    ax2 = axes[0, 1]
    if 'r_gamma_fanin_normalized' in df.columns:
        norm_r = df['r_gamma_fanin_normalized'].dropna()
        ax2.hist(norm_r, bins=15, color='#e74c3c', edgecolor='black', alpha=0.7)
        ax2.axvline(x=0, color='red', linestyle='--', linewidth=2)
        ax2.axvline(x=norm_r.mean(), color='green', linestyle='-', linewidth=2,
                    label=f"Mean: {norm_r.mean():.3f}")
        ax2.set_xlabel('Spearman r (Γ vs Normalized Fan-in)')
        ax2.set_title('NORMALIZED Correlations (Bias Controlled)')
        ax2.legend()
    else:
        ax2.text(0.5, 0.5, 'No normalized data', ha='center', va='center', transform=ax2.transAxes)
    ax2.set_ylabel('Number of Projects')

    # === Plot 3: Raw vs Normalized scatter ===
    ax3 = axes[1, 0]
    if 'r_gamma_fanin_normalized' in df.columns:
        mask = df['r_gamma_fanin_normalized'].notna()
        ax3.scatter(df.loc[mask, 'r_gamma_fanin'],
                   df.loc[mask, 'r_gamma_fanin_normalized'],
                   s=100, alpha=0.7, c='#9b59b6')
        ax3.plot([-1, 1], [-1, 1], 'k--', alpha=0.5, label='y=x')
        ax3.set_xlabel('Raw r')
        ax3.set_ylabel('Normalized r')
        ax3.set_title('Raw vs Normalized Correlation\n(Divergence = Commit Style Bias)')
        ax3.legend()
        ax3.set_xlim(-1, 1)
        ax3.set_ylim(-1, 1)
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, 'No normalized data', ha='center', va='center', transform=ax3.transAxes)

    # === Plot 4: Project-level results ===
    ax4 = axes[1, 1]
    df_sorted = df.sort_values('r_gamma_fanin', ascending=True)

    colors = ['#27ae60' if p < 0.05 else '#95a5a6' for p in df_sorted['p_gamma_fanin']]
    ax4.barh(df_sorted['project'], df_sorted['r_gamma_fanin'], color=colors, edgecolor='black')
    ax4.axvline(x=0, color='red', linestyle='-', linewidth=1)
    ax4.set_xlabel('Spearman r (Γ vs Fan-in)')
    ax4.set_title('Per-Project Correlations\n(Green = p < 0.05)')

    plt.tight_layout()
    filename = "comod_v42_global_correlation_summary.png"
    plt.savefig(filename, dpi=150)
    print(f"✅ Global summary saved: {filename}")
    plt.close(fig)


# ==============================================================================
# MAIN RUNNER
# ==============================================================================

def _process_single_repo(args):
    """Worker function for multiprocessing."""
    name, config, cache_dir = args

    analyzer = CoModificationAnalyzer(name, config, cache_dir)

    try:
        global_metrics, temporal_metrics = analyzer.run_analysis(use_cache=True)

        if global_metrics is not None:
            return {
                'name': name,
                'global_metrics': global_metrics,
                'temporal_metrics': temporal_metrics,
                'commit_style': analyzer.get_commit_style_metrics(),
                'success': True
            }
    except Exception as e:
        print(f"[{name}] ❌ Error: {e}")

    return {'name': name, 'success': False}



def run_comodification_analysis(
        repos_config: dict,
        gamma_dataframes: dict = None,
        cache_dir: str = None,
        max_workers: int = 6
) -> dict:
    """Run co-modification analysis on all configured repositories."""
    import concurrent.futures

    print("\n" + "=" * 80)
    print(f"CO-MODIFICATION COUPLING ANALYSIS (V42) — {max_workers} workers")
    print("=" * 80)
    print("With BIAS CONTROL (normalized by commit size)")
    print("=" * 80)

    all_global_metrics = {}
    all_temporal_metrics = {}
    all_correlations = []
    all_commit_styles = {}

    work_items = [(name, config, cache_dir) for name, config in repos_config.items()]

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_process_single_repo, item): item[0] for item in work_items}

        for future in concurrent.futures.as_completed(futures):
            name = futures[future]
            try:
                result = future.result()

                if result['success']:
                    all_global_metrics[name] = result['global_metrics']
                    all_temporal_metrics[name] = result['temporal_metrics']
                    all_commit_styles[name] = result.get('commit_style', {})

                    gamma_df = gamma_dataframes.get(name) if gamma_dataframes else None
                    if gamma_df is not None:
                        corr = correlate_with_gamma(result['temporal_metrics'], gamma_df, name)
                        if corr and 'r_gamma_fanin' in corr:
                            all_correlations.append(corr)
                            r_raw = corr['r_gamma_fanin']
                            r_norm = corr.get('r_gamma_fanin_normalized', np.nan)
                            norm_str = f", norm={r_norm:.3f}" if not np.isnan(r_norm) else ""
                            print(f"[{name}] Γ-FanIn: r={r_raw:.3f}{norm_str}, p={corr['p_gamma_fanin']:.4f}")

                    print(f"[{name}] ✅ Done")

            except Exception as e:
                print(f"[{name}] ❌ Error: {e}")

    # Generate plots
    print("\n" + "─" * 60)
    print("Generating plots...")
    for name in all_temporal_metrics:
        gamma_df = gamma_dataframes.get(name) if gamma_dataframes else None
        plot_comodification_evolution(all_temporal_metrics[name], name, gamma_df)

    # Summary
    if all_correlations:
        plot_global_correlation_summary(all_correlations)

        print("\n" + "=" * 80)
        print("GLOBAL CORRELATION SUMMARY")
        print("=" * 80)

        valid_corrs = [c for c in all_correlations if 'r_gamma_fanin' in c]
        if valid_corrs:
            r_values = [c['r_gamma_fanin'] for c in valid_corrs]
            p_values = [c['p_gamma_fanin'] for c in valid_corrs]

            print(f"\n--- RAW CORRELATIONS ---")
            print(f"Projects analyzed: {len(valid_corrs)}")
            print(f"Mean r: {np.mean(r_values):.3f}")
            print(f"Median r: {np.median(r_values):.3f}")
            print(f"Positive: {sum(1 for r in r_values if r > 0)}/{len(r_values)}")
            print(f"Negative: {sum(1 for r in r_values if r < 0)}/{len(r_values)}")
            print(f"Significant (p<0.05): {sum(1 for p in p_values if p < 0.05)}/{len(p_values)}")

            t_stat, t_pval = stats.ttest_1samp(r_values, 0)
            print(f"\nOne-sample t-test (H0: mean r = 0):")
            print(f"   t = {t_stat:.3f}, p = {t_pval:.4f}")

            mean_r = np.mean(r_values)
            if t_pval < 0.05:
                if mean_r > 0:
                    print("   ✅ POSITIVE: Maturity → MORE coupling")
                else:
                    print("   ✅ NEGATIVE: Maturity → LESS coupling (modularization)")
            else:
                print("   ⚠️ NOT SIGNIFICANT")

            # Normalized correlation summary
            norm_r = [c.get('r_gamma_fanin_normalized', np.nan) for c in valid_corrs]
            norm_r = [r for r in norm_r if not np.isnan(r)]

            if norm_r:
                print(f"\n--- BIAS CONTROL (Normalized by commit size) ---")
                print(f"Mean normalized r: {np.mean(norm_r):.3f}")
                print(f"Median normalized r: {np.median(norm_r):.3f}")
                print(f"Negative: {sum(1 for r in norm_r if r < 0)}/{len(norm_r)}")
                print(f"Positive: {sum(1 for r in norm_r if r > 0)}/{len(norm_r)}")

                t_stat_n, t_pval_n = stats.ttest_1samp(norm_r, 0)
                print(f"\nNormalized t-test:")
                print(f"   t = {t_stat_n:.3f}, p = {t_pval_n:.4f}")

                # Compare raw vs normalized
                divergence = np.mean(r_values) - np.mean(norm_r)
                print(f"\n--- BIAS CHECK ---")
                print(f"Raw mean - Normalized mean = {divergence:.3f}")
                if abs(divergence) > 0.1:
                    print(f"   ⚠️ BIAS DETECTED: Commit style affects results")
                else:
                    print(f"   ✅ LOW BIAS: Results robust to commit style")

    return {
        'global_metrics': all_global_metrics,
        'temporal_metrics': all_temporal_metrics,
        'correlations': all_correlations,
        'commit_styles': all_commit_styles
    }


# ==============================================================================
# STANDALONE EXECUTION
# ==============================================================================

if __name__ == "__main__":
    print("Co-Modification Analysis Module V42 (with Bias Control)")
    print("Run from main omega script or import it.")