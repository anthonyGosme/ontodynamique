import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
matplotlib.use('MacOSX')  # ou 'MacOSX' vu que vous êtes sur Mac
import matplotlib.pyplot as plt
# ... le reste des imports
def si_dashboard(csv_path, metric="confinement", title=None):
    df = pd.read_csv(csv_path)
    # garde seulement ce qui a la métrique
    df = df.dropna(subset=["project", "type", metric])

    # Observed
    obs = df[df["type"] == "Observed"].set_index("project")[metric]

    # Placebos (moyenne hiérarchique : mean par type puis mean par projet)
    plc = (
        df[df["type"] != "Observed"]
        .groupby(["project", "type"])[metric].mean()
        .groupby("project").mean()
    )

    common = obs.index.intersection(plc.index)
    obs_c = obs.loc[common]
    plc_c = plc.loc[common]
    delta = obs_c - plc_c

    print(f"\n=== {csv_path} ===")
    print(f"metric={metric} | N(common)={len(common)}")
    print(f"Observed median={obs_c.median():.4f} | PlaceboMean median={plc_c.median():.4f}")
    print(f"Delta median={delta.median():.4f} | mean={delta.mean():.4f} | pct(>0)={(delta>0).mean():.2%}")

    # ---- PLOTS ----
    plt.figure(figsize=(6, 6))
    plt.scatter(plc_c, obs_c, alpha=0.7)
    m = np.nanmax([obs_c.max(), plc_c.max()])
    plt.plot([0, m], [0, m], "--")
    plt.xlabel("PlaceboMean (per project)")
    plt.ylabel("Observed (per project)")
    plt.title(title or f"Observed vs PlaceboMean | {metric}")
    plt.tight_layout()
   # plt.show()

    plt.figure(figsize=(7, 4))
    plt.hist(delta.values, bins=20)
    plt.axvline(0, linestyle="--")
    plt.xlabel("Observed - PlaceboMean")
    plt.ylabel("Count")
    plt.title(f"Delta distribution | {metric}")
    plt.tight_layout()
   # plt.show()

    # boxplot par type (optionnel : si pas trop de types)
    # on limite à quelques types max sinon illisible
    top_types = (
        df["type"].value_counts()
        .index[:12]  # ajuste si besoin
        .tolist()
    )
    df2 = df[df["type"].isin(top_types)].copy()

    plt.figure(figsize=(12, 4))
    sns.boxplot(data=df2, x="type", y=metric)
    plt.xticks(rotation=45, ha="right")
    plt.title(f"Metric by type (top {len(top_types)} types) | {metric}")
    plt.tight_layout()
   # plt.show()

    return {"obs": obs_c, "plc": plc_c, "delta": delta}
if __name__ == "__main__":
    si_dashboard("figures/SI/SI_placebo_tests_gamma_min_lag2.csv", metric="confinement")
    si_dashboard("figures/SI/SI_placebo_tests_gamma_min_lag2.csv", metric="mean_strength")
    si_dashboard("figures/SI/SI_placebo_tests_gamma_min_lag2.csv", metric="variance")

    # B) Placebos "gamma perturbed" (index-safe)
    si_dashboard("figures/SI/gamma_placebo_index_safe_min_lag2_thr0.7.csv", metric="confinement")
    si_dashboard("figures/SI/gamma_placebo_index_safe_min_lag2_thr0.7.csv", metric="mean_strength")
    si_dashboard("figures/SI/gamma_placebo_index_safe_min_lag2_thr0.7.csv", metric="variance")

#si_dashboard("figures/SI/cross_swap_min_lag2_thr0.7.csv", metric="confinement")
#si_dashboard("figures/SI/cross_swap_min_lag2_thr0.7.csv", metric="mean_strength")
#si_dashboard("figures/SI/cross_swap_min_lag2_thr0.7.csv", metric="variance")
