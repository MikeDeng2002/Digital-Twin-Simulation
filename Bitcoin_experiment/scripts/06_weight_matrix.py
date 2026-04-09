"""
06_weight_matrix.py — Persona × Celebrity weight matrix analysis.

Loads per-celebrity OLS weights from 04_analyze.py output and produces:
  1. Heatmap of absolute contribution (w_i × price_i) per persona × celebrity
  2. Persona clustering by influence pattern (bull-sensitive vs bear-sensitive)
  3. Celebrity ranking by mean absolute contribution
  4. Persona ranking by total susceptibility

Usage (from Digital-Twin-Simulation/):
    poetry run python Bitcoin_experiment/scripts/06_weight_matrix.py
"""

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist

ANALYSIS_DIR = Path("Bitcoin_experiment/analysis")
FIGURES_DIR = Path("Bitcoin_experiment/figures")

BULLS = ["michael_saylor", "cathie_wood", "elon_musk", "jack_dorsey", "larry_fink", "donald_trump"]
BEARS = ["warren_buffett", "peter_schiff", "jamie_dimon", "nouriel_roubini"]
ALL_SLUGS = BULLS + BEARS

CELEBRITY_PRICES = {
    "michael_saylor": 1_000_000,
    "cathie_wood":    500_000,
    "jack_dorsey":    500_000,
    "donald_trump":   250_000,
    "elon_musk":      150_000,
    "larry_fink":     150_000,
    "warren_buffett":  15_000,
    "jamie_dimon":      5_000,
    "peter_schiff":     3_000,
    "nouriel_roubini":  2_000,
}

SHORT_NAMES = {
    "michael_saylor": "Saylor",
    "cathie_wood":    "Wood",
    "elon_musk":      "Musk",
    "jack_dorsey":    "Dorsey",
    "larry_fink":     "Fink",
    "donald_trump":   "Trump",
    "warren_buffett": "Buffett",
    "peter_schiff":   "Schiff",
    "jamie_dimon":    "Dimon",
    "nouriel_roubini":"Roubini",
}


def load_weight_matrix() -> pd.DataFrame:
    df = pd.read_csv(ANALYSIS_DIR / "w_celeb_per_celebrity_type_a.csv")
    df = df.set_index("pid")
    return df


def compute_contribution_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Convert raw weights to absolute dollar contributions (w_i × price_i).
    Also computes baseline_retention = baseline / mean_prediction."""
    contrib = pd.DataFrame(index=df.index)
    for slug in ALL_SLUGS:
        col = f"w_{slug}"
        if col in df.columns:
            contrib[slug] = df[col] * CELEBRITY_PRICES[slug]
    # Baseline retention: fraction of final prediction explained by prior
    # baseline_retention = baseline / mean_prediction
    # = baseline / (baseline + mean_shift)
    # Close to 1.0 → persona anchors strongly to own prior
    # Close to 0.0 → persona is heavily pulled by celebrities
    contrib["baseline"] = df["baseline"]
    contrib["mean_prediction"] = df["mean_prediction"]
    contrib["baseline_retention"] = df["baseline"] / df["mean_prediction"]
    return contrib


def plot_weight_heatmap(contrib: pd.DataFrame, clip_pct: float = 99.0):
    """Heatmap of dollar contributions + baseline retention column."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    bull_sum = contrib[BULLS].sum(axis=1)
    bear_sum = contrib[BEARS].sum(axis=1)
    bull_bias = bull_sum - bear_sum.abs()
    sorted_pids = bull_bias.sort_values(ascending=False).index

    celeb_data = contrib.loc[sorted_pids, ALL_SLUGS].values.astype(float)
    retention_data = contrib.loc[sorted_pids, "baseline_retention"].values.astype(float).reshape(-1, 1)

    # Clip extreme outliers for celebrity color scale
    vmax = np.nanpercentile(np.abs(celeb_data), clip_pct)
    vmin = -vmax

    # Two-panel figure: celebrity heatmap + retention bar
    fig, (ax_celeb, ax_ret) = plt.subplots(
        1, 2, figsize=(15, max(8, len(sorted_pids) * 0.18)),
        gridspec_kw={"width_ratios": [len(ALL_SLUGS), 1.2]}
    )

    # --- Celebrity influence panel ---
    im = ax_celeb.imshow(celeb_data, aspect="auto", cmap="RdBu_r", vmin=vmin, vmax=vmax,
                         interpolation="nearest")
    ax_celeb.set_xticks(range(len(ALL_SLUGS)))
    ax_celeb.set_xticklabels([SHORT_NAMES[s] for s in ALL_SLUGS], rotation=40, ha="right", fontsize=9)
    ytick_pos = list(range(0, len(sorted_pids), 5))
    ax_celeb.set_yticks(ytick_pos)
    ax_celeb.set_yticklabels([f"pid_{sorted_pids[i]}" for i in ytick_pos], fontsize=7)
    ax_celeb.axvline(x=len(BULLS) - 0.5, color="black", linewidth=1.5, linestyle="--")
    ax_celeb.text(len(BULLS) / 2 - 0.5, -1.8, "BULLS", ha="center", fontsize=9, color="navy")
    ax_celeb.text(len(BULLS) + len(BEARS) / 2 - 0.5, -1.8, "BEARS", ha="center", fontsize=9, color="darkred")
    cb1 = plt.colorbar(im, ax=ax_celeb, fraction=0.03, pad=0.02)
    cb1.set_label("Dollar contribution per appearance ($)", fontsize=8)
    ax_celeb.set_title("Celebrity Influence\n(dollar contribution = w × price)", fontsize=10, pad=18)
    ax_celeb.set_xlabel("Celebrity", fontsize=9)
    ax_celeb.set_ylabel("Persona (sorted: most bull-sensitive → most bear-sensitive)", fontsize=8)

    # --- Baseline retention panel ---
    im2 = ax_ret.imshow(retention_data, aspect="auto", cmap="Greens", vmin=0, vmax=1,
                        interpolation="nearest")
    ax_ret.set_xticks([0])
    ax_ret.set_xticklabels(["Baseline\nRetention"], fontsize=9)
    ax_ret.set_yticks(ytick_pos)
    ax_ret.set_yticklabels([f"pid_{sorted_pids[i]}" for i in ytick_pos], fontsize=7)
    cb2 = plt.colorbar(im2, ax=ax_ret, fraction=0.15, pad=0.04)
    cb2.set_label("baseline / mean_prediction\n(1.0 = fully anchored to prior)", fontsize=7)
    ax_ret.set_title("Self-Anchor\n(baseline retention)", fontsize=10, pad=18)

    plt.suptitle("Persona × Celebrity Influence Matrix  +  Baseline Retention", fontsize=12, y=1.01)
    plt.tight_layout()
    out = FIGURES_DIR / "weight_matrix_heatmap.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved heatmap → {out}")


def plot_persona_clusters(contrib: pd.DataFrame):
    """Cluster personas by their influence pattern and show dendrogram + profiles."""
    data = contrib[ALL_SLUGS].fillna(0).values
    pids = contrib.index.tolist()

    # Clip per-row extreme values so clustering isn't dominated by outliers
    for i in range(data.shape[0]):
        row = data[i]
        p95 = np.percentile(np.abs(row[row != 0]), 95) if np.any(row != 0) else 1
        data[i] = np.clip(row, -p95 * 3, p95 * 3)

    Z = linkage(data, method="ward", metric="euclidean")
    order = dendrogram(Z, no_plot=True)["leaves"]

    fig, axes = plt.subplots(1, 2, figsize=(16, max(6, len(pids) * 0.15)),
                              gridspec_kw={"width_ratios": [1, 3]})

    # Dendrogram
    dendrogram(Z, orientation="left", ax=axes[0], labels=[f"pid_{p}" for p in pids],
               color_threshold=0.7 * max(Z[:, 2]), leaf_font_size=6)
    axes[0].set_title("Hierarchical Clustering", fontsize=9)
    axes[0].set_xlabel("Ward distance")

    # Heatmap reordered by clustering
    sorted_data = data[order]
    vmax = np.nanpercentile(np.abs(sorted_data), 97)
    im = axes[1].imshow(sorted_data, aspect="auto", cmap="RdBu_r",
                        vmin=-vmax, vmax=vmax, interpolation="nearest")
    axes[1].set_xticks(range(len(ALL_SLUGS)))
    axes[1].set_xticklabels([SHORT_NAMES[s] for s in ALL_SLUGS], rotation=40, ha="right", fontsize=9)
    axes[1].set_yticks([])
    axes[1].axvline(x=len(BULLS) - 0.5, color="black", linewidth=1.5, linestyle="--")
    plt.colorbar(im, ax=axes[1], fraction=0.02, label="Dollar contribution ($)")
    axes[1].set_title("Influence Pattern (clustered order)", fontsize=9)

    plt.suptitle("Persona Clusters by Celebrity Influence Pattern", fontsize=12, y=1.01)
    plt.tight_layout()
    out = FIGURES_DIR / "persona_clusters.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved clusters → {out}")


def plot_celebrity_ranking(contrib: pd.DataFrame):
    """Bar chart: mean absolute dollar contribution per celebrity."""
    means = contrib[ALL_SLUGS].mean()
    stds  = contrib[ALL_SLUGS].std()
    names = [SHORT_NAMES[s] for s in ALL_SLUGS]
    colors = ["steelblue" if s in BULLS else "firebrick" for s in ALL_SLUGS]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(names, means.values, color=colors, alpha=0.75, edgecolor="black", linewidth=0.5)
    ax.errorbar(names, means.values, yerr=stds.values, fmt="none", color="black",
                capsize=4, linewidth=1)
    ax.axhline(0, color="black", linewidth=0.8)

    # Label bars
    for bar, val in zip(bars, means.values):
        offset = 2000 if val >= 0 else -4000
        ax.text(bar.get_x() + bar.get_width()/2, val + offset,
                f"${val:,.0f}", ha="center", va="bottom" if val >= 0 else "top", fontsize=8)

    ax.set_title("Mean Dollar Contribution per Celebrity Appearance\n(mean ± 1 std across personas)",
                 fontsize=11)
    ax.set_ylabel("Mean dollar shift in prediction ($)")
    ax.set_xlabel("Celebrity")

    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor="steelblue", label="Bull"),
                       Patch(facecolor="firebrick", label="Bear")]
    ax.legend(handles=legend_elements, loc="upper right")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    out = FIGURES_DIR / "celebrity_contribution_ranking.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved ranking → {out}")


def print_summary(contrib: pd.DataFrame, df: pd.DataFrame):
    print("\n" + "=" * 70)
    print("CELEBRITY RANKING — Mean dollar contribution per appearance")
    print("=" * 70)
    means = contrib[ALL_SLUGS].mean().sort_values(ascending=False)
    stds  = contrib[ALL_SLUGS].std()
    print(f"{'Celebrity':<15} {'Cat':<5} {'Mean $':>10} {'Std $':>10} {'Price':>12}")
    print("-" * 70)
    for slug in means.index:
        cat = "BULL" if slug in BULLS else "BEAR"
        print(f"{SHORT_NAMES[slug]:<15} {cat:<5} "
              f"{means[slug]:>10,.0f} {stds[slug]:>10,.0f} "
              f"{CELEBRITY_PRICES[slug]:>12,}")

    print("\n" + "=" * 70)
    print("PERSONA RANKING — Total bull contribution vs bear contribution")
    print("=" * 70)
    bull_contrib = contrib[BULLS].sum(axis=1)
    bear_contrib = contrib[BEARS].sum(axis=1)
    summary = pd.DataFrame({
        "bull_total": bull_contrib,
        "bear_total": bear_contrib,
        "net": bull_contrib + bear_contrib,
    }).sort_values("net", ascending=False)

    print(f"{'PID':<8} {'Bull contrib':>14} {'Bear contrib':>14} {'Net':>14}")
    print("-" * 55)
    for pid, row in summary.iterrows():
        print(f"pid_{pid:<4} {row['bull_total']:>14,.0f} {row['bear_total']:>14,.0f} {row['net']:>14,.0f}")

    # Save matrix CSV
    out = ANALYSIS_DIR / "contribution_matrix.csv"
    contrib.round(0).to_csv(out)
    print(f"\nContribution matrix saved to {out}")


def main():
    df = load_weight_matrix()
    contrib = compute_contribution_matrix(df)

    print(f"Loaded weight matrix: {len(df)} personas × {len(ALL_SLUGS)} celebrities")

    print_summary(contrib, df)
    plot_celebrity_ranking(contrib)
    plot_weight_heatmap(contrib)
    plot_persona_clusters(contrib)

    print("\nDone. Figures saved to Bitcoin_experiment/figures/")


if __name__ == "__main__":
    main()
