"""Compare OLS influence matrices across four network topologies."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = "text_simulation/text_simulation_output_interaction"

TOPOS = {
    "Ring":            f"{OUT_DIR}/bitcoin_sparse_ring/influence_matrix_A_ols.csv",
    "Random":          f"{OUT_DIR}/bitcoin_sparse_random/influence_matrix_A_ols.csv",
    "Clusters":        f"{OUT_DIR}/bitcoin_sparse_clusters/influence_matrix_A_ols.csv",
    "Fully Connected": f"{OUT_DIR}/bitcoin_fully_connected/influence_matrix_A_ols.csv",
}

# ============================================================
# 1. Load matrices
# ============================================================
matrices = {}
for label, path in TOPOS.items():
    df = pd.read_csv(path, index_col=0)
    matrices[label] = df.values
    print(f"{label:20s}: shape={df.values.shape}, "
          f"mean|A|={np.mean(np.abs(df.values)):.4f}, "
          f"max|A|={np.max(np.abs(df.values)):.4f}")

# ============================================================
# 2. Side-by-side heatmaps
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(18, 16))
axes = axes.flatten()

for ax, (label, A) in zip(axes, matrices.items()):
    n = A.shape[0]
    vmax = np.max(np.abs(A))
    if vmax == 0:
        vmax = 1
    im = ax.imshow(A, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="equal")
    agents = pd.read_csv(list(TOPOS.values())[0], index_col=0).index.tolist()
    ticks = [a.replace("pid_", "") for a in agents]
    ax.set_xticks(range(n))
    ax.set_xticklabels(ticks, rotation=90, fontsize=6)
    ax.set_yticks(range(n))
    ax.set_yticklabels(ticks, fontsize=6)
    ax.set_title(label, fontsize=13, fontweight="bold")
    ax.set_xlabel("Influencer", fontsize=10)
    ax.set_ylabel("Influenced", fontsize=10)
    plt.colorbar(im, ax=ax, shrink=0.8)

plt.suptitle("OLS Influence Matrices — Topology Comparison", fontsize=15, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/comparison_heatmaps_ols.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"\nSaved: {OUT_DIR}/comparison_heatmaps_ols.png")

# ============================================================
# 3. Summary statistics table
# ============================================================
print(f"\n{'Topology':<20} {'Mean|A|':>10} {'Std|A|':>10} {'Max|A|':>10} "
      f"{'MeanL1/row':>12} {'Sparsity':>10}")
print("-" * 76)

stats = {}
for label, A in matrices.items():
    n = A.shape[0]
    nonzero_mask = A != 0
    mean_abs     = np.mean(np.abs(A[nonzero_mask])) if nonzero_mask.any() else 0
    std_abs      = np.std(np.abs(A[nonzero_mask]))  if nonzero_mask.any() else 0
    max_abs      = np.max(np.abs(A))
    mean_l1_row  = np.mean([np.sum(np.abs(A[i, :])) for i in range(n)])
    sparsity     = 1.0 - nonzero_mask.mean()
    stats[label] = dict(mean_abs=mean_abs, std_abs=std_abs, max_abs=max_abs,
                        mean_l1_row=mean_l1_row, sparsity=sparsity)
    print(f"{label:<20} {mean_abs:10.4f} {std_abs:10.4f} {max_abs:10.4f} "
          f"{mean_l1_row:12.4f} {sparsity:10.4f}")

# ============================================================
# 4. L1-norm-per-row distributions (box plot)
# ============================================================
fig, ax = plt.subplots(figsize=(10, 6))
l1_data = []
labels  = []
for label, A in matrices.items():
    l1_rows = [np.sum(np.abs(A[i, :])) for i in range(A.shape[0])]
    l1_data.append(l1_rows)
    labels.append(label)

bp = ax.boxplot(l1_data, labels=labels, patch_artist=True,
                medianprops=dict(color="black", linewidth=2))
colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
for patch, color in zip(bp["boxes"], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax.set_ylabel("L1 Norm per Row  (sum_j |A[i,j]|)", fontsize=12)
ax.set_title("Distribution of Row-wise L1 Norms — OLS Influence Matrix", fontsize=13, fontweight="bold")
ax.axhline(1.0, color="gray", linestyle="--", linewidth=1, label="L1=1 (QP constraint)")
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/comparison_l1_norms_ols.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {OUT_DIR}/comparison_l1_norms_ols.png")

# ============================================================
# 5. Weight magnitude distributions (histogram overlay)
# ============================================================
fig, ax = plt.subplots(figsize=(10, 6))
colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
for (label, A), color in zip(matrices.items(), colors):
    nonzero_vals = np.abs(A[A != 0])
    ax.hist(nonzero_vals, bins=40, alpha=0.5, label=label, color=color, density=True)

ax.set_xlabel("|A[i,j]| (nonzero entries)", fontsize=12)
ax.set_ylabel("Density", fontsize=12)
ax.set_title("Distribution of Nonzero Weight Magnitudes — OLS", fontsize=13, fontweight="bold")
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/comparison_weight_dist_ols.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {OUT_DIR}/comparison_weight_dist_ols.png")

# ============================================================
# 6. OLS estimation variance heatmap — one per topology
# ============================================================
GRAPH_PATHS = {
    "Ring":            "text_simulation/interaction_graph_sparse_ring.json",
    "Random":          "text_simulation/interaction_graph_sparse_random.json",
    "Clusters":        "text_simulation/interaction_graph_sparse_clusters.json",
    "Fully Connected": "text_simulation/interaction_graph_fully_connected.json",
}
SCORE_PATHS = {
    "Ring":            f"{OUT_DIR}/bitcoin_sparse_ring/sentiment_scores.csv",
    "Random":          f"{OUT_DIR}/bitcoin_sparse_random/sentiment_scores.csv",
    "Clusters":        f"{OUT_DIR}/bitcoin_sparse_clusters/sentiment_scores.csv",
    "Fully Connected": f"{OUT_DIR}/bitcoin_fully_connected/sentiment_scores.csv",
}

def load_X_and_mask(scores_path, graph_path):
    df = pd.read_csv(scores_path)
    df = df[df["question"] == "Q1"]
    df = df[df["round"] >= 1]
    pivot = df.groupby(["persona_id", "round"])["score"].mean().reset_index()
    pivot_table = pivot.pivot(index="persona_id", columns="round", values="score")
    agents = sorted(pivot_table.index.tolist())
    pivot_table = pivot_table.loc[agents]
    X = pivot_table.values.astype(float)
    for i in range(X.shape[0]):
        row = pd.Series(X[i])
        X[i] = row.ffill().bfill().values
    X = np.nan_to_num(X, nan=0.0)

    with open(graph_path) as f:
        graph = json.load(f)
    agent_idx = {a: k for k, a in enumerate(agents)}
    n = len(agents)
    mask = np.zeros((n, n), dtype=int)
    for agent, neighbors in graph.items():
        if agent in agent_idx:
            for nb in neighbors:
                if nb in agent_idx:
                    mask[agent_idx[agent], agent_idx[nb]] = 1
    for i in range(n):
        mask[i, i] = 1
    return X, mask, agents

var_matrices = {}
for label in TOPOS:
    X, mask, agents = load_X_and_mask(SCORE_PATHS[label], GRAPH_PATHS[label])
    n_agents, n_rounds = X.shape
    X_input  = X[:, :-1]
    X_target = X[:, 1:]
    T = n_rounds - 1
    V = np.zeros((n_agents, n_agents))

    for i in range(n_agents):
        free_idx = np.where(mask[i, :] == 1)[0]
        n_free   = len(free_idx)
        X_free   = X_input[free_idx, :]       # (n_free, T)
        y_target = X_target[i, :]             # (T,)

        a_opt, _, _, _ = np.linalg.lstsq(X_free.T, y_target, rcond=None)
        residuals = X_free.T @ a_opt - y_target
        dof = max(1, T - n_free)
        sigma2 = np.sum(residuals ** 2) / dof

        XXT = X_free @ X_free.T               # (n_free, n_free)
        try:
            XXT_inv = np.linalg.pinv(XXT)
        except np.linalg.LinAlgError:
            XXT_inv = np.zeros_like(XXT)

        w_var = sigma2 * np.diag(XXT_inv)     # variance of each â_j
        for k, j_idx in enumerate(free_idx):
            V[i, j_idx] = w_var[k]

    var_matrices[label] = V
    print(f"{label:20s}: mean Var={V[mask==1].mean():.4f}, max Var={V.max():.4f}")

# Plot 2x2 grid
ticks = [a.replace("pid_", "") for a in agents]
n     = len(agents)
fig, axes = plt.subplots(2, 2, figsize=(18, 16))
axes = axes.flatten()

for ax, (label, V) in zip(axes, var_matrices.items()):
    vmax = np.percentile(V[V > 0], 95) if (V > 0).any() else 1
    im = ax.imshow(V, cmap="hot_r", vmin=0, vmax=vmax, aspect="equal")
    ax.set_xticks(range(n))
    ax.set_xticklabels(ticks, rotation=90, fontsize=6)
    ax.set_yticks(range(n))
    ax.set_yticklabels(ticks, fontsize=6)
    ax.set_title(label, fontsize=13, fontweight="bold")
    ax.set_xlabel("Influencer", fontsize=10)
    ax.set_ylabel("Influenced", fontsize=10)
    plt.colorbar(im, ax=ax, label="Var(â[i,j])", shrink=0.8)

plt.suptitle("OLS Estimation Variance of A[i,j] — Per Topology\n"
             "High value = uncertain weight estimate",
             fontsize=14, fontweight="bold")
plt.tight_layout()
var_path = f"{OUT_DIR}/comparison_variance_heatmap_ols.png"
plt.savefig(var_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"\nSaved: {var_path}")

# ============================================================
# 7. Check: do |A[i,j]| > 1 entries have higher variance?
# ============================================================
print(f"\n{'='*70}")
print("CHECK: |A[i,j]| > 1  vs  variance")
print(f"{'='*70}")
print(f"\n{'Topology':<20} {'N |A|>1':>8} {'MeanVar |A|>1':>15} {'MeanVar |A|<=1':>16} {'Ratio':>8}")
print("-" * 70)

for label in matrices:
    A = matrices[label]
    V = var_matrices[label]
    mask_large = np.abs(A) > 1
    mask_small = (np.abs(A) <= 1) & (A != 0)

    n_large      = mask_large.sum()
    mean_var_large = V[mask_large].mean() if n_large > 0 else float('nan')
    mean_var_small = V[mask_small].mean() if mask_small.any() else float('nan')
    ratio = mean_var_large / mean_var_small if mean_var_small > 0 else float('nan')

    print(f"{label:<20} {n_large:>8} {mean_var_large:>15.4f} {mean_var_small:>16.4f} {ratio:>8.2f}x")

    if n_large > 0:
        rows, cols = np.where(mask_large)
        print(f"  Entries with |A[i,j]|>1:")
        for r, c in zip(rows, cols):
            print(f"    A[{agents[r]:12s}, {agents[c]:12s}] = "
                  f"{A[r,c]:+.4f}  Var={V[r,c]:.4f}")

# Scatter plot: |A[i,j]| vs Var(A[i,j]) for all topologies
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
axes = axes.flatten()
colors_scatter = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

for ax, (label, A), color in zip(axes, matrices.items(), colors_scatter):
    V    = var_matrices[label]
    nz   = A != 0
    vals = np.abs(A[nz])
    vars_ = V[nz]
    ax.scatter(vals, vars_, alpha=0.5, s=20, color=color)
    ax.axvline(1.0, color="red", linestyle="--", linewidth=1.2, label="|A|=1 threshold")
    ax.set_xlabel("|A[i,j]|", fontsize=11)
    ax.set_ylabel("Var(â[i,j])", fontsize=11)
    ax.set_title(label, fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

plt.suptitle("|A[i,j]| vs OLS Estimation Variance — Do Large Weights Have Higher Uncertainty?",
             fontsize=13, fontweight="bold")
plt.tight_layout()
scatter_path = f"{OUT_DIR}/comparison_abs_vs_variance_ols.png"
plt.savefig(scatter_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"\nSaved scatter plot: {scatter_path}")

# ============================================================
# 8. Individual A matrix heatmap for each topology
# ============================================================
agents_list = pd.read_csv(list(TOPOS.values())[0], index_col=0).index.tolist()
ticks_ind   = [a.replace("pid_", "") for a in agents_list]
n_ind       = len(agents_list)

for label, A in matrices.items():
    fig, ax = plt.subplots(figsize=(14, 12))
    vmax = np.max(np.abs(A))
    if vmax == 0:
        vmax = 1
    im = ax.imshow(A, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="equal")
    ax.set_xticks(range(n_ind))
    ax.set_xticklabels(ticks_ind, rotation=90, fontsize=7)
    ax.set_yticks(range(n_ind))
    ax.set_yticklabels(ticks_ind, fontsize=7)
    ax.set_xlabel("Influencer Agent", fontsize=12)
    ax.set_ylabel("Influenced Agent", fontsize=12)
    ax.set_title(f"OLS Influence Matrix A — {label}\nA[i,j] = influence of agent j on agent i",
                 fontsize=13, fontweight="bold")
    plt.colorbar(im, ax=ax, label="Influence Weight", shrink=0.8)
    plt.tight_layout()
    fname = label.lower().replace(" ", "_")
    path  = f"{OUT_DIR}/influence_matrix_A_{fname}_ols.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")

print("\nDone.")
