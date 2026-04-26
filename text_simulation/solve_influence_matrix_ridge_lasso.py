"""Solve for influence matrix A using OLS, Ridge, and Lasso regression.

Methodology:
  x_i(t+1) = sum_j A[i,j] * x_j(t)
  For each agent i, we restrict to graph-connected neighbors (mask) and fit:
    OLS   : min  ||X_free.T @ a - y||^2
    Ridge : min  ||X_free.T @ a - y||^2 + alpha * ||a||^2   (alpha via GCV)
    Lasso : min  ||X_free.T @ a - y||^2 + alpha * ||a||_1   (alpha via k-fold CV)

  Lasso induces exact zeros within the neighborhood, revealing the truly
  sparse influence structure. Ridge shrinks all weights but keeps them nonzero,
  stabilizing estimates when neighbors are correlated.

Usage: python solve_influence_matrix_ridge_lasso.py <topology_name>
  topology_name: clusters, random, ring, or fully_connected
"""
import sys
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, Lasso, RidgeCV, LassoCV
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============================================================
# Parse topology argument
# ============================================================
if len(sys.argv) < 2:
    print("Usage: python solve_influence_matrix_ridge_lasso.py <topology_name>")
    print("  topology_name: clusters, random, ring, or fully_connected")
    sys.exit(1)

topo = sys.argv[1]
topo_config = {
    "clusters": {
        "scores": "text_simulation/text_simulation_output_interaction/bitcoin_sparse_clusters/sentiment_scores.csv",
        "graph":  "text_simulation/interaction_graph_sparse_clusters.json",
        "out_dir": "text_simulation/text_simulation_output_interaction/bitcoin_sparse_clusters",
        "label":  "Sparse Clusters",
    },
    "random": {
        "scores": "text_simulation/text_simulation_output_interaction/bitcoin_sparse_random/sentiment_scores.csv",
        "graph":  "text_simulation/interaction_graph_sparse_random.json",
        "out_dir": "text_simulation/text_simulation_output_interaction/bitcoin_sparse_random",
        "label":  "Sparse Random",
    },
    "ring": {
        "scores": "text_simulation/text_simulation_output_interaction/bitcoin_sparse_ring/sentiment_scores.csv",
        "graph":  "text_simulation/interaction_graph_sparse_ring.json",
        "out_dir": "text_simulation/text_simulation_output_interaction/bitcoin_sparse_ring",
        "label":  "Sparse Ring",
    },
    "fully_connected": {
        "scores": "text_simulation/text_simulation_output_interaction/bitcoin_fully_connected/sentiment_scores.csv",
        "graph":  "text_simulation/interaction_graph_fully_connected.json",
        "out_dir": "text_simulation/text_simulation_output_interaction/bitcoin_fully_connected",
        "label":  "Fully Connected",
    },
}

if topo not in topo_config:
    print(f"Unknown topology: {topo}. Choose from: {list(topo_config.keys())}")
    sys.exit(1)

cfg = topo_config[topo]
print(f"\n{'='*70}")
print(f"Solving influence matrix (OLS / Ridge / Lasso) for: {cfg['label']}")
print(f"{'='*70}")

# ============================================================
# 1. Load and preprocess data
# ============================================================
print("Loading data...")
df = pd.read_csv(cfg["scores"])
df = df[df["question"] == "Q1"]
df = df[df["round"] >= 1]

pivot = df.groupby(["persona_id", "round"])["score"].mean().reset_index()
pivot_table = pivot.pivot(index="persona_id", columns="round", values="score")

agents = sorted(pivot_table.index.tolist())
pivot_table = pivot_table.loc[agents]

all_rounds = sorted(pivot_table.columns.tolist())
print(f"Agents: {len(agents)}, Rounds available: {all_rounds[0]}-{all_rounds[-1]}")

X = pivot_table.values.astype(float)
n_agents, n_rounds = X.shape
print(f"X shape: {X.shape}")

for i in range(n_agents):
    row = pd.Series(X[i, :])
    X[i, :] = row.ffill().bfill().values

nan_count = np.isnan(X).sum()
if nan_count > 0:
    print(f"WARNING: {nan_count} NaN values remain, filling with 0")
    X = np.nan_to_num(X, nan=0.0)

# ============================================================
# 2. Build adjacency mask
# ============================================================
with open(cfg["graph"]) as f:
    graph = json.load(f)

mask = np.zeros((n_agents, n_agents), dtype=int)
agent_idx = {a: i for i, a in enumerate(agents)}

for agent, neighbors in graph.items():
    if agent in agent_idx:
        i = agent_idx[agent]
        for nb in neighbors:
            if nb in agent_idx:
                mask[i, agent_idx[nb]] = 1

for i in range(n_agents):
    mask[i, i] = 1

avg_neighbors = mask.sum(axis=1).mean()
print(f"Avg neighbors per agent (incl self): {avg_neighbors:.1f}")

# ============================================================
# 3. Regression setup
# ============================================================
X_input  = X[:, :-1]   # (n_agents, T)   states at t
X_target = X[:, 1:]    # (n_agents, T)   states at t+1
T = n_rounds - 1

# Alpha candidate grids
ALPHAS_RIDGE = np.logspace(-4, 2, 60)
ALPHAS_LASSO = np.logspace(-4, 1, 40)

A_ols   = np.zeros((n_agents, n_agents))
A_ridge = np.zeros((n_agents, n_agents))
A_lasso = np.zeros((n_agents, n_agents))

mse_ols   = np.zeros(n_agents)
mse_ridge = np.zeros(n_agents)
mse_lasso = np.zeros(n_agents)

alpha_ridge_sel = np.zeros(n_agents)
alpha_lasso_sel = np.zeros(n_agents)

# ============================================================
# 4. Fit per-agent: OLS, Ridge (GCV), Lasso (k-fold CV)
# ============================================================
print(f"\nSolving OLS / Ridge / Lasso for each agent (T={T} observations)...")

for i in range(n_agents):
    free_idx = np.where(mask[i, :] == 1)[0]
    n_free   = len(free_idx)

    # Design matrix: rows = time steps, cols = neighbor states
    Phi     = X_input[free_idx, :].T   # (T, n_free)
    y_tgt   = X_target[i, :]           # (T,)

    # ---- OLS ----
    a_ols_vec, _, _, _ = np.linalg.lstsq(Phi, y_tgt, rcond=None)

    # ---- Ridge (GCV — exact LOO approximation, no folds needed) ----
    rcv = RidgeCV(alphas=ALPHAS_RIDGE, cv=None, fit_intercept=False)
    rcv.fit(Phi, y_tgt)
    a_ridge_vec          = rcv.coef_
    alpha_ridge_sel[i]   = rcv.alpha_

    # ---- Lasso (k-fold CV, robust to small T) ----
    n_cv = min(5, max(2, T - 1))
    if T < 4:
        # Too few samples for reliable CV; fall back to a small fixed alpha
        lasso_model = Lasso(alpha=0.05, fit_intercept=False, max_iter=20000)
        lasso_model.fit(Phi, y_tgt)
        a_lasso_vec        = lasso_model.coef_
        alpha_lasso_sel[i] = 0.05
    else:
        lcv = LassoCV(
            alphas=ALPHAS_LASSO,
            cv=n_cv,
            fit_intercept=False,
            max_iter=20000,
        )
        lcv.fit(Phi, y_tgt)
        a_lasso_vec        = lcv.coef_
        alpha_lasso_sel[i] = lcv.alpha_

    # Store in full matrices
    for k, j_idx in enumerate(free_idx):
        A_ols[i, j_idx]   = a_ols_vec[k]
        A_ridge[i, j_idx] = a_ridge_vec[k]
        A_lasso[i, j_idx] = a_lasso_vec[k]

    mse_ols[i]   = np.mean((Phi @ a_ols_vec   - y_tgt) ** 2)
    mse_ridge[i] = np.mean((Phi @ a_ridge_vec - y_tgt) ** 2)
    mse_lasso[i] = np.mean((Phi @ a_lasso_vec - y_tgt) ** 2)

    if (i + 1) % 10 == 0 or i == 0:
        n_lasso_nz = np.count_nonzero(a_lasso_vec)
        print(
            f"  Agent {i+1:2d}/{n_agents} ({agents[i]}): "
            f"MSE OLS={mse_ols[i]:.4f}  Ridge={mse_ridge[i]:.4f}  Lasso={mse_lasso[i]:.4f} "
            f"| α_ridge={alpha_ridge_sel[i]:.4f}  α_lasso={alpha_lasso_sel[i]:.4f} "
            f"| Lasso nonzero={n_lasso_nz}/{n_free}"
        )

# ============================================================
# 5. Summary statistics
# ============================================================
def sparsity(A, mask_mat):
    """Fraction of mask-allowed entries that are exactly zero."""
    allowed = mask_mat == 1
    return (A[allowed] == 0).mean()

print(f"\n{'='*70}")
print(f"RESULTS — {cfg['label']}")
print(f"{'='*70}")
print(f"\n{'Method':<8} {'Mean MSE':>10} {'Max|A|':>10} "
      f"{'MeanL1/row':>12} {'Lasso sparsity':>16}")
print("-" * 62)
for name, A_m, mse_m in [("OLS",   A_ols,   mse_ols),
                          ("Ridge", A_ridge, mse_ridge),
                          ("Lasso", A_lasso, mse_lasso)]:
    mean_l1 = np.mean([np.sum(np.abs(A_m[i, :])) for i in range(n_agents)])
    spar     = sparsity(A_m, mask)
    print(f"{name:<8} {mse_m.mean():10.4f} {np.max(np.abs(A_m)):10.4f} "
          f"{mean_l1:12.4f} {spar:16.4f}")

print(f"\nSelected alpha_ridge: mean={alpha_ridge_sel.mean():.4f}  "
      f"min={alpha_ridge_sel.min():.4f}  max={alpha_ridge_sel.max():.4f}")
print(f"Selected alpha_lasso: mean={alpha_lasso_sel.mean():.4f}  "
      f"min={alpha_lasso_sel.min():.4f}  max={alpha_lasso_sel.max():.4f}")

# ============================================================
# 6. Save influence matrices as CSV
# ============================================================
out_dir = cfg["out_dir"]
for name, A_m in [("ols", A_ols), ("ridge", A_ridge), ("lasso", A_lasso)]:
    path = f"{out_dir}/influence_matrix_A_{name}.csv"
    pd.DataFrame(A_m, index=agents, columns=agents).to_csv(path)
    print(f"Saved: {path}")

# ============================================================
# 7. Three-way heatmap comparison
# ============================================================
tick_labels = [a.replace("pid_", "") for a in agents]
fig, axes = plt.subplots(1, 3, figsize=(24, 8))

for ax, (name, A_m) in zip(axes, [("OLS", A_ols), ("Ridge", A_ridge), ("Lasso", A_lasso)]):
    vmax = np.max(np.abs(A_m))
    vmax = vmax if vmax > 0 else 1.0
    im = ax.imshow(A_m, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="equal")
    ax.set_xticks(range(n_agents))
    ax.set_xticklabels(tick_labels, rotation=90, fontsize=6)
    ax.set_yticks(range(n_agents))
    ax.set_yticklabels(tick_labels, fontsize=6)
    ax.set_xlabel("Influencer Agent", fontsize=10)
    ax.set_ylabel("Influenced Agent", fontsize=10)
    nonzero_in_mask = np.count_nonzero(A_m[mask == 1])
    total_in_mask   = (mask == 1).sum()
    ax.set_title(
        f"{name}  ({cfg['label']})\n"
        f"nonzero in mask: {nonzero_in_mask}/{total_in_mask}  "
        f"MSE={mse_ols.mean() if name=='OLS' else (mse_ridge.mean() if name=='Ridge' else mse_lasso.mean()):.4f}",
        fontsize=10, fontweight="bold"
    )
    plt.colorbar(im, ax=ax, label="Influence Weight", shrink=0.75)

plt.suptitle(
    f"Influence Matrix A — OLS vs Ridge vs Lasso\n{cfg['label']} Network",
    fontsize=13, fontweight="bold"
)
plt.tight_layout()
heatmap_path = f"{out_dir}/influence_matrix_comparison_ridge_lasso.png"
plt.savefig(heatmap_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"\nSaved heatmap comparison: {heatmap_path}")

# ============================================================
# 8. Sparsity analysis: per-agent nonzero counts
# ============================================================
nz_ols   = [(mask[i, :] == 1).sum() for i in range(n_agents)]  # all mask entries are nonzero in OLS
nz_ridge = [(mask[i, :] == 1).sum() for i in range(n_agents)]  # Ridge keeps all nonzero
nz_lasso = [np.count_nonzero(A_lasso[i, mask[i, :] == 1]) for i in range(n_agents)]
n_allowed = [(mask[i, :] == 1).sum() for i in range(n_agents)]

fig, axes = plt.subplots(1, 2, figsize=(16, 5))

# Left: bar chart of nonzero counts per agent (Lasso vs allowed)
x_pos = np.arange(n_agents)
axes[0].bar(x_pos - 0.2, n_allowed, 0.4, label="Allowed (mask)", color="#4C72B0", alpha=0.7)
axes[0].bar(x_pos + 0.2, nz_lasso,  0.4, label="Lasso nonzero",  color="#DD8452", alpha=0.7)
axes[0].set_xticks(x_pos)
axes[0].set_xticklabels(tick_labels, rotation=90, fontsize=6)
axes[0].set_ylabel("Number of nonzero weights", fontsize=11)
axes[0].set_title(f"Lasso Sparsification Within Neighborhood\n{cfg['label']}", fontsize=12, fontweight="bold")
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3, axis="y")

# Right: distribution of |A[i,j]| for OLS vs Ridge vs Lasso (nonzero entries only)
for name, A_m, color in [("OLS",   A_ols,   "#4C72B0"),
                          ("Ridge", A_ridge, "#55A868"),
                          ("Lasso", A_lasso, "#DD8452")]:
    vals = np.abs(A_m[mask == 1])
    vals = vals[vals > 1e-10]
    if len(vals) > 0:
        axes[1].hist(vals, bins=25, alpha=0.5, label=name, color=color, density=True)

axes[1].set_xlabel("|A[i,j]| (nonzero, mask-allowed entries)", fontsize=11)
axes[1].set_ylabel("Density", fontsize=11)
axes[1].set_title(f"Weight Magnitude Distribution\n{cfg['label']}", fontsize=12, fontweight="bold")
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)

plt.suptitle("Sparsity & Weight Analysis — OLS / Ridge / Lasso", fontsize=13, fontweight="bold")
plt.tight_layout()
sparsity_path = f"{out_dir}/influence_sparsity_ridge_lasso.png"
plt.savefig(sparsity_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved sparsity analysis: {sparsity_path}")

# ============================================================
# 9. MSE and alpha comparison (per-agent)
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# MSE per agent
axes[0].plot(mse_ols,   'b-o', markersize=4, linewidth=1.2, label="OLS",   alpha=0.8)
axes[0].plot(mse_ridge, 'g-s', markersize=4, linewidth=1.2, label="Ridge", alpha=0.8)
axes[0].plot(mse_lasso, 'r-^', markersize=4, linewidth=1.2, label="Lasso", alpha=0.8)
axes[0].set_xlabel("Agent index", fontsize=11)
axes[0].set_ylabel("Training MSE", fontsize=11)
axes[0].set_title("Per-agent Training MSE", fontsize=12, fontweight="bold")
axes[0].legend(fontsize=9)
axes[0].grid(True, alpha=0.3)

# Selected alpha_ridge per agent
axes[1].semilogy(alpha_ridge_sel, 'g-o', markersize=4, linewidth=1.2)
axes[1].set_xlabel("Agent index", fontsize=11)
axes[1].set_ylabel("alpha (log scale)", fontsize=11)
axes[1].set_title("Ridge: Selected alpha per agent (GCV)", fontsize=12, fontweight="bold")
axes[1].grid(True, alpha=0.3)

# Selected alpha_lasso per agent
axes[2].semilogy(alpha_lasso_sel, 'r-o', markersize=4, linewidth=1.2)
axes[2].set_xlabel("Agent index", fontsize=11)
axes[2].set_ylabel("alpha (log scale)", fontsize=11)
axes[2].set_title("Lasso: Selected alpha per agent (CV)", fontsize=12, fontweight="bold")
axes[2].grid(True, alpha=0.3)

plt.suptitle(f"Regression Diagnostics — {cfg['label']}", fontsize=13, fontweight="bold")
plt.tight_layout()
diag_path = f"{out_dir}/regression_diagnostics_ridge_lasso.png"
plt.savefig(diag_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved diagnostics: {diag_path}")

# ============================================================
# 10. Ridge variance of A[i,j] (closed form)
#     Var_ridge(a) = sigma^2 * (Phi^T Phi + alpha I)^{-1} Phi^T Phi (Phi^T Phi + alpha I)^{-1}
# ============================================================
print("\nComputing Ridge estimation variance...")
V_ridge = np.zeros((n_agents, n_agents))

for i in range(n_agents):
    free_idx = np.where(mask[i, :] == 1)[0]
    n_free   = len(free_idx)
    Phi      = X_input[free_idx, :].T        # (T, n_free)
    y_tgt    = X_target[i, :]
    alpha    = alpha_ridge_sel[i]

    a_r    = A_ridge[i, free_idx]
    resid  = Phi @ a_r - y_tgt
    dof    = max(1, T - n_free)
    sigma2 = np.sum(resid ** 2) / dof

    PhiTPhi = Phi.T @ Phi                          # (n_free, n_free)
    reg_inv = np.linalg.pinv(PhiTPhi + alpha * np.eye(n_free))
    # Sandwich covariance
    cov_ridge = sigma2 * reg_inv @ PhiTPhi @ reg_inv
    w_var = np.diag(cov_ridge)

    for k, j_idx in enumerate(free_idx):
        V_ridge[i, j_idx] = w_var[k]

# Plot Ridge variance heatmap
fig, ax = plt.subplots(figsize=(12, 10))
vmax_v = np.percentile(V_ridge[V_ridge > 0], 95) if (V_ridge > 0).any() else 1.0
im = ax.imshow(V_ridge, cmap="hot_r", vmin=0, vmax=vmax_v, aspect="equal")
ax.set_xticks(range(n_agents))
ax.set_xticklabels(tick_labels, rotation=90, fontsize=6)
ax.set_yticks(range(n_agents))
ax.set_yticklabels(tick_labels, fontsize=6)
ax.set_xlabel("Influencer Agent", fontsize=12)
ax.set_ylabel("Influenced Agent", fontsize=12)
ax.set_title(
    f"Ridge Estimation Variance of A[i,j] — {cfg['label']}\n"
    "High value = uncertain weight (may reflect collinearity between agents)",
    fontsize=12, fontweight="bold"
)
plt.colorbar(im, ax=ax, label="Var(â_ridge[i,j])", shrink=0.8)
plt.tight_layout()
var_path = f"{out_dir}/ridge_variance_heatmap.png"
plt.savefig(var_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved Ridge variance heatmap: {var_path}")

# Save variance matrix
pd.DataFrame(V_ridge, index=agents, columns=agents).to_csv(
    f"{out_dir}/ridge_variance_matrix.csv"
)

# ============================================================
# 11. Lasso: effective sparse influence graph
#     Show edges that survived Lasso shrinkage
# ============================================================
lasso_edges = np.abs(A_lasso) > 1e-10
lasso_edge_frac = lasso_edges[mask == 1].mean()
print(f"\nLasso: {lasso_edge_frac*100:.1f}% of mask-allowed edges survive (nonzero)")
print(f"       {(~lasso_edges[mask == 1]).sum()} edges zeroed out of {(mask == 1).sum()} allowed")

# Overlay: mask edges vs Lasso-surviving edges
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Mask (graph structure)
axes[0].imshow(mask, cmap="Blues", aspect="equal")
axes[0].set_title(f"Graph Mask (allowed edges)\n{cfg['label']}", fontsize=12, fontweight="bold")
axes[0].set_xlabel("Influencer", fontsize=10)
axes[0].set_ylabel("Influenced", fontsize=10)
axes[0].set_xticks(range(n_agents))
axes[0].set_xticklabels(tick_labels, rotation=90, fontsize=6)
axes[0].set_yticks(range(n_agents))
axes[0].set_yticklabels(tick_labels, fontsize=6)

# Lasso nonzero pattern
axes[1].imshow(lasso_edges.astype(float), cmap="Oranges", aspect="equal")
axes[1].set_title(
    f"Lasso Nonzero Pattern ({lasso_edge_frac*100:.1f}% of allowed survive)\n{cfg['label']}",
    fontsize=12, fontweight="bold"
)
axes[1].set_xlabel("Influencer", fontsize=10)
axes[1].set_ylabel("Influenced", fontsize=10)
axes[1].set_xticks(range(n_agents))
axes[1].set_xticklabels(tick_labels, rotation=90, fontsize=6)
axes[1].set_yticks(range(n_agents))
axes[1].set_yticklabels(tick_labels, fontsize=6)

plt.suptitle("Graph Mask vs Lasso-Recovered Sparse Influence Network",
             fontsize=13, fontweight="bold")
plt.tight_layout()
graph_path = f"{out_dir}/lasso_sparse_influence_graph.png"
plt.savefig(graph_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved sparse influence graph: {graph_path}")

print(f"\nDone with {cfg['label']} (OLS / Ridge / Lasso)!")
