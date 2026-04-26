"""Solve for influence matrix A from sentiment trajectories using masked OLS.

Methodology:
  We model opinion dynamics as x_i(t+1) = sum_j A[i,j] * x_j(t).
  For each agent i, we restrict to connected neighbors (graph mask) and solve
  unconstrained least squares: min_a ||X_free.T @ a - y_target||^2
  via np.linalg.lstsq (SVD-based, no constraints).

Usage: python solve_influence_matrix_ols.py <topology_name>
  topology_name: clusters, random, ring, or fully_connected
"""
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============================================================
# Parse topology argument
# ============================================================
if len(sys.argv) < 2:
    print("Usage: python solve_influence_matrix_ols.py <topology_name>")
    print("  topology_name: clusters, random, ring, or fully_connected")
    sys.exit(1)

topo = sys.argv[1]
topo_config = {
    "clusters": {
        "scores": "text_simulation/text_simulation_output_interaction/bitcoin_sparse_clusters/sentiment_scores.csv",
        "graph": "text_simulation/interaction_graph_sparse_clusters.json",
        "out_dir": "text_simulation/text_simulation_output_interaction/bitcoin_sparse_clusters",
        "label": "Sparse Clusters",
    },
    "random": {
        "scores": "text_simulation/text_simulation_output_interaction/bitcoin_sparse_random/sentiment_scores.csv",
        "graph": "text_simulation/interaction_graph_sparse_random.json",
        "out_dir": "text_simulation/text_simulation_output_interaction/bitcoin_sparse_random",
        "label": "Sparse Random",
    },
    "ring": {
        "scores": "text_simulation/text_simulation_output_interaction/bitcoin_sparse_ring/sentiment_scores.csv",
        "graph": "text_simulation/interaction_graph_sparse_ring.json",
        "out_dir": "text_simulation/text_simulation_output_interaction/bitcoin_sparse_ring",
        "label": "Sparse Ring",
    },
    "fully_connected": {
        "scores": "text_simulation/text_simulation_output_interaction/bitcoin_fully_connected/sentiment_scores.csv",
        "graph": "text_simulation/interaction_graph_fully_connected.json",
        "out_dir": "text_simulation/text_simulation_output_interaction/bitcoin_fully_connected",
        "label": "Fully Connected",
    },
}

if topo not in topo_config:
    print(f"Unknown topology: {topo}. Choose from: {list(topo_config.keys())}")
    sys.exit(1)

cfg = topo_config[topo]
print(f"\n{'='*70}")
print(f"Solving influence matrix (OLS) for: {cfg['label']}")
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

X = pivot_table.values
n_agents, n_rounds = X.shape
print(f"X shape: {X.shape}")

# Handle NaN
for i in range(n_agents):
    row = pd.Series(X[i, :])
    row = row.ffill().bfill()
    X[i, :] = row.values

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
                j = agent_idx[nb]
                mask[i, j] = 1

# Include self-influence
for i in range(n_agents):
    mask[i, i] = 1

avg_neighbors = mask.sum(axis=1).mean()
print(f"Avg neighbors per agent (incl self): {avg_neighbors:.1f}")

# ============================================================
# 3. Solve OLS per agent using graph mask
# ============================================================
print("\nSolving OLS for each agent...")

A = np.zeros((n_agents, n_agents))
mse_per_agent = np.zeros(n_agents)

X_input = X[:, :-1]   # (n_agents, T-1)  inputs  x(t)
X_target = X[:, 1:]   # (n_agents, T-1)  targets x(t+1)

for i in range(n_agents):
    free_idx = np.where(mask[i, :] == 1)[0]

    X_free = X_input[free_idx, :]   # (n_free, T-1)
    y_target = X_target[i, :]       # (T-1,)

    # Least squares: min ||X_free.T @ a - y_target||^2
    a_opt, _, _, _ = np.linalg.lstsq(X_free.T, y_target, rcond=None)

    for k, j_idx in enumerate(free_idx):
        A[i, j_idx] = a_opt[k]

    pred = a_opt @ X_free
    mse_per_agent[i] = np.mean((pred - y_target) ** 2)

    if (i + 1) % 10 == 0 or i == 0:
        print(f"  Agent {i+1}/{n_agents} ({agents[i]}): L1={np.sum(np.abs(a_opt)):.4f}, MSE={mse_per_agent[i]:.4f}")

# ============================================================
# 4. Print results
# ============================================================
print(f"\n{'='*70}")
print(f"RESULTS (OLS) — {cfg['label']}")
print(f"{'='*70}")

print(f"\n{'Agent':<12} {'L1 Norm':>8} {'MSE':>10}")
print("-" * 32)
for i, agent in enumerate(agents):
    l1 = np.sum(np.abs(A[i, :]))
    print(f"{agent:<12} {l1:8.4f} {mse_per_agent[i]:10.4f}")

print(f"\nOverall: Mean MSE = {mse_per_agent.mean():.4f}, Max L1 = {np.max([np.sum(np.abs(A[i,:])) for i in range(n_agents)]):.4f}")

# ============================================================
# 5. Save A matrix as CSV
# ============================================================
out_dir = cfg["out_dir"]
A_df = pd.DataFrame(A, index=agents, columns=agents)
csv_path = f"{out_dir}/influence_matrix_A_ols.csv"
A_df.to_csv(csv_path)
print(f"\nSaved influence matrix to {csv_path}")

# ============================================================
# 6. Plot heatmap
# ============================================================
fig, ax = plt.subplots(figsize=(14, 12))
vmax = np.max(np.abs(A))
if vmax == 0:
    vmax = 1
im = ax.imshow(A, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="equal")
ax.set_xticks(range(n_agents))
ax.set_xticklabels([a.replace("pid_", "") for a in agents], rotation=90, fontsize=7)
ax.set_yticks(range(n_agents))
ax.set_yticklabels([a.replace("pid_", "") for a in agents], fontsize=7)
ax.set_xlabel("Influencer Agent", fontsize=12)
ax.set_ylabel("Influenced Agent", fontsize=12)
ax.set_title(f"Influence Matrix A — OLS ({cfg['label']} Network)\nA[i,j] = influence of agent j on agent i", fontsize=13, fontweight="bold")
plt.colorbar(im, ax=ax, label="Influence Weight", shrink=0.8)
plt.tight_layout()
heatmap_path = f"{out_dir}/influence_matrix_heatmap_ols.png"
plt.savefig(heatmap_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved heatmap to {heatmap_path}")

# ============================================================
# 7. Plot predicted vs actual for 3 sample agents
# ============================================================
sample_agents = [agents[0], agents[len(agents)//2], agents[-1]]
fig, axes_plot = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

for idx, agent in enumerate(sample_agents):
    ax = axes_plot[idx]
    ii = agent_idx[agent]
    actual = X[ii, :]
    predicted = np.zeros(n_rounds)
    predicted[0] = actual[0]
    for t in range(n_rounds - 1):
        predicted[t + 1] = A[ii, :] @ X[:, t]

    ax.plot(all_rounds, actual, 'b-o', markersize=4, linewidth=1.5, label="Actual", alpha=0.8)
    ax.plot(all_rounds, predicted, 'r--s', markersize=3, linewidth=1.2, label="Predicted (AX)", alpha=0.8)
    ax.set_ylabel("Sentiment Score", fontsize=11)
    ax.set_title(f"{agent}", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_ylim(-3.5, 3.5)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.5)
    ax.grid(True, alpha=0.3)

axes_plot[-1].set_xlabel("Round", fontsize=11)
plt.suptitle(f"Predicted vs Actual Sentiment — OLS ({cfg['label']})", fontsize=14, fontweight="bold")
plt.tight_layout()
pred_path = f"{out_dir}/predicted_vs_actual_ols.png"
plt.savefig(pred_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved predicted vs actual to {pred_path}")
print(f"\nDone with {cfg['label']} (OLS)!")
