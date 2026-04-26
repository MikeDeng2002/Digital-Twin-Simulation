"""Solve for influence matrix A from sentiment trajectories using constrained optimization."""
import json
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============================================================
# 1. Load and preprocess data
# ============================================================
print("Loading data...")
scores_path = "text_simulation/text_simulation_output_interaction/bitcoin_fully_connected/sentiment_scores.csv"
graph_path = "text_simulation/interaction_graph_fully_connected.json"

df = pd.read_csv(scores_path)
# Keep only Q1 rows with round >= 1
df = df[df["question"] == "Q1"]
df = df[df["round"] >= 1]

# Pivot: rows = persona_id, columns = round, values = score
pivot = df.groupby(["persona_id", "round"])["score"].mean().reset_index()
pivot_table = pivot.pivot(index="persona_id", columns="round", values="score")

# Sort agents alphabetically
agents = sorted(pivot_table.index.tolist())
pivot_table = pivot_table.loc[agents]

# Get contiguous rounds
all_rounds = sorted(pivot_table.columns.tolist())
print(f"Agents: {len(agents)}, Rounds available: {all_rounds[0]}-{all_rounds[-1]}")

# Build X matrix (n_agents x n_rounds)
X = pivot_table.values  # shape: (n_agents, n_rounds)
n_agents, n_rounds = X.shape
print(f"X shape: {X.shape}")

# Handle NaN: forward-fill then backward-fill per agent
for i in range(n_agents):
    row = pd.Series(X[i, :])
    row = row.ffill().bfill()
    X[i, :] = row.values

# Verify no NaN left
nan_count = np.isnan(X).sum()
if nan_count > 0:
    print(f"WARNING: {nan_count} NaN values remain, filling with 0")
    X = np.nan_to_num(X, nan=0.0)

# ============================================================
# 2. Build adjacency mask
# ============================================================
with open(graph_path) as f:
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

print(f"Mask nonzeros per row: {mask.sum(axis=1).mean():.0f} (should be 30 for fully connected + self)")

# ============================================================
# 3. Solve optimization for each agent
# ============================================================
print("\nSolving optimization for each agent...")

A = np.zeros((n_agents, n_agents))
mse_per_agent = np.zeros(n_agents)

for i in range(n_agents):
    # Free indices: where mask[i, :] == 1
    free_idx = np.where(mask[i, :] == 1)[0]
    n_free = len(free_idx)

    # Target: X[i, t+1] for t = 0..T-2
    # Input: X[:, t] for t = 0..T-2
    # We use columns 0..T-2 as input, columns 1..T-1 as target
    X_input = X[:, :-1]  # (n_agents, T-1)
    y_target = X[i, 1:]  # (T-1,)

    # Extract only free columns from X_input
    X_free = X_input[free_idx, :]  # (n_free, T-1)

    # Decision variables: [a_0, ..., a_{n_free-1}, u_0, ..., u_{n_free-1}]
    # a_j are the weights, u_j are auxiliary variables for L1
    # Total: 2 * n_free variables

    def objective(z):
        a = z[:n_free]
        pred = a @ X_free  # (T-1,)
        residuals = pred - y_target
        return np.sum(residuals ** 2)

    def grad_objective(z):
        a = z[:n_free]
        pred = a @ X_free
        residuals = pred - y_target  # (T-1,)
        grad_a = 2.0 * (X_free @ residuals)  # (n_free,)
        grad_u = np.zeros(n_free)
        return np.concatenate([grad_a, grad_u])

    # Constraints:
    # (1) sum(u) <= 1
    # (2) a_j <= u_j  =>  u_j - a_j >= 0
    # (3) a_j >= -u_j  =>  u_j + a_j >= 0
    # (4) u_j >= 0 (handled by bounds)

    constraints = []

    # sum(u) <= 1  =>  1 - sum(u) >= 0
    constraints.append({
        'type': 'ineq',
        'fun': lambda z: 1.0 - np.sum(z[n_free:]),
        'jac': lambda z: np.concatenate([np.zeros(n_free), -np.ones(n_free)])
    })

    # u_j - a_j >= 0 for each j
    for j in range(n_free):
        def make_upper(j=j):
            def fun(z):
                return z[n_free + j] - z[j]
            def jac(z):
                g = np.zeros(2 * n_free)
                g[j] = -1.0
                g[n_free + j] = 1.0
                return g
            return {'type': 'ineq', 'fun': fun, 'jac': jac}
        constraints.append(make_upper(j))

    # u_j + a_j >= 0 for each j
    for j in range(n_free):
        def make_lower(j=j):
            def fun(z):
                return z[n_free + j] + z[j]
            def jac(z):
                g = np.zeros(2 * n_free)
                g[j] = 1.0
                g[n_free + j] = 1.0
                return g
            return {'type': 'ineq', 'fun': fun, 'jac': jac}
        constraints.append(make_lower(j))

    # Bounds: a_j unbounded (but effectively in [-1, 1]), u_j >= 0
    bounds = [(None, None)] * n_free + [(0, None)] * n_free

    # Initial guess: small random values
    z0 = np.zeros(2 * n_free)
    z0[:n_free] = 0.01 * np.random.randn(n_free)
    z0[n_free:] = np.abs(z0[:n_free]) + 0.01

    result = minimize(objective, z0, method='SLSQP', jac=grad_objective,
                      bounds=bounds, constraints=constraints,
                      options={'maxiter': 1000, 'ftol': 1e-12})

    a_opt = result.x[:n_free]

    # Store in A matrix
    for k, j in enumerate(free_idx):
        A[i, j] = a_opt[k]

    # Compute MSE
    pred = a_opt @ X_free
    mse_per_agent[i] = np.mean((pred - y_target) ** 2)

    if (i + 1) % 10 == 0 or i == 0:
        print(f"  Agent {i+1}/{n_agents} ({agents[i]}): L1={np.sum(np.abs(a_opt)):.4f}, MSE={mse_per_agent[i]:.4f}")

# ============================================================
# 4. Print results
# ============================================================
print("\n" + "=" * 70)
print("RESULTS")
print("=" * 70)

print(f"\n{'Agent':<12} {'L1 Norm':>8} {'MSE':>10}")
print("-" * 32)
for i, agent in enumerate(agents):
    l1 = np.sum(np.abs(A[i, :]))
    print(f"{agent:<12} {l1:8.4f} {mse_per_agent[i]:10.4f}")

print(f"\nOverall: Mean MSE = {mse_per_agent.mean():.4f}, Max L1 = {np.max([np.sum(np.abs(A[i,:])) for i in range(n_agents)]):.4f}")

# ============================================================
# 5. Save A matrix as CSV
# ============================================================
out_dir = "text_simulation/text_simulation_output_interaction/bitcoin_fully_connected"
A_df = pd.DataFrame(A, index=agents, columns=agents)
csv_path = f"{out_dir}/influence_matrix_A.csv"
A_df.to_csv(csv_path)
print(f"\nSaved influence matrix to {csv_path}")

# ============================================================
# 6. Plot heatmap
# ============================================================
fig, ax = plt.subplots(figsize=(14, 12))
vmax = np.max(np.abs(A))
im = ax.imshow(A, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="equal")
ax.set_xticks(range(n_agents))
ax.set_xticklabels([a.replace("pid_", "") for a in agents], rotation=90, fontsize=7)
ax.set_yticks(range(n_agents))
ax.set_yticklabels([a.replace("pid_", "") for a in agents], fontsize=7)
ax.set_xlabel("Influencer Agent", fontsize=12)
ax.set_ylabel("Influenced Agent", fontsize=12)
ax.set_title("Influence Matrix A (Fully Connected Network)\nA[i,j] = influence of agent j on agent i", fontsize=13, fontweight="bold")
plt.colorbar(im, ax=ax, label="Influence Weight", shrink=0.8)
plt.tight_layout()
heatmap_path = f"{out_dir}/influence_matrix_heatmap.png"
plt.savefig(heatmap_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved heatmap to {heatmap_path}")

# ============================================================
# 7. Plot predicted vs actual for 3 sample agents
# ============================================================
sample_agents = [agents[0], agents[len(agents)//2], agents[-1]]
fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

for idx, agent in enumerate(sample_agents):
    ax = axes[idx]
    i = agent_idx[agent]
    actual = X[i, :]
    predicted = np.zeros(n_rounds)
    predicted[0] = actual[0]
    for t in range(n_rounds - 1):
        predicted[t + 1] = A[i, :] @ X[:, t]

    rounds_plot = all_rounds
    ax.plot(rounds_plot, actual, 'b-o', markersize=4, linewidth=1.5, label="Actual", alpha=0.8)
    ax.plot(rounds_plot, predicted, 'r--s', markersize=3, linewidth=1.2, label="Predicted (AX)", alpha=0.8)
    ax.set_ylabel("Sentiment Score", fontsize=11)
    ax.set_title(f"{agent}", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_ylim(-3.5, 3.5)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.5)
    ax.grid(True, alpha=0.3)

axes[-1].set_xlabel("Round", fontsize=11)
plt.suptitle("Predicted vs Actual Sentiment (Influence Model)", fontsize=14, fontweight="bold")
plt.tight_layout()
pred_path = f"{out_dir}/predicted_vs_actual.png"
plt.savefig(pred_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved predicted vs actual to {pred_path}")
