"""Plot sentiment score trajectories for four bitcoin topologies."""
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

topologies = {
    "Clusters": "text_simulation/text_simulation_output_interaction/bitcoin_sparse_clusters/sentiment_scores.csv",
    "Random": "text_simulation/text_simulation_output_interaction/bitcoin_sparse_random/sentiment_scores.csv",
    "Ring": "text_simulation/text_simulation_output_interaction/bitcoin_sparse_ring/sentiment_scores.csv",
    "Fully Connected": "text_simulation/text_simulation_output_interaction/bitcoin_fully_connected/sentiment_scores.csv",
}

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# --- Helper: compute per-persona mean score per round ---
def load_and_aggregate(path):
    df = pd.read_csv(path)
    agg = df.groupby(["round", "persona_id"])["score"].mean().reset_index()
    return agg

data = {name: load_and_aggregate(path) for name, path in topologies.items()}

colors = {"Clusters": "#e74c3c", "Random": "#3498db", "Ring": "#2ecc71", "Fully Connected": "#9b59b6"}

# ============================================================
# Plot 1: Mean sentiment per round with std band
# ============================================================
ax = axes[0, 0]
for name, agg in data.items():
    stats = agg.groupby("round")["score"].agg(["mean", "std"]).reset_index()
    ax.plot(stats["round"], stats["mean"], label=name, color=colors[name], linewidth=2)
    ax.fill_between(stats["round"], stats["mean"] - stats["std"], stats["mean"] + stats["std"],
                     alpha=0.15, color=colors[name])
ax.set_xlabel("Round")
ax.set_ylabel("Sentiment Score")
ax.set_title("Mean Sentiment ± 1 Std Dev")
ax.legend()
ax.set_xlim(0, 50)
ax.set_ylim(-3, 3)
ax.axhline(0, color="gray", linestyle="--", linewidth=0.5)
ax.grid(True, alpha=0.3)

# ============================================================
# Plot 2: Standard deviation over rounds
# ============================================================
ax = axes[0, 1]
for name, agg in data.items():
    stats = agg.groupby("round")["score"].std().reset_index()
    ax.plot(stats["round"], stats["score"], label=name, color=colors[name], linewidth=2)
ax.set_xlabel("Round")
ax.set_ylabel("Std Dev of Sentiment")
ax.set_title("Opinion Spread (Std Dev) Over Rounds")
ax.legend()
ax.set_xlim(0, 50)
ax.set_ylim(0, 2.5)
ax.grid(True, alpha=0.3)

# ============================================================
# Plots 3-5: Individual persona trajectories per topology
# ============================================================



# Plot 3: Individual persona traces (one subplot per topology, using bottom-left as combined)
ax = axes[1, 0]
for name, agg in data.items():
    for pid in agg["persona_id"].unique():
        pdf = agg[agg["persona_id"] == pid].sort_values("round")
        ax.plot(pdf["round"], pdf["score"], alpha=0.12, color=colors[name], linewidth=0.7)
    # Add mean line
    stats = agg.groupby("round")["score"].mean().reset_index()
    ax.plot(stats["round"], stats["score"], color=colors[name], linewidth=2.5, label=f"{name} mean")
ax.set_xlabel("Round")
ax.set_ylabel("Sentiment Score")
ax.set_title("Individual Persona Trajectories (All Topologies)")
ax.legend(fontsize=9)
ax.set_xlim(0, 50)
ax.set_ylim(-3.5, 3.5)
ax.axhline(0, color="gray", linestyle="--", linewidth=0.5)
ax.grid(True, alpha=0.3)

# Plot 4: Box plots at key rounds
ax = axes[1, 1]
key_rounds = [0, 10, 20, 30, 40, 50]
positions = []
labels = []
box_data = []
box_colors = []
width = 0.2
for i, rd in enumerate(key_rounds):
    for j, (name, agg) in enumerate(data.items()):
        scores = agg[agg["round"] == rd]["score"].values
        if len(scores) > 0:
            box_data.append(scores)
            positions.append(i * 1.2 + j * width)
            box_colors.append(colors[name])

bp = ax.boxplot(box_data, positions=positions, widths=width * 0.8, patch_artist=True,
                showfliers=False, medianprops=dict(color="black", linewidth=1.5))
for patch, color in zip(bp["boxes"], box_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)

# X-axis labels
tick_positions = [i * 1.2 + 1.5 * width for i in range(len(key_rounds))]
ax.set_xticks(tick_positions)
ax.set_xticklabels([f"R{r}" for r in key_rounds])
ax.set_ylabel("Sentiment Score")
ax.set_title("Score Distribution at Key Rounds")
ax.set_ylim(-3.5, 3.5)
ax.axhline(0, color="gray", linestyle="--", linewidth=0.5)
ax.grid(True, alpha=0.3, axis="y")

# Legend for box plot
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=colors[n], alpha=0.6, label=n) for n in colors]
ax.legend(handles=legend_elements, fontsize=9)

plt.suptitle("Bitcoin Opinion Consensus: Four Network Topologies (50 Rounds)", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("text_simulation/text_simulation_output_interaction/sentiment_comparison_4topologies.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved to text_simulation/text_simulation_output_interaction/sentiment_comparison_4topologies.png")
