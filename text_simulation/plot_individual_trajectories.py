"""Plot each agent's sentiment trajectory separately for each topology."""
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

def load_and_aggregate(path):
    df = pd.read_csv(path)
    agg = df.groupby(["round", "persona_id"])["score"].mean().reset_index()
    return agg

# Use a colormap with enough distinct colors for 30 personas
cmap = plt.cm.get_cmap("tab20", 20)
extra_cmap = plt.cm.get_cmap("tab20b", 20)

fig, axes = plt.subplots(4, 1, figsize=(16, 24), sharex=True)

for idx, (name, path) in enumerate(topologies.items()):
    ax = axes[idx]
    agg = load_and_aggregate(path)
    personas = sorted(agg["persona_id"].unique())

    for i, pid in enumerate(personas):
        pdf = agg[agg["persona_id"] == pid].sort_values("round")
        if i < 20:
            color = cmap(i)
        else:
            color = extra_cmap(i - 20)
        ax.plot(pdf["round"], pdf["score"], marker=".", markersize=3, linewidth=1.2,
                label=pid, color=color, alpha=0.85)

    ax.set_ylabel("Sentiment Score", fontsize=12)
    ax.set_title(f"{name} Topology — Individual Agent Trajectories", fontsize=13, fontweight="bold")
    ax.set_ylim(-3.5, 3.5)
    ax.set_xlim(0, 50)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.5)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=6, ncol=6, loc="upper right", framealpha=0.8)

axes[-1].set_xlabel("Round", fontsize=12)
plt.suptitle("Bitcoin Opinion: Per-Agent Sentiment Trajectories (50 Rounds)", fontsize=15, fontweight="bold")
plt.tight_layout()
out_path = "text_simulation/text_simulation_output_interaction/individual_trajectories_4topologies.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved to {out_path}")
