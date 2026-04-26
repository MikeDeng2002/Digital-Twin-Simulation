"""Plot network graphs for all topologies side by side."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

topologies = {
    "Sparse Clusters": "text_simulation/interaction_graph_sparse_clusters.json",
    "Sparse Random": "text_simulation/interaction_graph_sparse_random.json",
    "Sparse Ring": "text_simulation/interaction_graph_sparse_ring.json",
    "Fully Connected": "text_simulation/interaction_graph_fully_connected.json",
}

def load_graph(path):
    with open(path) as f:
        adj = json.load(f)
    G = nx.Graph()
    for node, neighbors in adj.items():
        G.add_node(node)
        for n in neighbors:
            G.add_edge(node, n)
    return G

fig, axes = plt.subplots(2, 2, figsize=(18, 16))
axes = axes.flatten()

for idx, (name, path) in enumerate(topologies.items()):
    ax = axes[idx]
    try:
        G = load_graph(path)
    except FileNotFoundError:
        ax.set_title(f"{name} (not found)", fontsize=13, fontweight="bold")
        ax.axis("off")
        continue

    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    degrees = [d for _, d in G.degree()]
    avg_deg = np.mean(degrees)

    # Choose layout based on topology
    if "Ring" in name:
        pos = nx.circular_layout(G)
    elif "Cluster" in name:
        pos = nx.spring_layout(G, seed=42, k=2.0, iterations=100)
    elif "Fully" in name:
        pos = nx.circular_layout(G)
    else:
        pos = nx.spring_layout(G, seed=42, k=1.5, iterations=80)

    # Color nodes by degree
    node_colors = [d for _, d in G.degree()]

    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.25, width=0.6, edge_color="gray")
    nc = nx.draw_networkx_nodes(G, pos, ax=ax, node_size=200,
                                 node_color=node_colors, cmap=plt.cm.YlOrRd,
                                 edgecolors="black", linewidths=0.5)
    # Short labels (just the number part)
    labels = {n: n.replace("pid_", "") for n in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, ax=ax, font_size=6, font_weight="bold")

    ax.set_title(f"{name}\n{n_nodes} nodes, {n_edges} edges, avg degree {avg_deg:.1f}",
                 fontsize=12, fontweight="bold")
    ax.axis("off")

plt.suptitle("Network Topologies for Bitcoin Opinion Simulation", fontsize=15, fontweight="bold")
plt.tight_layout()
out_path = "text_simulation/text_simulation_output_interaction/network_topologies.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved to {out_path}")
