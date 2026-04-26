"""Generate sparse interaction graphs for 30 personas.

Creates three graph topologies:
1. Random sparse (~5 neighbors per node)
2. Clustered (3 clusters of 10, sparse cross-cluster bridges)
3. Ring + shortcuts (small-world style)

All graphs are undirected (symmetric adjacency lists).
Uses a fixed random seed for reproducibility.
"""

import json
import random
import os

SEED = 42

# All 30 personas: 15 original + 15 new
ORIGINAL_15 = [
    "pid_1000", "pid_1001", "pid_1002", "pid_1003", "pid_1004",
    "pid_1036", "pid_1057", "pid_1064", "pid_1111", "pid_1128",
    "pid_1199", "pid_1204", "pid_1228", "pid_1365", "pid_1457",
]

NEW_15 = [
    "pid_5", "pid_15", "pid_25", "pid_50", "pid_75",
    "pid_150", "pid_200", "pid_300", "pid_400", "pid_600",
    "pid_700", "pid_1500", "pid_1600", "pid_1700", "pid_1900",
]

ALL_30 = sorted(ORIGINAL_15 + NEW_15, key=lambda x: int(x.split("_")[1]))


def add_edge(graph, a, b):
    """Add undirected edge between a and b."""
    if b not in graph[a]:
        graph[a].append(b)
    if a not in graph[b]:
        graph[b].append(a)


def make_empty_graph(nodes):
    return {n: [] for n in nodes}


def generate_random_sparse(nodes, avg_degree=5, seed=SEED):
    """Random sparse graph: each node targets ~avg_degree neighbors."""
    rng = random.Random(seed)
    graph = make_empty_graph(nodes)

    for node in nodes:
        others = [n for n in nodes if n != node]
        rng.shuffle(others)
        # Pick avg_degree neighbors (some may already be connected)
        for neighbor in others[:avg_degree]:
            add_edge(graph, node, neighbor)

    # Sort neighbor lists for readability
    for node in graph:
        graph[node] = sorted(graph[node], key=lambda x: int(x.split("_")[1]))

    total_edges = sum(len(v) for v in graph.values()) // 2
    print(f"Random sparse: {len(nodes)} nodes, {total_edges} edges, "
          f"avg degree {sum(len(v) for v in graph.values()) / len(nodes):.1f}")
    return graph


def generate_clustered(nodes, num_clusters=3, bridge_edges=3, seed=SEED):
    """Clustered graph: nodes split into clusters, fully connected within,
    with bridge_edges random cross-cluster edges between each pair."""
    rng = random.Random(seed)
    graph = make_empty_graph(nodes)

    # Shuffle and split into clusters
    shuffled = list(nodes)
    rng.shuffle(shuffled)
    cluster_size = len(nodes) // num_clusters
    clusters = []
    for i in range(num_clusters):
        start = i * cluster_size
        end = start + cluster_size if i < num_clusters - 1 else len(nodes)
        clusters.append(shuffled[start:end])

    print(f"Clusters: {[len(c) for c in clusters]}")
    for i, cluster in enumerate(clusters):
        print(f"  Cluster {chr(65+i)}: {sorted(cluster, key=lambda x: int(x.split('_')[1]))}")

    # Fully connect within each cluster
    for cluster in clusters:
        for i in range(len(cluster)):
            for j in range(i + 1, len(cluster)):
                add_edge(graph, cluster[i], cluster[j])

    # Add bridge edges between each pair of clusters
    for i in range(num_clusters):
        for j in range(i + 1, num_clusters):
            possible_bridges = [
                (a, b) for a in clusters[i] for b in clusters[j]
            ]
            rng.shuffle(possible_bridges)
            for a, b in possible_bridges[:bridge_edges]:
                add_edge(graph, a, b)

    # Sort neighbor lists
    for node in graph:
        graph[node] = sorted(graph[node], key=lambda x: int(x.split("_")[1]))

    total_edges = sum(len(v) for v in graph.values()) // 2
    print(f"Clustered: {len(nodes)} nodes, {total_edges} edges, "
          f"avg degree {sum(len(v) for v in graph.values()) / len(nodes):.1f}")
    return graph


def generate_ring_shortcuts(nodes, num_shortcuts=10, seed=SEED):
    """Ring + random shortcuts (small-world style).
    Each node connects to its 2 ring neighbors + random shortcut edges."""
    rng = random.Random(seed)
    graph = make_empty_graph(nodes)

    n = len(nodes)
    # Ring: connect each node to next and previous
    for i in range(n):
        add_edge(graph, nodes[i], nodes[(i + 1) % n])

    # Add random shortcut edges
    all_pairs = [(nodes[i], nodes[j]) for i in range(n) for j in range(i + 2, n)
                 if not (i == 0 and j == n - 1)]  # exclude ring neighbors
    rng.shuffle(all_pairs)
    added = 0
    for a, b in all_pairs:
        if added >= num_shortcuts:
            break
        if b not in graph[a]:
            add_edge(graph, a, b)
            added += 1

    # Sort neighbor lists
    for node in graph:
        graph[node] = sorted(graph[node], key=lambda x: int(x.split("_")[1]))

    total_edges = sum(len(v) for v in graph.values()) // 2
    print(f"Ring + shortcuts: {len(nodes)} nodes, {total_edges} edges, "
          f"avg degree {sum(len(v) for v in graph.values()) / len(nodes):.1f}")
    return graph


def save_graph(graph, filepath):
    """Save graph as JSON with sorted keys."""
    # Sort by numeric pid
    sorted_graph = dict(sorted(graph.items(), key=lambda x: int(x[0].split("_")[1])))
    with open(filepath, 'w') as f:
        json.dump(sorted_graph, f, indent=2)
    print(f"Saved: {filepath}")


if __name__ == "__main__":
    output_dir = os.path.dirname(os.path.abspath(__file__))

    print("Generating sparse interaction graphs for 30 personas...\n")

    # 1. Random sparse
    random_graph = generate_random_sparse(ALL_30)
    save_graph(random_graph, os.path.join(output_dir, "interaction_graph_sparse_random.json"))
    print()

    # 2. Clustered
    clustered_graph = generate_clustered(ALL_30)
    save_graph(clustered_graph, os.path.join(output_dir, "interaction_graph_sparse_clusters.json"))
    print()

    # 3. Ring + shortcuts
    ring_graph = generate_ring_shortcuts(ALL_30)
    save_graph(ring_graph, os.path.join(output_dir, "interaction_graph_sparse_ring.json"))

    print("\nDone! All 3 graph files generated.")
