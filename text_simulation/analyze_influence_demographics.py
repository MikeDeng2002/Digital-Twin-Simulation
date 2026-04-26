"""Analyze how persona demographics relate to influence matrix A_{i,j} weights (sparse ring)."""
import re
import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============================================================
# Paths
# ============================================================
BASE = Path(__file__).resolve().parent.parent
OUT_DIR = BASE / "text_simulation" / "text_simulation_output_interaction" / "bitcoin_sparse_ring"
A_CSV = OUT_DIR / "influence_matrix_A.csv"
GRAPH_JSON = BASE / "text_simulation" / "interaction_graph_sparse_ring.json"
PERSONA_DIR = BASE / "data" / "mega_persona_summary_text"

# ============================================================
# 1. Load influence matrix and graph
# ============================================================
print("=" * 70)
print("INFLUENCE MATRIX & DEMOGRAPHIC ANALYSIS (Sparse Ring)")
print("=" * 70)

A_df = pd.read_csv(A_CSV, index_col=0)
agents = list(A_df.index)
A = A_df.values
n = len(agents)
print(f"\nLoaded {n}x{n} influence matrix with agents: {agents[:5]} ...")

with open(GRAPH_JSON) as f:
    graph = json.load(f)

# Build mask (same logic as solve script)
agent_idx = {a: i for i, a in enumerate(agents)}
mask = np.zeros((n, n), dtype=int)
for agent, neighbors in graph.items():
    if agent in agent_idx:
        i = agent_idx[agent]
        for nb in neighbors:
            if nb in agent_idx:
                mask[i, agent_idx[nb]] = 1
for i in range(n):
    mask[i, i] = 1

# ============================================================
# 2. Parse demographics from persona summary text files
# ============================================================
DEMO_FIELDS = [
    ("Geographic region", "region"),
    ("Gender", "sex"),
    ("Age", "age"),
    ("Religion", "religion"),
    ("Political affiliation", "party"),
    ("Income", "income"),
    ("Political views", "views"),
]

def parse_demographics(pid: str) -> dict:
    """Parse demographic fields from the persona summary text file."""
    txt_path = PERSONA_DIR / f"{pid}_mega_persona.txt"
    if not txt_path.exists():
        return {}
    text = txt_path.read_text(encoding="utf-8")
    demo = {}
    for field_name, key in DEMO_FIELDS:
        match = re.search(rf"^{re.escape(field_name)}:\s*(.+)$", text, re.MULTILINE)
        if match:
            demo[key] = match.group(1).strip()
    return demo

print("\nParsing demographics...")
demographics = {}
for pid in agents:
    demographics[pid] = parse_demographics(pid)

# Print sample
sample_pid = agents[0]
print(f"  Sample ({sample_pid}): {demographics[sample_pid]}")

# ============================================================
# 3. Helper: compare demographics
# ============================================================
def same_demo(pid_i, pid_j, key):
    """Return True if two personas share the same value for a demographic key."""
    d_i = demographics.get(pid_i, {})
    d_j = demographics.get(pid_j, {})
    v_i = d_i.get(key)
    v_j = d_j.get(key)
    if v_i is None or v_j is None:
        return None
    return v_i == v_j

# ============================================================
# Analysis 1: Top 20 Influence Pairs
# ============================================================
print("\n" + "=" * 70)
print("ANALYSIS 1: TOP 20 INFLUENCE PAIRS BY |A_{i,j}|")
print("=" * 70)

# Collect all off-diagonal nonzero entries
pairs = []
for i in range(n):
    for j in range(n):
        if i != j and mask[i, j] == 1:
            val = A[i, j]
            if abs(val) > 1e-10:  # skip near-zero
                pairs.append((agents[i], agents[j], val))

pairs.sort(key=lambda x: abs(x[2]), reverse=True)

# Separate positive and negative
pos_pairs = [(a, b, v) for a, b, v in pairs if v > 0]
neg_pairs = [(a, b, v) for a, b, v in pairs if v < 0]

def print_pair_table(pair_list, title, top_k=10):
    print(f"\n--- {title} (top {top_k}) ---")
    header = (f"{'Rank':<5} {'i (influenced)':<14} {'j (influencer)':<14} {'A[i,j]':>10}"
              f"  {'Party_i':<15} {'Party_j':<15} {'Views_i':<12} {'Views_j':<12}"
              f"  {'Age_i':<8} {'Age_j':<8} {'Region_i':<8} {'Region_j':<8}")
    print(header)
    print("-" * len(header))
    for rank, (pid_i, pid_j, val) in enumerate(pair_list[:top_k], 1):
        di = demographics.get(pid_i, {})
        dj = demographics.get(pid_j, {})
        # Shorten region to first word
        ri = di.get("region", "?").split("(")[0].strip()[:7]
        rj = dj.get("region", "?").split("(")[0].strip()[:7]
        print(f"{rank:<5} {pid_i:<14} {pid_j:<14} {val:>10.4f}"
              f"  {di.get('party','?'):<15} {dj.get('party','?'):<15}"
              f"  {di.get('views','?'):<12} {dj.get('views','?'):<12}"
              f"  {di.get('age','?'):<8} {dj.get('age','?'):<8}"
              f"  {ri:<8} {rj:<8}")

print_pair_table(pos_pairs, "Strongest POSITIVE (reinforcing) influence", top_k=10)
print_pair_table(neg_pairs, "Strongest NEGATIVE (opposing) influence", top_k=10)

# ============================================================
# Analysis 2: Demographic Similarity — Positive vs Negative
# ============================================================
print("\n" + "=" * 70)
print("ANALYSIS 2: DEMOGRAPHIC SIMILARITY — POSITIVE vs NEGATIVE PAIRS")
print("=" * 70)

demo_keys = ["party", "views", "age", "region", "sex", "religion", "income"]

def compute_similarity_rates(pair_list):
    rates = {}
    for key in demo_keys:
        matches = [same_demo(a, b, key) for a, b, _ in pair_list]
        valid = [m for m in matches if m is not None]
        if valid:
            rates[key] = sum(valid) / len(valid)
        else:
            rates[key] = float("nan")
    return rates

pos_rates = compute_similarity_rates(pos_pairs)
neg_rates = compute_similarity_rates(neg_pairs)

print(f"\n{'Demographic':<15} {'Pos match%':>12} {'Neg match%':>12} {'Diff (P-N)':>12}")
print("-" * 55)
for key in demo_keys:
    p = pos_rates[key]
    ne = neg_rates[key]
    diff = p - ne
    print(f"{key:<15} {p:>11.1%} {ne:>11.1%} {diff:>+11.1%}")

print(f"\nTotal positive pairs: {len(pos_pairs)}, negative pairs: {len(neg_pairs)}")

# ============================================================
# Analysis 3: Point-Biserial Correlation
# ============================================================
print("\n" + "=" * 70)
print("ANALYSIS 3: CORRELATION — DEMOGRAPHIC SIMILARITY vs A_{i,j}")
print("=" * 70)

# Build feature vectors for all off-diagonal masked pairs
features_signed = {key: [] for key in demo_keys}
features_abs = {key: [] for key in demo_keys}
a_vals_signed = []
a_vals_abs = []

for i in range(n):
    for j in range(n):
        if i != j and mask[i, j] == 1:
            val = A[i, j]
            a_vals_signed.append(val)
            a_vals_abs.append(abs(val))
            for key in demo_keys:
                s = same_demo(agents[i], agents[j], key)
                if s is None:
                    s = False  # treat missing as different
                features_signed[key].append(int(s))
                features_abs[key].append(int(s))

a_signed = np.array(a_vals_signed)
a_abs = np.array(a_vals_abs)

corr_results = {}
print(f"\n{'Feature':<15} {'r(signed)':>12} {'p(signed)':>12} {'r(|A|)':>12} {'p(|A|)':>12}")
print("-" * 67)
for key in demo_keys:
    feat = np.array(features_signed[key])
    # Point-biserial = Pearson between binary and continuous
    if feat.std() == 0:
        r_s, p_s, r_a, p_a = 0, 1, 0, 1
    else:
        r_s, p_s = stats.pointbiserialr(feat, a_signed)
        r_a, p_a = stats.pointbiserialr(feat, a_abs)
    corr_results[key] = {"r_signed": r_s, "p_signed": p_s, "r_abs": r_a, "p_abs": p_a}
    sig_s = "*" if p_s < 0.05 else ""
    sig_a = "*" if p_a < 0.05 else ""
    print(f"{key:<15} {r_s:>+11.4f}{sig_s} {p_s:>11.4f} {r_a:>+11.4f}{sig_a} {p_a:>11.4f}")

print("\n* = p < 0.05")

# ============================================================
# Plot: Bar chart of correlation coefficients
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

labels = [k.replace("_", " ").title() for k in demo_keys]
r_signed_vals = [corr_results[k]["r_signed"] for k in demo_keys]
r_abs_vals = [corr_results[k]["r_abs"] for k in demo_keys]
p_signed_vals = [corr_results[k]["p_signed"] for k in demo_keys]
p_abs_vals = [corr_results[k]["p_abs"] for k in demo_keys]

colors_signed = ["#2196F3" if p < 0.05 else "#BBDEFB" for p in p_signed_vals]
colors_abs = ["#4CAF50" if p < 0.05 else "#C8E6C9" for p in p_abs_vals]

x = np.arange(len(demo_keys))

# Panel 1: Signed A_{i,j}
ax = axes[0]
bars = ax.bar(x, r_signed_vals, color=colors_signed, edgecolor="black", linewidth=0.5)
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=10)
ax.set_ylabel("Point-Biserial r", fontsize=12)
ax.set_title("Correlation with Signed A[i,j]", fontsize=13, fontweight="bold")
ax.axhline(0, color="black", linewidth=0.8)
ax.grid(axis="y", alpha=0.3)
# Add value labels
for bar, val in zip(bars, r_signed_vals):
    ypos = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, ypos + 0.005 * np.sign(ypos),
            f"{val:+.3f}", ha="center", va="bottom" if ypos >= 0 else "top", fontsize=8)

# Panel 2: |A_{i,j}|
ax = axes[1]
bars = ax.bar(x, r_abs_vals, color=colors_abs, edgecolor="black", linewidth=0.5)
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=10)
ax.set_ylabel("Point-Biserial r", fontsize=12)
ax.set_title("Correlation with |A[i,j]|", fontsize=13, fontweight="bold")
ax.axhline(0, color="black", linewidth=0.8)
ax.grid(axis="y", alpha=0.3)
for bar, val in zip(bars, r_abs_vals):
    ypos = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, ypos + 0.005 * np.sign(ypos),
            f"{val:+.3f}", ha="center", va="bottom" if ypos >= 0 else "top", fontsize=8)

# Legend for significance
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor="#2196F3", edgecolor="black", label="p < 0.05"),
                   Patch(facecolor="#BBDEFB", edgecolor="black", label="p >= 0.05")]
axes[0].legend(handles=legend_elements, fontsize=9, loc="best")
legend_elements2 = [Patch(facecolor="#4CAF50", edgecolor="black", label="p < 0.05"),
                    Patch(facecolor="#C8E6C9", edgecolor="black", label="p >= 0.05")]
axes[1].legend(handles=legend_elements2, fontsize=9, loc="best")

fig.suptitle("Demographic Similarity vs Influence Weight (Sparse Ring)",
             fontsize=14, fontweight="bold")
plt.tight_layout()

out_fig = OUT_DIR / "influence_demographic_analysis.png"
plt.savefig(out_fig, dpi=150, bbox_inches="tight")
plt.close()
print(f"\nSaved figure to {out_fig}")
print("\nDone.")
