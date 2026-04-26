"""
plot_pricing_per_question.py — Heatmap: 40 product preference questions × 40 configs per model.

Usage (from Digital-Twin-Simulation/):
    python evaluation/plot_pricing_per_question.py
"""

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from pathlib import Path

# ── Question labels ───────────────────────────────────────────────────────────
lines = open("text_simulation/text_questions/pid_1.txt").readlines()
q_labels = {}   # qn -> "Category | $price"
q_full   = {}   # qn -> full short description
for line in lines:
    m = re.match(r"Please consider the following product category: (.+)", line.strip())
    if m:
        text  = line.strip()
        cat   = re.search(r"product category: (.+?)\. Suppose", text)
        prod  = re.search(r"in that category: (.+?)\. The product", text)
        price = re.search(r"priced at: (\S+)\.", text)
        n     = len(q_labels) + 1
        cat_s   = cat.group(1)[:22]   if cat   else "?"
        prod_s  = prod.group(1)[:30]  if prod  else "?"
        price_s = price.group(1)      if price else "?"
        q_labels[n] = f"Q{n:02d} {cat_s} ({price_s})"
        q_full[n]   = f"Q{n:02d} | {prod_s} | {price_s}"

# ── Load precomputed CSV ──────────────────────────────────────────────────────
df = pd.read_csv("evaluation/pricing_per_question_per_config.csv")

SETTINGS   = ["skill_v1","skill_v2","skill_v3","raw",
              "raw_start_v1","raw_start_v2","raw_start_v3",
              "skill_v1_raw_end","skill_v2_raw_end","skill_v3_raw_end"]
REASONINGS = ["none","low","medium","high"]
Q_COLS     = [f"Q{i:02d}" for i in range(1, 41)]

# ── Build matrix: rows=questions (40), cols=configs (40) per suite ────────────
def build_matrix(suite):
    sub = df[df["suite"] == suite].copy()
    # enforce order
    sub["s_ord"] = sub["setting"].map({s: i for i, s in enumerate(SETTINGS)})
    sub["r_ord"] = sub["reasoning"].map({r: i for i, r in enumerate(REASONINGS)})
    sub = sub.sort_values(["s_ord","r_ord"])
    mat  = sub[Q_COLS].values.T   # shape (40 questions, 40 configs)
    xlabels = [f"{r['setting']}\n{r['reasoning']}" for _, r in sub.iterrows()]
    return mat, xlabels

# ── Plot ───────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(28, 18))
fig.suptitle("Product Preference Accuracy per Question × Config\n(LLM vs wave1-3, 20 personas each)",
             fontsize=13, fontweight="bold", y=0.995)

gs = gridspec.GridSpec(1, 2, figure=fig, left=0.12, right=0.97,
                       top=0.96, bottom=0.12, wspace=0.06)

suite_titles = {
    "nano_temp0": "gpt-5.4-nano (temp=0)",
    "mini_temp0": "gpt-5.4-mini (temp=0)",
}

# colour blocks for settings
setting_colors = [
    "#4e79a7","#4e79a7","#4e79a7",   # skill_v1/2/3
    "#e15759",                        # raw
    "#f28e2b","#f28e2b","#f28e2b",   # raw_start_v1/2/3
    "#76b7b2","#76b7b2","#76b7b2",   # skill_v1/2/3_raw_end
]

for col_idx, suite in enumerate(["nano_temp0", "mini_temp0"]):
    mat, xlabels = build_matrix(suite)

    ax = fig.add_subplot(gs[0, col_idx])
    im = ax.imshow(mat, aspect="auto", cmap="RdYlGn", vmin=0.30, vmax=0.90)

    # x-axis: config labels (setting\nreasoning)
    ax.set_xticks(range(40))
    ax.set_xticklabels(xlabels, fontsize=5.5, rotation=90, ha="center")

    # y-axis: question labels
    ax.set_yticks(range(40))
    ax.set_yticklabels([q_labels[i+1] for i in range(40)], fontsize=7)

    ax.set_title(f"{suite_titles[suite]}", fontsize=11, fontweight="bold", pad=8)

    # annotate cells
    for i in range(40):
        for j in range(40):
            v = mat[i, j]
            if not np.isnan(v):
                cell_frac = (v - 0.30) / 0.60
                color = "white" if cell_frac > 0.65 else "black"
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=4.5, color=color)

    # draw vertical separators every 4 configs (between settings)
    for sep in range(4, 40, 4):
        ax.axvline(sep - 0.5, color="white", lw=1.5)

    # coloured top strip for setting groups
    for j in range(40):
        s_idx = j // 4
        ax.add_patch(plt.Rectangle(
            (j - 0.5, -1.6), 1.0, 0.9,
            color=setting_colors[s_idx], clip_on=False, transform=ax.transData
        ))

    # setting name labels above strip
    s_names_short = ["skv1","skv2","skv3","raw","rsv1","rsv2","rsv3","sv1re","sv2re","sv3re"]
    for s_idx, sname in enumerate(s_names_short):
        ax.text(s_idx * 4 + 1.5, -1.15, sname,
                ha="center", va="center", fontsize=5.5, color="white",
                fontweight="bold", transform=ax.transData)

    plt.colorbar(im, ax=ax, fraction=0.02, pad=0.01, label="Accuracy")

# Legend
legend_elements = [
    Patch(color="#4e79a7", label="skill_v1/2/3"),
    Patch(color="#e15759", label="raw"),
    Patch(color="#f28e2b", label="raw_start_v1/2/3"),
    Patch(color="#76b7b2", label="skill_v1/2/3_raw_end"),
]
fig.legend(handles=legend_elements, loc="lower center", ncol=4,
           fontsize=8, frameon=True, bbox_to_anchor=(0.54, 0.005))

out = Path("evaluation/pricing_per_question_heatmap.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved → {out}")
plt.show()
