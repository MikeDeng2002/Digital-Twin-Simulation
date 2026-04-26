"""
plot_by_question_type.py — Visualize accuracy breakdown by question type.

Usage (from Digital-Twin-Simulation/):
    python evaluation/plot_by_question_type.py
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from pathlib import Path

SETTING_ORDER = [
    "skill_v1", "skill_v2", "skill_v3",
    "raw",
    "raw_start_v1", "raw_start_v2", "raw_start_v3",
    "skill_v1_raw_end", "skill_v2_raw_end", "skill_v3_raw_end",
]
REASONING_ORDER = ["none", "low", "medium", "high"]

SUITE_META = {
    "nano_temp0": {"csv": "evaluation/nano_temp0_by_question_type.csv", "label": "gpt-5.4-nano (temp=0)"},
    "mini_temp0": {"csv": "evaluation/mini_temp0_by_question_type.csv", "label": "gpt-5.4-mini (temp=0)"},
}


def load_pivot(csv_path, metric):
    df = pd.read_csv(csv_path)
    piv = df.pivot_table(index="setting", columns="reasoning", values=metric, aggfunc="first")
    piv = piv.reindex(index=SETTING_ORDER, columns=REASONING_ORDER)
    piv["avg"] = piv[REASONING_ORDER].mean(axis=1).round(3)
    return piv


def draw_heatmap(ax, data, cols, title, vmin, vmax, cmap, col_labels=None):
    vals = data[cols].values.astype(float)
    im = ax.imshow(vals, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    labels = col_labels if col_labels else cols
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_yticks(range(len(SETTING_ORDER)))
    ax.set_yticklabels(SETTING_ORDER, fontsize=8)
    ax.set_title(title, fontsize=9.5, fontweight="bold", pad=5)
    for i in range(len(SETTING_ORDER)):
        for j, col in enumerate(cols):
            val = vals[i, j]
            if np.isnan(val):
                ax.text(j, i, "—", ha="center", va="center", fontsize=8, color="gray")
            else:
                cell_frac = (val - vmin) / (vmax - vmin) if vmax > vmin else 0
                color = "white" if cell_frac > 0.6 else "black"
                ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=7, color=color)
    return im


fig = plt.figure(figsize=(20, 12))
fig.suptitle("Accuracy by Question Type — Cognitive Biases vs Product Preferences\n(20 personas, temp=0, LLM vs wave1-3)",
             fontsize=13, fontweight="bold", y=0.99)

gs = gridspec.GridSpec(2, 3, figure=fig,
                       left=0.07, right=0.97, top=0.92, bottom=0.05,
                       wspace=0.45, hspace=0.40)

acc_cols = REASONING_ORDER + ["avg"]
col_labels = REASONING_ORDER + ["avg"]

for row, (suite, meta) in enumerate(SUITE_META.items()):
    bias_piv    = load_pivot(meta["csv"], "bias_acc")
    pricing_piv = load_pivot(meta["csv"], "pricing_acc")
    overall_piv = load_pivot(meta["csv"], "overall_acc")

    # Col 0: Cognitive Biases heatmap
    ax0 = fig.add_subplot(gs[row, 0])
    im0 = draw_heatmap(ax0, bias_piv, acc_cols,
                       f"{meta['label']}\nCognitive Biases Accuracy",
                       vmin=0.63, vmax=0.79, cmap="YlOrRd",
                       col_labels=col_labels)
    ax0.axvline(3.5, color="white", lw=2)
    plt.colorbar(im0, ax=ax0, fraction=0.046, pad=0.04)

    # Col 1: Product Preferences heatmap
    ax1 = fig.add_subplot(gs[row, 1])
    im1 = draw_heatmap(ax1, pricing_piv, acc_cols,
                       f"{meta['label']}\nProduct Preferences Accuracy",
                       vmin=0.53, vmax=0.69, cmap="YlGnBu",
                       col_labels=col_labels)
    ax1.axvline(3.5, color="white", lw=2)
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    # Col 2: Scatter — bias vs pricing per (setting, reasoning)
    ax2 = fig.add_subplot(gs[row, 2])
    colors = {"none": "#4e79a7", "low": "#f28e2b", "medium": "#e15759", "high": "#76b7b2"}
    for reasoning in REASONING_ORDER:
        bias_vals    = bias_piv[reasoning].dropna()
        pricing_vals = pricing_piv[reasoning].reindex(bias_vals.index)
        ax2.scatter(bias_vals, pricing_vals,
                    color=colors[reasoning], label=reasoning,
                    s=50, alpha=0.85, zorder=3)

    # Reference lines: human ceiling and avg
    ax2.axvline(0.831, color="green", lw=1, ls="--", alpha=0.7, label="Human (bias ceiling)")
    bias_avg = bias_piv["avg"].dropna().mean()
    pricing_avg = pricing_piv["avg"].dropna().mean()
    ax2.axvline(bias_avg, color="#4e79a7", lw=1.2, ls="-.", alpha=0.6, label=f"Bias avg ({bias_avg:.3f})")
    ax2.axhline(pricing_avg, color="#f28e2b", lw=1.2, ls="-.", alpha=0.6, label=f"Pricing avg ({pricing_avg:.3f})")

    ax2.set_xlabel("Cognitive Biases Accuracy", fontsize=9)
    ax2.set_ylabel("Product Preferences Accuracy", fontsize=9)
    ax2.set_title(f"{meta['label']}\nBias vs Pricing Accuracy", fontsize=9.5, fontweight="bold", pad=5)
    ax2.set_xlim(0.62, 0.82)
    ax2.set_ylim(0.52, 0.72)
    ax2.legend(fontsize=7, loc="upper right")
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(labelsize=8)

out = Path("evaluation/accuracy_by_question_type.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved → {out}")
plt.show()
