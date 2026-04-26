"""
plot_accuracy_vs_reasoning.py — Combined figure: accuracy vs reasoning tokens.

Usage (from Digital-Twin-Simulation/):
    python evaluation/plot_accuracy_vs_reasoning.py
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from pathlib import Path

# ── Data ──────────────────────────────────────────────────────────────────────
REASONING_ORDER = ["none", "low", "medium", "high"]
SETTING_ORDER = [
    "skill_v1", "skill_v2", "skill_v3",
    "raw",
    "raw_start_v1", "raw_start_v2", "raw_start_v3",
    "skill_v1_raw_end", "skill_v2_raw_end", "skill_v3_raw_end",
]

COLORS = {
    "none":   "#4e79a7",
    "low":    "#f28e2b",
    "medium": "#e15759",
    "high":   "#76b7b2",
    "avg":    "#555555",
}

SUITE_META = {
    "nano_temp0": {
        "acc_csv":    "evaluation/nano_temp0_20p_results.csv",
        "tok_csv":    "evaluation/nano_temp0_reasoning_tokens.csv",
        "label":      "gpt-5.4-nano  (temp=0)",
    },
    "mini_temp0": {
        "acc_csv":    "evaluation/mini_temp0_20p_results.csv",
        "tok_csv":    "evaluation/mini_temp0_reasoning_tokens.csv",
        "label":      "gpt-5.4-mini  (temp=0)",
    },
}


def load_suite(meta):
    acc = pd.read_csv(meta["acc_csv"])
    tok = pd.read_csv(meta["tok_csv"])

    acc_piv = acc.pivot(index="setting", columns="reasoning", values="LLM_Accuracy")
    acc_piv = acc_piv.reindex(index=SETTING_ORDER, columns=REASONING_ORDER)
    acc_piv["avg"] = acc_piv[REASONING_ORDER].mean(axis=1).round(3)

    tok_piv = tok.groupby(["setting", "reasoning"])["reasoning_tokens"].mean().unstack("reasoning")
    tok_piv = tok_piv.reindex(index=SETTING_ORDER, columns=REASONING_ORDER)

    return acc_piv, tok_piv


def draw_heatmap(ax, data, cols, title, fmt, cmap, vmin, vmax, col_labels=None):
    vals = data[cols].values.astype(float)
    im = ax.imshow(vals, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    labels = col_labels if col_labels else cols
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_yticks(range(len(SETTING_ORDER)))
    ax.set_yticklabels(SETTING_ORDER, fontsize=8)
    ax.set_title(title, fontsize=10, fontweight="bold", pad=6)
    for i in range(len(SETTING_ORDER)):
        for j, col in enumerate(cols):
            val = vals[i, j]
            if np.isnan(val):
                ax.text(j, i, "—", ha="center", va="center", fontsize=8, color="gray")
            else:
                txt = fmt.format(val)
                cell_frac = (val - vmin) / (vmax - vmin) if vmax > vmin else 0
                color = "white" if cell_frac > 0.6 else "black"
                ax.text(j, i, txt, ha="center", va="center", fontsize=7.5, color=color)
    return im


# ── Main ──────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 11))
fig.suptitle("Accuracy vs Reasoning Tokens — nano_temp0 & mini_temp0\n(20 personas, temp=0)",
             fontsize=13, fontweight="bold", y=0.98)

gs = gridspec.GridSpec(2, 3, figure=fig,
                       left=0.07, right=0.97, top=0.91, bottom=0.06,
                       wspace=0.50, hspace=0.38)

suite_keys = ["nano_temp0", "mini_temp0"]

for row, suite in enumerate(suite_keys):
    meta = SUITE_META[suite]
    acc_piv, tok_piv = load_suite(meta)

    # ── Col 0: Accuracy heatmap (none/low/medium/high + avg) ──────────────
    ax_acc = fig.add_subplot(gs[row, 0])
    acc_cols = REASONING_ORDER + ["avg"]
    im_acc = draw_heatmap(
        ax_acc, acc_piv, acc_cols,
        f"{meta['label']}\nLLM Accuracy",
        "{:.3f}", "YlGn", vmin=0.63, vmax=0.76,
        col_labels=REASONING_ORDER + ["avg"],
    )
    # Separate avg column with a vertical line
    ax_acc.axvline(3.5, color="white", lw=2)
    plt.colorbar(im_acc, ax=ax_acc, fraction=0.046, pad=0.04)

    # ── Col 1: Reasoning token heatmap ────────────────────────────────────
    ax_tok = fig.add_subplot(gs[row, 1])
    tok_k = tok_piv / 1000.0
    vmax_tok = 16 if suite == "mini_temp0" else 7
    im_tok = ax_tok.imshow(tok_k[REASONING_ORDER].values.astype(float),
                           aspect="auto", cmap="Blues", vmin=0, vmax=vmax_tok)
    ax_tok.set_xticks(range(len(REASONING_ORDER)))
    ax_tok.set_xticklabels(REASONING_ORDER, fontsize=9)
    ax_tok.set_yticks(range(len(SETTING_ORDER)))
    ax_tok.set_yticklabels(SETTING_ORDER, fontsize=8)
    ax_tok.set_title(f"{meta['label']}\nAvg Reasoning Tokens (K)",
                     fontsize=10, fontweight="bold", pad=6)
    for i in range(len(SETTING_ORDER)):
        for j in range(len(REASONING_ORDER)):
            val = tok_k[REASONING_ORDER].values[i, j]
            if np.isnan(val):
                ax_tok.text(j, i, "—", ha="center", va="center", fontsize=8, color="gray")
            else:
                txt = f"{val:.1f}K" if val >= 0.1 else "0"
                cell_frac = val / vmax_tok
                ax_tok.text(j, i, txt, ha="center", va="center", fontsize=7.5,
                            color="white" if cell_frac > 0.6 else "black")
    plt.colorbar(im_tok, ax=ax_tok, fraction=0.046, pad=0.04)

    # ── Col 2: Scatter — avg accuracy vs avg reasoning tokens ─────────────
    ax_sc = fig.add_subplot(gs[row, 2])
    for reasoning in REASONING_ORDER:
        acc_vals  = acc_piv[reasoning].dropna()
        tok_vals  = tok_piv[reasoning].reindex(acc_vals.index).fillna(0) / 1000.0
        ax_sc.scatter(tok_vals, acc_vals,
                      color=COLORS[reasoning], label=reasoning,
                      s=45, alpha=0.85, zorder=3)

    # Avg accuracy per setting (horizontal summary)
    avg_acc = acc_piv["avg"].dropna()
    ax_sc.axhline(avg_acc.mean(), color=COLORS["avg"], lw=1.5, ls="-.",
                  label=f"suite avg ({avg_acc.mean():.3f})")
    ax_sc.axhline(0.831, color="green", lw=1, ls="--", label="Human ceiling")
    ax_sc.set_xlabel("Avg Reasoning Tokens (K)", fontsize=9)
    ax_sc.set_ylabel("LLM Accuracy", fontsize=9)
    ax_sc.set_title(f"{meta['label']}\nAccuracy vs Reasoning Tokens",
                    fontsize=10, fontweight="bold", pad=6)
    ax_sc.set_ylim(0.62, 0.78)
    ax_sc.legend(fontsize=7.5, loc="lower right")
    ax_sc.grid(True, alpha=0.3)
    ax_sc.tick_params(labelsize=8)

out = Path("evaluation/accuracy_vs_reasoning_tokens.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved → {out}")
plt.show()
