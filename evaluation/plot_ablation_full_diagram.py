"""
plot_ablation_full_diagram.py — Full diagram: all reasoning levels × versions × conditions.

Two layouts:
  - Fig A (4×2): one row per reasoning level, pricing + bias side by side
  - Fig B (2×4): one row per metric, one column per condition, lines per version

Usage (from Digital-Twin-Simulation/):
    python evaluation/plot_ablation_full_diagram.py
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
VERSION_DATA = {
    "v1_direct":   "evaluation/nano_v2_ablation_temp0_by_question_type.csv",
    "v2_inferred": "evaluation/nano_v2_ablation_v2_temp0_by_question_type.csv",
    "v3_maximum":  "evaluation/nano_v2_ablation_v3_temp0_by_question_type.csv",
}
RAW_CSV = "evaluation/nano_temp0_by_question_type.csv"

CONDITIONS    = ["bg", "bg_dp", "bg_ep", "bg_dp_ep"]
COND_LABELS   = ["BG only", "BG+DP", "BG+EP", "BG+DP+EP"]
REASONINGS    = ["none", "low", "medium", "high"]
REASON_LABELS = ["None", "Low", "Medium", "High"]

VERSION_COLORS = {
    "v1_direct":   "#4878d0",
    "v2_inferred": "#ee854a",
    "v3_maximum":  "#6acc65",
}
VERSION_DISPLAY = {
    "v1_direct":   "v1_direct",
    "v2_inferred": "v2_inferred",
    "v3_maximum":  "v3_maximum",
}

# Load data
dfs = {v: pd.read_csv(p) for v, p in VERSION_DATA.items()}

# Raw baseline
try:
    raw_df = pd.read_csv(RAW_CSV)
    raw_row = raw_df[raw_df["setting"] == "raw"]
    raw_pricing = raw_row["pricing_acc"].mean() if len(raw_row) else None
    raw_bias    = raw_row["bias_acc"].mean()    if len(raw_row) else None
except Exception:
    raw_pricing = raw_bias = None


def get_val(df, condition, reasoning, metric):
    row = df[(df["setting"] == condition) & (df["reasoning"] == reasoning)]
    if len(row) == 0:
        return np.nan
    return float(row[metric].iloc[0])


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE A: 4 rows (reasoning) × 2 cols (metric) — grouped bar per condition
# ─────────────────────────────────────────────────────────────────────────────
fig_a, axes_a = plt.subplots(4, 2, figsize=(16, 22))
fig_a.suptitle(
    "Full Ablation Results — All Reasoning Levels × Skill Versions\n"
    "(20 personas, gpt-5.4-nano, temp=0, MAD accuracy vs wave1-3)",
    fontsize=15, fontweight="bold", y=0.995
)

x = np.arange(len(CONDITIONS))
n_versions = len(dfs)
width = 0.24
offsets = np.array([-1, 0, 1]) * width

METRIC_CFG = [
    ("pricing_acc", "Product Preferences (Pricing) Accuracy", 0.38, 0.56, "#4878d0", raw_pricing),
    ("bias_acc",    "Cognitive Bias Accuracy",                 0.65, 0.80, "#e15759", raw_bias),
]

for row_i, reasoning in enumerate(REASONINGS):
    for col_i, (metric, metric_title, vmin, vmax, _, raw_val) in enumerate(METRIC_CFG):
        ax = axes_a[row_i, col_i]

        for vi, (version, df) in enumerate(dfs.items()):
            vals = [get_val(df, c, reasoning, metric) for c in CONDITIONS]
            bars = ax.bar(x + offsets[vi], vals, width,
                          label=VERSION_DISPLAY[version],
                          color=VERSION_COLORS[version],
                          alpha=0.88, edgecolor="white", linewidth=0.5)
            for bar, v in zip(bars, vals):
                if not np.isnan(v):
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 0.001,
                            f"{v:.3f}", ha="center", va="bottom",
                            fontsize=7.5, fontweight="bold")

        if raw_val is not None:
            ax.axhline(raw_val, color="black", lw=1.8, ls="--", alpha=0.75,
                       label=f"Raw data ({raw_val:.3f})", zorder=5)

        ax.set_xticks(x)
        ax.set_xticklabels(COND_LABELS, fontsize=11)
        ax.set_ylim(vmin, vmax)
        ax.set_ylabel("MAD Accuracy", fontsize=10)
        ax.set_title(f"Reasoning = {reasoning.upper()}  |  {metric_title}",
                     fontsize=11, fontweight="bold", pad=6)
        ax.tick_params(axis="y", labelsize=10)
        ax.grid(axis="y", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        if row_i == 0:
            ax.legend(fontsize=9.5, loc="upper left",
                      framealpha=0.85, edgecolor="gray")

fig_a.tight_layout(rect=[0, 0, 1, 0.995])
out_a = Path("evaluation/ablation_full_diagram_by_reasoning.png")
fig_a.savefig(out_a, dpi=150, bbox_inches="tight")
print(f"Saved → {out_a}")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE B: 2 rows (metric) × 4 cols (condition) — line plots per version
# ─────────────────────────────────────────────────────────────────────────────
fig_b, axes_b = plt.subplots(2, 4, figsize=(20, 10))
fig_b.suptitle(
    "Accuracy vs Reasoning Level — All Versions × Conditions\n"
    "(20 personas, gpt-5.4-nano, temp=0)",
    fontsize=14, fontweight="bold", y=1.01
)

METRIC_LINE_CFG = [
    ("pricing_acc", "Product Preferences (Pricing)", 0.40, 0.56, raw_pricing),
    ("bias_acc",    "Cognitive Bias",                 0.66, 0.80, raw_bias),
]

for row_i, (metric, metric_title, vmin, vmax, raw_val) in enumerate(METRIC_LINE_CFG):
    for col_i, (condition, cond_label) in enumerate(zip(CONDITIONS, COND_LABELS)):
        ax = axes_b[row_i, col_i]

        for version, df in dfs.items():
            vals = [get_val(df, condition, r, metric) for r in REASONINGS]
            ax.plot(REASON_LABELS, vals,
                    color=VERSION_COLORS[version], marker="o",
                    lw=2.5, ms=7, label=VERSION_DISPLAY[version], zorder=3)
            # Annotate each point
            for xi, v in enumerate(vals):
                if not np.isnan(v):
                    ax.text(xi, v + 0.002, f"{v:.3f}",
                            ha="center", va="bottom", fontsize=7.5,
                            color=VERSION_COLORS[version], fontweight="bold")

        if raw_val is not None:
            ax.axhline(raw_val, color="black", lw=1.5, ls="--", alpha=0.65,
                       label=f"Raw ({raw_val:.3f})", zorder=2)

        ax.set_ylim(vmin, vmax)
        ax.set_title(f"{cond_label}\n{metric_title}", fontsize=11, fontweight="bold", pad=5)
        ax.set_xlabel("Reasoning Level", fontsize=10)
        if col_i == 0:
            ax.set_ylabel("MAD Accuracy", fontsize=10)
        ax.tick_params(labelsize=9.5)
        ax.grid(True, alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        if row_i == 0 and col_i == 0:
            ax.legend(fontsize=9, loc="lower right", framealpha=0.85)

fig_b.tight_layout()
out_b = Path("evaluation/ablation_full_diagram_line_plots.png")
fig_b.savefig(out_b, dpi=150, bbox_inches="tight")
print(f"Saved → {out_b}")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE C: Summary heatmap — 3 versions, condition × reasoning, both metrics
# ─────────────────────────────────────────────────────────────────────────────
fig_c, axes_c = plt.subplots(2, 3, figsize=(18, 10))
fig_c.suptitle(
    "Heatmap: MAD Accuracy by Condition × Reasoning — Pricing & Cognitive Bias\n"
    "(20 personas, gpt-5.4-nano, temp=0)",
    fontsize=14, fontweight="bold"
)

HEATMAP_CFG = [
    ("pricing_acc", "Pricing Accuracy",       0.40, 0.56, "Blues"),
    ("bias_acc",    "Cognitive Bias Accuracy", 0.66, 0.80, "Reds"),
]

for col_i, (version, df) in enumerate(dfs.items()):
    for row_i, (metric, metric_label, vmin, vmax, cmap) in enumerate(HEATMAP_CFG):
        ax = axes_c[row_i, col_i]

        vals = np.array([
            [get_val(df, c, r, metric) for r in REASONINGS]
            for c in CONDITIONS
        ])

        im = ax.imshow(vals, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_xticks(range(len(REASONINGS)))
        ax.set_xticklabels(REASON_LABELS, fontsize=11)
        ax.set_yticks(range(len(CONDITIONS)))
        ax.set_yticklabels(COND_LABELS, fontsize=11)
        ax.set_title(f"{VERSION_DISPLAY[version]}\n{metric_label}",
                     fontsize=11, fontweight="bold", pad=6)

        # Avg column
        for i, c in enumerate(CONDITIONS):
            row_vals = vals[i]
            avg = np.nanmean(row_vals)
            for j, v in enumerate(row_vals):
                if not np.isnan(v):
                    cell_frac = (v - vmin) / (vmax - vmin + 1e-9)
                    txt_color = "white" if cell_frac > 0.55 else "black"
                    ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                            fontsize=10, color=txt_color, fontweight="bold")

        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=9)

fig_c.tight_layout()
out_c = Path("evaluation/ablation_full_diagram_heatmap.png")
fig_c.savefig(out_c, dpi=150, bbox_inches="tight")
print(f"Saved → {out_c}")

plt.close("all")
print("\nAll figures saved.")

# ── Console summary ───────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("FULL RESULTS TABLE — Pricing Accuracy")
print("=" * 80)
for version, df in dfs.items():
    print(f"\n{VERSION_DISPLAY[version]}:")
    piv = df.pivot_table(index="setting", columns="reasoning",
                         values="pricing_acc", aggfunc="first")
    piv = piv.reindex(index=CONDITIONS, columns=REASONINGS)
    piv.index = COND_LABELS
    piv.columns = REASON_LABELS
    piv["AVG"] = piv.mean(axis=1).round(3)
    print(piv.to_string())

print("\n" + "=" * 80)
print("FULL RESULTS TABLE — Cognitive Bias Accuracy")
print("=" * 80)
for version, df in dfs.items():
    print(f"\n{VERSION_DISPLAY[version]}:")
    piv = df.pivot_table(index="setting", columns="reasoning",
                         values="bias_acc", aggfunc="first")
    piv = piv.reindex(index=CONDITIONS, columns=REASONINGS)
    piv.index = COND_LABELS
    piv.columns = REASON_LABELS
    piv["AVG"] = piv.mean(axis=1).round(3)
    print(piv.to_string())
