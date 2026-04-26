"""
plot_ablation_version_comparison.py — Compare ablation conditions across v1/v2/v3 skill versions.

Shows pricing and cognitive bias accuracy for all 3 versions × 4 ablation conditions,
broken down by reasoning level.

Usage (from Digital-Twin-Simulation/):
    python evaluation/plot_ablation_version_comparison.py
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from pathlib import Path

# ── Data sources ──────────────────────────────────────────────────────────────
VERSION_DATA = {
    "v1_direct":  "evaluation/nano_v2_ablation_temp0_by_question_type.csv",
    "v2_inferred": "evaluation/nano_v2_ablation_v2_temp0_by_question_type.csv",
    "v3_maximum": "evaluation/nano_v2_ablation_v3_temp0_by_question_type.csv",
}

# Raw baseline: grab from nano_temp0 suite (raw condition)
RAW_CSV = "evaluation/nano_temp0_by_question_type.csv"

CONDITION_ORDER = ["bg", "bg_dp", "bg_ep", "bg_dp_ep"]
CONDITION_LABELS = ["BG only", "BG+DP", "BG+EP", "BG+DP+EP"]
REASONING_ORDER  = ["none", "low", "medium", "high"]

VERSION_COLORS = {
    "v1_direct":   "#4e79a7",
    "v2_inferred": "#e15759",
    "v3_maximum":  "#59a14f",
}
REASONING_MARKERS = {"none": "o", "low": "s", "medium": "^", "high": "D"}
REASONING_LINESTYLES = {"none": ":", "low": "--", "medium": "-.", "high": "-"}


def load_data(csv_path, conditions=None):
    df = pd.read_csv(csv_path)
    if conditions:
        df = df[df["setting"].isin(conditions)]
    return df


def get_avg_by_condition(df, metric):
    """Average metric across reasoning levels per condition."""
    return df.groupby("setting")[metric].mean().reindex(CONDITION_ORDER)


def get_by_condition_reasoning(df, metric):
    """Return pivot: condition × reasoning."""
    piv = df.pivot_table(index="setting", columns="reasoning", values=metric, aggfunc="first")
    return piv.reindex(index=CONDITION_ORDER, columns=REASONING_ORDER)


# ── Load all version data ─────────────────────────────────────────────────────
version_dfs = {v: load_data(p, CONDITION_ORDER) for v, p in VERSION_DATA.items()}

# Raw baseline (averaged across all reasoning × settings)
try:
    raw_df = load_data(RAW_CSV)
    raw_df_raw = raw_df[raw_df["setting"] == "raw"]
    raw_pricing_avg = raw_df_raw["pricing_acc"].mean() if len(raw_df_raw) > 0 else None
    raw_bias_avg    = raw_df_raw["bias_acc"].mean()    if len(raw_df_raw) > 0 else None
except Exception:
    raw_pricing_avg = None
    raw_bias_avg = None

# ── Figure 1: Grouped bar chart — avg across reasoning levels ────────────────
fig1, axes1 = plt.subplots(1, 2, figsize=(14, 6))
fig1.suptitle(
    "Ablation Comparison Across Skill Versions — Averaged Over Reasoning Levels\n(20 personas, gpt-5.4-nano, temp=0)",
    fontsize=13, fontweight="bold"
)

x = np.arange(len(CONDITION_ORDER))
width = 0.25

for ax, metric, title, ylabel, vmin, vmax, raw_val in [
    (axes1[0], "pricing_acc", "Product Preferences (Pricing)", "MAD Accuracy", 0.40, 0.55, raw_pricing_avg),
    (axes1[1], "bias_acc",    "Cognitive Biases",              "MAD Accuracy", 0.65, 0.80, raw_bias_avg),
]:
    for i, (version, df) in enumerate(version_dfs.items()):
        avgs = get_avg_by_condition(df, metric)
        offset = (i - 1) * width
        bars = ax.bar(x + offset, avgs.values, width,
                      label=version, color=VERSION_COLORS[version],
                      alpha=0.85, edgecolor="white")
        # value labels
        for bar in bars:
            h = bar.get_height()
            if not np.isnan(h):
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.002,
                        f"{h:.3f}", ha="center", va="bottom", fontsize=7.5)

    if raw_val is not None:
        ax.axhline(raw_val, color="black", lw=1.5, ls="--", alpha=0.7, label=f"Raw data ({raw_val:.3f})")

    ax.set_xticks(x)
    ax.set_xticklabels(CONDITION_LABELS, fontsize=10)
    ax.set_ylim(vmin, vmax)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.legend(fontsize=8.5)
    ax.grid(axis="y", alpha=0.3)
    ax.tick_params(labelsize=9)

fig1.tight_layout()
out1 = Path("evaluation/ablation_version_comparison_avg.png")
fig1.savefig(out1, dpi=150, bbox_inches="tight")
print(f"Saved → {out1}")


# ── Figure 2: By reasoning level (4×2 grid — 4 reasoning levels, 2 metrics) ──
fig2, axes2 = plt.subplots(4, 2, figsize=(14, 20))
fig2.suptitle(
    "Ablation by Reasoning Level — Pricing vs Cognitive Bias Accuracy\n(20 personas, gpt-5.4-nano, temp=0)",
    fontsize=13, fontweight="bold"
)

for row, reasoning in enumerate(REASONING_ORDER):
    for col, (metric, title, vmin, vmax, raw_val) in enumerate([
        ("pricing_acc", "Product Preferences", 0.38, 0.56, raw_pricing_avg),
        ("bias_acc",    "Cognitive Biases",    0.63, 0.80, raw_bias_avg),
    ]):
        ax = axes2[row, col]
        x = np.arange(len(CONDITION_ORDER))
        width = 0.25

        for i, (version, df) in enumerate(version_dfs.items()):
            sub = df[df["reasoning"] == reasoning]
            vals = sub.set_index("setting")[metric].reindex(CONDITION_ORDER)
            offset = (i - 1) * width
            bars = ax.bar(x + offset, vals.values, width,
                          label=version, color=VERSION_COLORS[version],
                          alpha=0.85, edgecolor="white")
            for bar in bars:
                h = bar.get_height()
                if not np.isnan(h):
                    ax.text(bar.get_x() + bar.get_width() / 2, h + 0.001,
                            f"{h:.3f}", ha="center", va="bottom", fontsize=6.5)

        if raw_val is not None:
            ax.axhline(raw_val, color="black", lw=1.2, ls="--", alpha=0.6,
                       label=f"Raw ({raw_val:.3f})")

        ax.set_xticks(x)
        ax.set_xticklabels(CONDITION_LABELS, fontsize=9)
        ax.set_ylim(vmin, vmax)
        ax.set_ylabel("MAD Accuracy", fontsize=9)
        ax.set_title(f"Reasoning={reasoning} | {title}", fontsize=10, fontweight="bold")
        if row == 0 and col == 0:
            ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        ax.tick_params(labelsize=8)

fig2.tight_layout()
out2 = Path("evaluation/ablation_version_comparison_by_reasoning.png")
fig2.savefig(out2, dpi=150, bbox_inches="tight")
print(f"Saved → {out2}")


# ── Figure 3: Line plot — version × condition across reasoning ────────────────
fig3, axes3 = plt.subplots(2, 4, figsize=(22, 9))
fig3.suptitle(
    "Accuracy vs Reasoning Level — All 3 Versions × 4 Ablation Conditions\n(20 personas, gpt-5.4-nano, temp=0)",
    fontsize=13, fontweight="bold"
)

for col, condition in enumerate(CONDITION_ORDER):
    for row, (metric, metric_label, vmin, vmax) in enumerate([
        ("pricing_acc", "Pricing Accuracy",        0.38, 0.56),
        ("bias_acc",    "Cognitive Bias Accuracy",  0.63, 0.80),
    ]):
        ax = axes3[row, col]

        for version, df in version_dfs.items():
            sub = df[df["setting"] == condition]
            sub = sub.set_index("reasoning").reindex(REASONING_ORDER)
            vals = sub[metric].values.astype(float)
            ax.plot(REASONING_ORDER, vals,
                    color=VERSION_COLORS[version], marker="o", lw=2, ms=6,
                    label=version, zorder=3)

        if row == 0 and raw_pricing_avg is not None:
            ax.axhline(raw_pricing_avg, color="black", lw=1.2, ls="--",
                       alpha=0.6, label=f"Raw ({raw_pricing_avg:.3f})")
        if row == 1 and raw_bias_avg is not None:
            ax.axhline(raw_bias_avg, color="black", lw=1.2, ls="--",
                       alpha=0.6, label=f"Raw ({raw_bias_avg:.3f})")

        ax.set_ylim(vmin, vmax)
        ax.set_title(f"{CONDITION_LABELS[col]}\n{metric_label}", fontsize=9.5, fontweight="bold")
        ax.set_xlabel("Reasoning Level", fontsize=8)
        ax.set_ylabel(metric_label, fontsize=8)
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.3)
        if row == 0 and col == 0:
            ax.legend(fontsize=8)

fig3.tight_layout()
out3 = Path("evaluation/ablation_version_line_plots.png")
fig3.savefig(out3, dpi=150, bbox_inches="tight")
print(f"Saved → {out3}")


# ── Figure 4: Summary heatmap — all 3 versions side by side ──────────────────
fig4, axes4 = plt.subplots(2, 3, figsize=(18, 9))
fig4.suptitle(
    "Heatmap Summary: Condition × Reasoning by Version\n(20 personas, gpt-5.4-nano, temp=0)",
    fontsize=13, fontweight="bold"
)

for col, (version, df) in enumerate(version_dfs.items()):
    for row, (metric, metric_label, vmin, vmax, cmap) in enumerate([
        ("pricing_acc", "Pricing Accuracy",       0.40, 0.56, "YlGnBu"),
        ("bias_acc",    "Cog. Bias Accuracy",     0.65, 0.80, "YlOrRd"),
    ]):
        ax = axes4[row, col]
        piv = get_by_condition_reasoning(df, metric)
        vals = piv.values.astype(float)

        im = ax.imshow(vals, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_xticks(range(len(REASONING_ORDER)))
        ax.set_xticklabels(REASONING_ORDER, fontsize=9)
        ax.set_yticks(range(len(CONDITION_ORDER)))
        ax.set_yticklabels(CONDITION_LABELS, fontsize=9)
        ax.set_title(f"{version}\n{metric_label}", fontsize=10, fontweight="bold")

        for i in range(len(CONDITION_ORDER)):
            for j in range(len(REASONING_ORDER)):
                v = vals[i, j]
                if not np.isnan(v):
                    cell_frac = (v - vmin) / (vmax - vmin)
                    color = "white" if cell_frac > 0.6 else "black"
                    ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=9, color=color)

        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

fig4.tight_layout()
out4 = Path("evaluation/ablation_version_heatmap.png")
fig4.savefig(out4, dpi=150, bbox_inches="tight")
print(f"Saved → {out4}")

plt.close("all")
print("\nAll 4 comparison figures saved.")

# ── Print summary table ───────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SUMMARY: Average accuracy across reasoning levels")
print("=" * 70)
for metric, label in [("pricing_acc", "PRICING"), ("bias_acc", "COG. BIAS")]:
    print(f"\n{label}:")
    rows = []
    for version, df in version_dfs.items():
        row = {"version": version}
        for cond in CONDITION_ORDER:
            sub = df[df["setting"] == cond]
            row[cond] = round(sub[metric].mean(), 3)
        rows.append(row)
    summary = pd.DataFrame(rows).set_index("version")
    summary.columns = CONDITION_LABELS
    print(summary.to_string())
