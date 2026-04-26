"""
plot_ablation_fixed_all_versions.py — Full ablation chart: v1/v2/v3 × 3 metrics with raw baseline.
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# ── Data ──────────────────────────────────────────────────────────────────────

conditions   = ['raw\n(baseline)', 'bg', 'bg+dp', 'bg+ep', 'bg+dp+ep']
raw_overall  = 0.691;  raw_bias = 0.748;  raw_pricing = 0.618

data = {
    'V1 (v1_direct)': {
        'overall':  [raw_overall, 0.672, 0.684, 0.706, 0.681],
        'bias':     [raw_bias,    0.746, 0.744, 0.758, 0.710],
        'pricing':  [raw_pricing, 0.578, 0.606, 0.639, 0.642],
    },
    'V2 (v2_inferred)': {
        'overall':  [raw_overall, 0.695, 0.695, 0.704, 0.681],
        'bias':     [raw_bias,    0.750, 0.759, 0.750, 0.711],
        'pricing':  [raw_pricing, 0.624, 0.613, 0.644, 0.641],
    },
    'V3 (v3_maximum)': {
        'overall':  [raw_overall, 0.691, 0.703, 0.706, 0.705],
        'bias':     [raw_bias,    0.755, 0.753, 0.753, 0.755],
        'pricing':  [raw_pricing, 0.608, 0.638, 0.644, 0.639],
    },
}

human_ceiling = 0.831
colors = ['#b0b0b0', '#4e79a7', '#f28e2b', '#e15759', '#59a14f']
metrics = [('overall', 'Overall Accuracy', (0.62, 0.75)),
           ('bias',    'Cognitive Bias Accuracy', (0.68, 0.78)),
           ('pricing', 'Product Preference Accuracy', (0.54, 0.68))]

versions = list(data.keys())

# ── Plot: 3 rows (versions) × 3 cols (metrics) ───────────────────────────────

fig, axes = plt.subplots(3, 3, figsize=(15, 12), sharey=False)
fig.suptitle('Skill Ablation — gpt-5.4-nano High Reasoning, 20 Personas\n(Correct Question Files)',
             fontsize=14, fontweight='bold', y=1.01)

for row, version in enumerate(versions):
    for col, (metric, title, ylim) in enumerate(metrics):
        ax = axes[row][col]
        vals = data[version][metric]
        bars = ax.bar(conditions, vals, color=colors, edgecolor='white', linewidth=0.7, width=0.6)

        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=7.5, fontweight='bold')

        ax.axhline(human_ceiling, color='#d62728', linestyle='--', linewidth=1.0, alpha=0.7)
        ax.axvline(0.5, color='grey', linestyle=':', linewidth=0.7, alpha=0.4)
        ax.set_ylim(ylim)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='y', alpha=0.3, linewidth=0.5)
        ax.tick_params(axis='x', labelsize=8)
        ax.tick_params(axis='y', labelsize=8)

        if row == 0:
            ax.set_title(title, fontsize=11, fontweight='bold', pad=6)
        if col == 0:
            ax.set_ylabel(f'{version}\n\nAccuracy', fontsize=9, fontweight='bold')
        if row == 2 and col == 2:
            ax.text(4.3, human_ceiling + 0.003, f'Human\n({human_ceiling})',
                    fontsize=7, color='#d62728', ha='left')

# Legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor=colors[0], label='raw — raw Wave1-3 responses'),
    Patch(facecolor=colors[1], label='bg — background only'),
    Patch(facecolor=colors[2], label='bg+dp — bg + decision procedure'),
    Patch(facecolor=colors[3], label='bg+ep — bg + evaluation profile'),
    Patch(facecolor=colors[4], label='bg+dp+ep — full (all 3 components)'),
]
fig.legend(handles=legend_elements, loc='lower center', ncol=5, fontsize=9,
           bbox_to_anchor=(0.5, -0.04), frameon=True, edgecolor='lightgrey')

plt.tight_layout()
out = 'experiments/ablation_fixed_all_versions_chart.png'
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f'Saved → {out}')
plt.close()
