"""
plot_v3_ablation_fixed.py — Bar chart for v3 ablation_fixed results with raw baseline.
"""

import matplotlib.pyplot as plt
import numpy as np

# ── Data ──────────────────────────────────────────────────────────────────────

conditions = ['raw\n(baseline)', 'bg', 'bg+dp', 'bg+ep', 'bg+dp+ep']

overall  = [0.691,  0.691,  0.703,  0.706,  0.705]
cognitive = [0.748,  0.755,  0.753,  0.753,  0.755]
pricing  = [0.618,  0.608,  0.638,  0.644,  0.639]

human_ceiling = 0.831
random_baseline = 0.570  # avg across conditions

# ── Plot ───────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=False)
fig.suptitle('V3 (v3_maximum) Skill Ablation — gpt-5.4-nano High Reasoning\n20 Personas, Correct Question Files',
             fontsize=13, fontweight='bold', y=1.02)

metrics = [
    (overall,   'Overall Accuracy',              [0.60, 0.75], axes[0]),
    (cognitive, 'Cognitive Bias Accuracy',        [0.68, 0.80], axes[1]),
    (pricing,   'Product Preference Accuracy',   [0.55, 0.70], axes[2]),
]

colors = ['#b0b0b0',  # raw baseline — grey
          '#4e79a7',  # bg
          '#f28e2b',  # bg+dp
          '#e15759',  # bg+ep
          '#59a14f',  # bg+dp+ep
          ]

for values, title, ylim, ax in metrics:
    bars = ax.bar(conditions, values, color=colors, edgecolor='white', linewidth=0.8, width=0.6)

    # Value labels on bars
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                f'{val:.3f}', ha='center', va='bottom', fontsize=8.5, fontweight='bold')

    # Human ceiling line
    ax.axhline(human_ceiling, color='#d62728', linestyle='--', linewidth=1.2, label=f'Human ceiling ({human_ceiling})')

    # Raw baseline shading (first bar region)
    ax.axvline(0.5, color='grey', linestyle=':', linewidth=0.8, alpha=0.5)

    ax.set_title(title, fontsize=11, fontweight='bold', pad=8)
    ax.set_ylim(ylim)
    ax.set_ylabel('Accuracy', fontsize=9)
    ax.tick_params(axis='x', labelsize=9)
    ax.tick_params(axis='y', labelsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3, linewidth=0.5)

    if ax == axes[2]:
        ax.legend(fontsize=8, loc='lower right')

# Condition legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor=colors[0], label='raw — raw Wave1-3 responses as persona'),
    Patch(facecolor=colors[1], label='bg — background only'),
    Patch(facecolor=colors[2], label='bg+dp — background + decision procedure'),
    Patch(facecolor=colors[3], label='bg+ep — background + evaluation profile'),
    Patch(facecolor=colors[4], label='bg+dp+ep — full (all 3 components)'),
]
fig.legend(handles=legend_elements, loc='lower center', ncol=3, fontsize=8.5,
           bbox_to_anchor=(0.5, -0.08), frameon=True, edgecolor='lightgrey')

plt.tight_layout()
out = 'experiments/v3_ablation_fixed_chart.png'
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f'Saved → {out}')
plt.close()
