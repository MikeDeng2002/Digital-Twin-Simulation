# Bitcoin Celebrity Influence Experiment — Results Report

**Date:** 2026-04-08
**Model:** o4-mini (center personas) + gpt-4.1-mini (celebrity generation)
**Representation:** v3_maximum
**Sample:** 99 personas with trial data (83 complete or near-complete)
**Trials per persona:** 50 (target); 5–31 for partial personas
**Celebrity pool:** 6 bulls (Saylor, Wood, Musk, Dorsey, Fink, Trump) + 4 bears (Buffett, Schiff, Dimon, Roubini)
**Design:** 2 bulls + 2 bears drawn per trial (seed = pid × 1000 + trial)

---

## Celebrity Predictions (LLM-Generated, Cached)

| Celebrity | Category | Predicted Price (end-2025) |
|-----------|----------|---------------------------|
| Michael Saylor | BULL | $1,000,000 |
| Cathie Wood | BULL | $500,000 |
| Jack Dorsey | BULL | $500,000 |
| Donald Trump | BULL | $250,000 |
| Elon Musk | BULL | $150,000 |
| Larry Fink | BULL | $150,000 |
| Warren Buffett | BEAR | $15,000 |
| Jamie Dimon | BEAR | $5,000 |
| Peter Schiff | BEAR | $3,000 |
| Nouriel Roubini | BEAR | $2,000 |

---

## Baseline Distribution

- **n = 99 personas** with baselines (o4-mini, no neighbors)
- **Mean baseline:** $60,222
- Range: $10,000–$100,000

---

## 1. Per-Celebrity Influence Weights (Type A — Number Only)

### Regression Model

For each persona, OLS estimates:

```
prediction = intercept + w_baseline × baseline
           + w_saylor × (Saylor price if present, else 0)
           + w_wood   × (Wood price if present, else 0)
           + ...
           + w_roubini × (Roubini price if present, else 0)
```

Each celebrity gets its own weight. The feature for celebrity `i` in trial `t` is their price when they appear, zero otherwise. This lets us measure the marginal dollar-for-dollar influence of each individual celebrity.

### Per-Celebrity Influence Summary (mean across 78–99 personas)

| Celebrity | Category | Mean w | Median w | Std | % Positive |
|-----------|----------|--------|----------|-----|------------|
| Peter Schiff | BEAR | +2.676 | +2.414 | 5.34 | 74.4% |
| Larry Fink | BULL | +0.306 | +0.303 | 0.15 | 98.7% |
| Elon Musk | BULL | +0.170 | +0.174 | 0.15 | 92.2% |
| Donald Trump | BULL | +0.145 | +0.144 | 0.06 | 98.7% |
| Jack Dorsey | BULL | +0.143 | +0.139 | 0.06 | 100.0% |
| Cathie Wood | BULL | +0.117 | +0.118 | 0.05 | 98.7% |
| Michael Saylor | BULL | +0.058 | +0.055 | 0.03 | 98.7% |
| Jamie Dimon | BEAR | +0.052 | −0.333 | 4.68 | 44.9% |
| Warren Buffett | BEAR | −0.444 | −0.632 | 1.64 | 30.8% |
| Nouriel Roubini | BEAR | −1.737 | −1.714 | 4.25 | 24.4% |

### Interpreting the Weights

**Important:** The coefficient `w_i` represents the dollar change in prediction per $1 increase in celebrity `i`'s price (when present). Because bear celebrities have very low prices ($2K–$15K), their weights must be large in magnitude to produce any absolute dollar effect:

| Celebrity | Price | Mean w | Implied contribution per appearance |
|-----------|-------|--------|-------------------------------------|
| Saylor | $1,000,000 | 0.058 | +$58,000 |
| Wood | $500,000 | 0.117 | +$58,500 |
| Dorsey | $500,000 | 0.143 | +$71,500 |
| Trump | $250,000 | 0.145 | +$36,250 |
| Fink | $150,000 | 0.306 | +$45,900 |
| Musk | $150,000 | 0.170 | +$25,500 |
| Buffett | $15,000 | −0.444 | −$6,660 |
| Dimon | $5,000 | +0.052 | +$260 |
| Schiff | $3,000 | +2.676 | +$8,028 |
| Roubini | $2,000 | −1.737 | −$3,474 |

**Key findings:**

1. **Bulls uniformly positive:** All 6 bull celebrities have mean w > 0, with near-100% of personas showing positive influence. They consistently pull predictions upward.

2. **Bears noisy and divided:** Bear weights have high variance and mixed signs across personas. Buffett and Roubini are net-negative (downward pull), while Schiff and Dimon are surprisingly net-positive in aggregate — personas who recognize Schiff as a bear may still anchor slightly upward on his $3K price (which is not zero).

3. **Largest per-appearance absolute effect:** Dorsey and Wood each contribute ~$58–72K per appearance despite lower raw weights than Fink, because their prices are higher.

4. **Saylor underperforms his price:** Despite predicting $1M, his mean w is the lowest among bulls (0.058), giving a ~$58K contribution. His extreme prediction may trigger skepticism.

5. **Larry Fink most consistently influential:** Highest median w among all celebrities (0.303), present in 98.7% of positive-influence personas — possibly more credible as a mainstream financial figure than crypto maximalists.

---

## 2. Per-Persona Heterogeneity

Top 10 personas by mean absolute celebrity influence weight:

| PID | w_baseline | R² | Notable pattern |
|-----|-----------|-----|-----------------|
| 72 | −0.177 | 1.000 | Extreme bear sensitivity; Dimon w=23.7 |
| 69 | +0.170 | 0.322 | Schiff w=16.9 |
| 75 | +0.160 | 0.624 | Schiff w=18.7 |
| 71 | +0.249 | 0.753 | Schiff w=−10.7 (contrarian to Schiff) |
| 37 | +0.055 | 0.431 | Dimon w=13.6 |
| 24 | +0.152 | 0.488 | Schiff w=10.4 |
| 65 | +0.267 | 0.410 | Musk w=0.59, Buffett w=−2.6 |
| 67 | +0.089 | 0.609 | Schiff w=6.6, Dimon w=6.4 |
| 17 | +0.049 | 0.234 | Schiff w=8.3 |
| 23 | +0.152 | 0.446 | Schiff w=12.5 |

**Observation:** The bear celebrities (especially Schiff) drive most of the per-persona variance — personas differ primarily in how they respond to low, bearish price signals.

---

## 3. Type A vs Type B: Effect of Adding Reasoning

**Mean diff (B − A) = −$13,696** across all 99 personas.

Adding celebrity reasoning alongside prices consistently *reduces* predictions (~86% of personas). Numbers alone are more persuasive than numbers with explicit justification — reasoning triggers critical evaluation rather than persuasion.

| Statistic | Value |
|-----------|-------|
| Mean B − A | −$13,696 |
| Largest drop | pid_32: −$51,469 |
| Largest increase | pid_95: +$50,000 |

---

## 4. Summary

1. **All bulls are positive influencers** — 92–100% of personas show positive w per bull celebrity.
2. **Bears are divisive** — Buffett and Roubini are net-negative; Schiff and Dimon are noisy with mixed signs.
3. **Larry Fink is the most reliable bull influencer** (median w = 0.303, 98.7% positive), despite a moderate $150K prediction — mainstream credibility beats extremism.
4. **Saylor's extreme $1M prediction backfires** — lowest w among bulls (0.058), suggesting skepticism when a prediction is too far from prior.
5. **Bears generate the most heterogeneity** — individual differences in how personas respond to Schiff, Dimon, Roubini explain most of the cross-persona variance.
6. **Reasoning reduces prices** — Type B consistently lower than Type A (mean −$13.7K).

---

## Files

| File | Content |
|------|---------|
| `results/v3_maximum/baselines/pid_*.json` | Baseline predictions (99 personas) |
| `results/v3_maximum/type_ab/pid_*.json` | Trial data (Type A + B, 99 personas) |
| `analysis/w_celeb_per_celebrity_type_a.csv` | Full per-celebrity OLS weights per persona |
| `analysis/celebrity_influence_summary.csv` | Celebrity influence summary across personas |
1| `analysis/type_a_vs_b_comparison.csv` | Type A vs B prediction differences |
| `analysis/results_report.md` | This file |
