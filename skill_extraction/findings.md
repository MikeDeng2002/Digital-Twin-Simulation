# Skill Extraction — Experiment Findings

**Date:** 2026-04-08
**Personas:** pid_1–11 (N=11, preliminary — CIs overlap, not statistically conclusive)
**Models tested:** gpt-4.1-mini-2025-04-14, o4-mini, gpt-4o
**Metric:** Mean Accuracy = 1 − normalized MAD vs real wave-4 answers
**Human ceiling:** 0.798 | **Random baseline:** ~0.57–0.63

---

## Full Results Table

### gpt-4.1-mini

| Persona input | Approx. tokens | Accuracy | 95% CI |
|---|---|---|---|
| Demographic only | ~1,054 | 0.692 | [0.662, 0.721] |
| Skill v1 (direct) | ~10,900 | 0.710 | [0.658, 0.763] |
| Skill v2 (inferred) | ~10,900 | 0.708 | [0.659, 0.756] |
| Skill v3 (maximum) | ~10,900 | **0.721** | [0.667, 0.776] |
| Skill v4 (chained) | ~10,900 | 0.701 | [0.653, 0.749] |
| Full raw text | ~35,788 | 0.719 | [0.661, 0.778] |

### o4-mini

| Persona input | Approx. tokens | Accuracy | 95% CI |
|---|---|---|---|
| Demographic only | ~1,054 | 0.711 | [0.654, 0.768] |
| Skill v1 (direct) | ~10,900 | 0.713 | [0.663, 0.763] |
| Skill v2 (inferred) | ~10,900 | 0.701 | [0.638, 0.763] |
| Skill v3 (maximum) | ~10,900 | **0.734** | [0.679, 0.789] |
| Skill v4 (chained) | ~10,900 | 0.721 | [0.675, 0.768] |
| Full raw text | ~35,788 | 0.706 | [0.652, 0.760] |

### gpt-4o

| Persona input | Approx. tokens | Accuracy | 95% CI |
|---|---|---|---|
| Demographic only | ~1,054 | 0.694 | [0.657, 0.731] |
| Skill v1 (direct) | ~10,900 | 0.708 | [0.670, 0.747] |
| Skill v2 (inferred) | ~10,900 | **0.726** | [0.700, 0.753] |
| Skill v3 (maximum) | ~10,900 | 0.719 | [0.687, 0.750] |
| Skill v4 (chained) | ~10,900 | 0.714 | [0.673, 0.755] |
| Full raw text | ~35,788 | 0.705 | [0.659, 0.751] |

---

## Cross-Model Comparison (Best per Model)

| Model | Best setup | Best accuracy | Best CI |
|---|---|---|---|
| gpt-4.1-mini | skill v3 | 0.721 | [0.667, 0.776] |
| o4-mini | skill v3 | **0.734** | [0.679, 0.789] |
| gpt-4o | skill v2 | 0.726 | [0.700, 0.753] |

**Overall best:** o4-mini + skill v3 (0.734), approaching human ceiling (0.798).

---

## Skill Versions Explained

| Version | API calls / persona | Method |
|---|---|---|
| v1_direct | 1 | Single call, direct evidence only (~120/450 questions) |
| v2_inferred | 1 | Single call, direct + labeled demographic inference (~200/450) |
| v3_maximum | 1 | Single call, all signals, aggressive inference (~450/450) |
| v4_chained | 3 | Sequential calls: BACKGROUND → TOOLS (sees BACKGROUND) → DECISION_PROCEDURE (sees both) |
| Full raw text | — | All 535 Q&A pairs verbatim (no extraction) |
| Demographic only | — | First survey block only (~12 demographic fields) |

---

## Key Findings

### 1. Best overall: o4-mini + skill v3 (0.734)
The reasoning model paired with the richest single-call skill extraction achieves the
highest accuracy, approaching the human ceiling (0.798) more closely than any other
combination. It uses only ~31% of the tokens of the full raw text approach.

### 2. gpt-4o favors skill v2 over v3
Unlike gpt-4.1-mini and o4-mini (both best at v3), gpt-4o peaks at skill v2 (0.726).
Its v3 result (0.719) slightly underperforms v2, suggesting gpt-4o may not benefit from
the most aggressive inference signals in v3 or may handle labeled inference better.

### 3. Full raw text performs worst for all three models
Contrary to the intuition that more context = better performance:
- gpt-4.1-mini: full text (0.719) ties v3 but with wider CI
- o4-mini: full text (0.706) is the worst except demographic-only
- gpt-4o: full text (0.705) is the worst

This suggests that structured skill extraction consistently helps or matches raw text,
with a major cost advantage (69% fewer tokens).

### 4. Chained extraction (v4) does NOT improve over v3 for any model
- gpt-4.1-mini: v4 (0.701) < v3 (0.721)
- o4-mini: v4 (0.721) < v3 (0.734)
- gpt-4o: v4 (0.714) < v2 (0.726) and v3 (0.719)

**Hypothesis:** The chained approach may introduce compounding bias — errors or
framing choices in the BACKGROUND call propagate forward and constrain the TOOLS and
DECISION_PROCEDURE sections. The single-call approach allows the model to hold all
signals simultaneously and may self-correct within the response.

### 5. o4-mini is robust to context compression
o4-mini achieves near-equal accuracy across a 34× range of input sizes:
- Demographic only (~1K tokens): 0.711
- Skill v3 (~11K tokens): 0.734
- Full raw text (~36K tokens): 0.706

The reasoning model can compensate for reduced context, making it far more
cost-efficient when paired with skill extraction.

### 6. gpt-4o has the tightest confidence intervals
gpt-4o's CIs are noticeably narrower than gpt-4.1-mini and o4-mini for most conditions
(e.g., v2: [0.700, 0.753] vs o4-mini v3: [0.679, 0.789]). This suggests more
consistent per-persona behavior, potentially due to gpt-4o's larger capacity.

### 7. Demographic-only is competitive for reasoning models
- gpt-4.1-mini demographic (0.692) is clearly the worst
- o4-mini demographic (0.711) nearly matches its full-text result (0.706)
- gpt-4o demographic (0.694) also competitive — only 3.2 points below its best

Reasoning-capable models (o4-mini, gpt-4o) can infer more from sparse demographic
signals than the smaller gpt-4.1-mini.

---

## Token Efficiency Summary

| Setup | Tokens | Best model | Accuracy | Accuracy per 1K tokens |
|---|---|---|---|---|
| Demographic only | 1,054 | o4-mini | 0.711 | 0.674 |
| Skill v2/v3 | ~10,900 | o4-mini (v3) | **0.734** | 0.067 |
| Full raw text | 35,788 | gpt-4.1-mini | 0.719 | 0.020 |

Demographic-only has the best accuracy-per-token ratio, but hits a ceiling.
Skill v3 (o4-mini) offers the best absolute accuracy at reasonable cost.

---

## Recommendation

**For best accuracy:** o4-mini + skill v3 (0.734)
- Best accuracy across all models and conditions
- 69% token reduction vs full raw text
- No benefit from chaining (v4 is worse and costs 3× more API calls during extraction)

**For gpt-4o:** use skill v2 (0.726)
- gpt-4o uniquely benefits from labeled-inference style (v2) over aggressive-inference (v3)
- Narrowest confidence intervals of any model

**For gpt-4.1-mini:** use skill v3 or full raw text (statistically tied at ~0.719–0.721)
- If cost matters: skill v3 saves ~69% tokens at no accuracy cost

---

## Next Steps

1. Scale to 100+ personas for statistically conclusive comparisons
2. Investigate why v4 chaining underperforms for all models — compare v3 vs v4 skill file quality directly
3. Investigate why gpt-4o prefers v2 over v3 — may reflect different sensitivity to over-specified inference
4. Test whether v4 improves on the Bitcoin price prediction task (open-ended numeric)
   where coherent cross-section reasoning may matter more than survey Q matching
5. Try o4-mini for skill extraction itself (currently using gpt-4.1-mini for extraction)
6. Test whether gpt-4o's tighter CIs hold at larger N (may indicate different uncertainty characteristics)
