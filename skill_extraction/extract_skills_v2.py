"""
extract_skills_v2.py — Generate skill profiles using the new 3-file design.

New design (EXTRACTION_PROMPTS.md):
  - 3 files per version: background.txt, decision_procedure.txt, evaluation_profile.txt
  - 9 API calls per persona (3 files × 3 versions), each call dedicated to one file
  - No tools.txt, no w_social scalar estimate
  - Structured markdown sections inside each file

Output:
  text_simulation/skills_v2/pid_{pid}/
  ├── v1_direct/
  │   ├── background.txt
  │   ├── decision_procedure.txt
  │   └── evaluation_profile.txt
  ├── v2_inferred/
  │   └── ...
  └── v3_maximum/
      └── ...

Usage (from Digital-Twin-Simulation/):
    python skill_extraction/extract_skills_v2.py --pid 1
    python skill_extraction/extract_skills_v2.py --pid 1 --version v1_direct
    python skill_extraction/extract_skills_v2.py --pid 1 --force
"""

import os
import argparse
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

MODEL      = "gpt-4.1-mini-2025-04-14"
VERSIONS   = ["v1_direct", "v2_inferred", "v3_maximum"]

PERSONA_DIR = Path("text_simulation/text_personas")
OUTPUT_BASE = Path("text_simulation/skills_v2")

# Max tokens per version (from EXTRACTION_PROMPTS.md)
MAX_TOKENS = {
    "v1_direct":  1500,
    "v2_inferred": 2000,
    "v3_maximum":  2500,
}

# ── Prompts ───────────────────────────────────────────────────────────────────

PROMPT_BACKGROUND_V1 = """\
You are extracting a digital twin skill profile from a survey transcript.

STRICT RULE: Use ONLY questions that directly and explicitly ask about the
topic. Do NOT infer from demographics. Do NOT guess from related answers. If
no direct evidence exists for a component, say so explicitly rather than
fabricating an answer.

This file describes what the person knows, believes, and is — their
demographics, stated beliefs, personality, and directly-reported
knowledge. Do NOT include reasoning style (that's decision_procedure.txt) or
preferences/willingness-to-pay (that's evaluation_profile.txt).

The survey covers personality scales (Big Five, Need for Closure, Need for
Cognition, Maximizer, Self-monitoring, Self-concept clarity), values,
economic games, cognitive tests, and demographics.

Output EXACTLY this format with no other text:

---BACKGROUND---
Write prose in these sections, using markdown ## headers. No bullet lists in
the body prose.

## Demographics
Direct demographic facts: age, sex, region, race, education, marital status,
employment, income, household size, religion, political identity. State
these as reported.

## Stated beliefs and self-concept
Direct self-report items about who the person is and what they believe.
Quote or paraphrase free-response items if present.

## Personality
Big Five profile using direct item responses. Cite item numbers. State each
trait level only where items directly support it.

## Values (directly reported)
List the values scale ratings as reported. Do not interpret acquiescence
patterns or infer a values hierarchy — just report what was rated.

## Factual knowledge (directly tested)
Report performance on any items that directly test factual knowledge
(financial literacy, basic probability, etc.). Cite which items and the
answers given.

## Summary
Three to five sentences capturing only directly reported content. Do not
infer domain familiarity or lived knowledge beyond what was explicitly
asked.

TRANSCRIPT:
{transcript}"""


PROMPT_BACKGROUND_V2 = """\
You are extracting a digital twin skill profile from a survey transcript.

RULES:
- Direct evidence: cite the scale name and item number
- Demographic inference: allowed but label it explicitly as:
  "Inferred from [demographic fact]: [conclusion] because [reasoning]"
- Cross-scale inference: allowed when two or more scales point in the same
  direction — note which scales agree
- No speculation beyond what the data can reasonably support

This file describes what the person knows, believes, and is — their
demographics, stated beliefs, personality, values, and inferred domain
familiarity. Do NOT include reasoning style (that's decision_procedure.txt)
or willingness-to-pay/preference structure (that's evaluation_profile.txt).

The survey covers personality scales (Big Five, Need for Closure, Need for
Cognition, Maximizer, Self-monitoring, Self-concept clarity), values,
economic games, cognitive tests, and demographics.

Output EXACTLY this format with no other text:

---BACKGROUND---
Write prose in these sections, using markdown ## headers. No bullet lists in
the body prose.

## Demographics
Direct demographic facts as reported.

## Stated beliefs and self-concept
Synthesize direct self-report items with free-response content. If the
free-response aspirational-self and actual-self descriptions reveal something
not captured in scales, surface it.

## Personality
Big Five profile with trait-level summaries supported by item evidence. Note
any contradictions between personality items and behavioral evidence
(e.g., low self-reported Neuroticism with high somatic anxiety symptoms).

## Values (directly reported, with acquiescence caveat)
Report direct values ratings. If the ratings are compressed (most values in
a narrow band like 7-8 on a 9-point scale), name this explicitly and extract
only the relative structure — which values are uniquely high or uniquely
low. Do NOT treat a compressed ranking as a clean hierarchy.

## Factual knowledge (directly tested)
Report performance on directly tested items (financial literacy, basic
probability). Note the pattern of correct-vs-incorrect answers.

## Likely domain familiarity
Inferred from demographics, education, income, employment, and lifestyle
signals: what domains is this person likely to have lived knowledge of, and
what domains are they likely unfamiliar with? Label each inference. Keep
this section brief — it is meant to condition downstream predictions on
realistic knowledge access, not to generate speculative biographies. Examples
of legitimate inferences:
- "Inferred from income and household size: familiar with grocery pricing in
  the $X range, unlikely to be familiar with premium/specialty grocery
  pricing."
- "Inferred from education level: basic financial literacy confirmed by
  direct items; unlikely familiarity with specialized financial instruments
  or investment terminology."

## Summary
Three to five sentences capturing who this person is, what they believe, and
what they're likely to know about. Consistent with the evidence above.

TRANSCRIPT:
{transcript}"""


PROMPT_BACKGROUND_V3 = """\
You are extracting the richest possible background profile from a digital
twin survey transcript. Use every available signal — direct scales, free-
response text, behavioral patterns, demographic context, and any cross-scale
convergence. Infer aggressively but always state your reasoning chain.

This file describes what the person knows, believes, and is. Do NOT include
reasoning style (that's decision_procedure.txt) or willingness-to-pay
(that's evaluation_profile.txt).

Goal: produce the most complete description of this person's identity,
beliefs, personality, and likely knowledge base, for use by a downstream
simulation that will answer questions as if it were this person.

CRITICAL EXTRACTION PRINCIPLES:

1. ACQUIESCENCE IN VALUES RATINGS — name it when present; extract only
   relative structure.

2. FREE-RESPONSE TEXT IS HIGH-SIGNAL — aspirational and actual-self
   descriptions often reveal self-concept that scales miss. Quote or
   paraphrase them when informative.

3. SELF-REPORT CONTRADICTIONS ARE DATA — if self-reported Neuroticism is
   low but somatic anxiety symptoms are moderate-to-severe, this is a
   finding about self-awareness, not a contradiction to smooth over.

4. DOMAIN FAMILIARITY MUST BE INFERRED RESPONSIBLY — use education,
   income, employment, household, and region to condition what topics the
   person is likely to know about. State confidence for each inference.
   Do not fabricate a biography.

The survey covers: Big Five (44 items), Need for Cognition (18 items),
Need for Closure (15 items), Maximizer (6 items), Self-monitoring (13
items), Self-concept clarity (12 items), Values (24 items), Minimalism
(12 items), Empathy (20 items), consumerism/uniqueness scales, economic
games, cognitive tests, word association chain, social desirability,
full demographics, and free-response items.

Output EXACTLY this format with no other text:

---BACKGROUND---
Write prose in these sections, using markdown ## headers. No bullet lists in
the body prose.

## Demographics
Full demographic profile. Note any demographic details likely to be
predictive of knowledge, behavior, or constraints (e.g., time poverty from
combined employment and household load).

## Stated beliefs and self-concept
Direct self-report items plus free-response content. Include quotes or
paraphrases from aspirational-self and actual-self responses when they
reveal something the scales don't. Note any gap between how the person
describes themselves and how they behave elsewhere in the transcript.

## Personality
Full Big Five profile with specific item evidence. Cross-reference with
Self-monitoring, Self-concept clarity, and empathy scales. Note any
contradictions (low Neuroticism + high somatic anxiety, etc.) and what they
imply about self-report reliability for this person.

## Values (directly reported)
Report values ratings with acquiescence caveat if applicable. Extract
relative structure — uniquely high and uniquely low values. Note if the
value profile looks achievement-oriented, security-oriented, self-
transcendence-oriented, or mixed. Do NOT interpret what values imply for
pricing or preference behavior here — that's evaluation_profile.txt.

## Political and policy beliefs
Direct political identity items plus any policy-specific endorsements in the
transcript. Note strength of partisan identification vs. moderate/
independent positioning.

## Factual knowledge (directly tested)
Performance on financial literacy, probability, and any other directly
tested knowledge items. Note which domains the person demonstrably knows.

## Likely domain familiarity
Full inference section. Using demographics, education, income, employment,
household, region, and any free-response evidence of life experience,
predict what domains this person has lived knowledge of and which they
likely don't. Examples of dimensions to consider:
- Household finance and budgeting depth
- Consumer product familiarity across tiers
- Health, medical, and pharmaceutical knowledge
- Technology and digital-services familiarity
- Political, policy, and civic knowledge
- Specialized professional domains

Label each inference with confidence. Do not generate speculative
biographies — stick to what is defensibly likely given the demographic and
lifestyle signals.

## Self-report reliability indicators
Note any signals affecting how much to trust this person's self-reports:
- Social desirability score
- Acquiescence patterns in values and Likert items
- Stated-vs-behavioral gaps observed elsewhere (for context; full analysis
  lives in decision_procedure.txt)
- Free-response self-awareness

This section conditions downstream interpretation of the persona's own
statements during simulation.

## Summary
Four to six sentences capturing who this person is, what they believe, and
the domains in which their knowledge is likely strong vs. weak.

TRANSCRIPT:
{transcript}"""


PROMPT_DECISION_V1 = """\
You are extracting a digital twin skill profile from a survey transcript.

STRICT RULE: Use ONLY questions that directly and explicitly measure the
reasoning constructs below. Do NOT infer from demographics. Do NOT guess
from related answers. If no direct evidence exists for a component, say so
explicitly rather than fabricating an answer.

This file describes how the person reasons under uncertainty. It is
descriptive only — do NOT output any scalar prediction, social-influence
weight, or committed numeric forecast at the end.

The survey contains direct psychometric scales for:
- Need for Cognition (18 items)
- Need for Closure (15 items)
- Maximizer scale (6 items)
- Self-concept clarity (12 items)
- Self-monitoring (13 items)
- Social desirability
- Cognitive test items (syllogisms, Wason, analogies, CRT-style items) —
  cite specific item numbers and answers
- Intertemporal choice tasks

Output EXACTLY this format with no other text:

---DECISION_PROCEDURE---
Write prose in these sections, using markdown ## headers. No bullet lists in
the body prose.

## Effort regulation and cognitive style
Describe the person's Need for Cognition profile and performance on
cognitive test items. Cite specific item numbers and answers. If they made
errors on CRT-style items, name the specific intuitive-trap pattern. If no
cognitive test items were answered or NFC was not measured, state this
explicitly.

## Closure and uncertainty tolerance
Describe the person's Need for Closure profile using direct scale items.
Cite item numbers. If no NFCC items are present, state this.

## Optimization strategy
Describe Maximizer scale responses directly. Cite item numbers.

## Self-concept stability
Describe Self-concept clarity scale responses directly. Cite item numbers.

## Intertemporal patience
Describe responses to direct intertemporal choice items. If not present,
state this.

## Bias susceptibility
List ONLY biases for which direct scale evidence exists in the transcript.
Do not predict susceptibility to biases whose psychometric correlates
aren't directly measured. For each bias you do name, cite the specific
item or scale that supports the prediction.

## Summary
Three to five sentences capturing only what was directly measured. Do not
extrapolate beyond the cited items.

TRANSCRIPT:
{transcript}"""


PROMPT_DECISION_V2 = """\
You are extracting a digital twin skill profile from a survey transcript.

RULES:
- Direct evidence: cite the scale name and item number
- Demographic inference: allowed but label it explicitly as:
  "Inferred from [demographic fact]: [conclusion] because [reasoning]"
- Cross-scale inference: allowed when two or more scales point in the same
  direction — note which scales agree
- No speculation beyond what the data can reasonably support

This file describes how the person reasons under uncertainty. It is
descriptive only — do NOT output any scalar prediction, social-influence
weight, or committed numeric forecast at the end.

CRITICAL EXTRACTION PRINCIPLE — STATED VS REVEALED GAP:
If the person's stated attitudes (from personality or values scales)
conflict with their behavioral choices (economic games, intertemporal
tasks, cognitive test error patterns, free-response self-descriptions),
surface the conflict explicitly. Do NOT average or smooth over the gap.
This gap is often the single most predictive feature of reasoning behavior.

The survey covers: Big Five, Need for Cognition, Need for Closure,
Maximizer, Self-monitoring, Self-concept clarity, Values, economic games
(dictator, ultimatum, trust), lottery/risk tasks, intertemporal choice,
cognitive tests (syllogisms, Wason, analogies, CRT-style items), social
desirability, and demographics.

Output EXACTLY this format with no other text:

---DECISION_PROCEDURE---
Write prose in these sections, using markdown ## headers. No bullet lists in
the body prose.

## Effort regulation and cognitive style
Synthesize Need for Cognition scale items with actual cognitive test
performance. Name specific error patterns. Cross-reference with
Conscientiousness from Big Five where relevant.

## Closure and uncertainty tolerance
Combine NFCC scale responses with behavioral signals (cognitive test time
pressure, free-response length, decision consistency across similar items).
Infer commitment speed with reasoning.

## Optimization strategy
Maximizer scale plus domain-specific exceptions. If the person shows
satisficing on some domains but maximizing on others, name the split with
evidence.

## Self-concept stability
SCC scale plus any observed contradictions between self-descriptions across
the transcript. Name specific contradictions if they exist.

## Behavioral vs. stated-preference gap
This is the most important section if the gap exists. Compare:
- Economic game choices vs. stated values/altruism items
- Free-response aspirational-self vs. actual-self descriptions
- Self-reported personality vs. behavioral evidence elsewhere
If a meaningful gap is present, describe what it implies for reasoning
reliability and self-report trustworthiness. If no gap is present, state
this briefly.

## Intertemporal patience
From delayed-reward tasks plus Conscientiousness and planning signals.

## Bias susceptibility predictions
Predict susceptibility to each bias in the expected task battery. Use this
framework:

Predictable from cognitive signals (state a confident prediction):
- Probability matching vs. maximizing
- Dominator (denominator) neglect
- Belief bias in syllogisms
- Wason matching bias

Weakly predictable or not predictable from cognitive signals (note the
limits, per Stanovich & West 2008):
- Framing effects
- Anchoring and adjustment
- Base-rate neglect
- Conjunction fallacy (Linda problem)
- Outcome bias
- Sunk-cost effect
- Less-is-more effect
- Omission bias
- Allais / certainty effects
- Myside bias

For the second group, acknowledge predictive limits rather than over-claim.
For the first group, commit to a directional prediction with reasoning.

## Summary
Three to five sentences capturing the reasoning profile. The summary must be
consistent with the evidence above — do not overstate confidence.

TRANSCRIPT:
{transcript}"""


PROMPT_DECISION_V3 = """\
You are extracting the richest possible reasoning profile from a digital
twin survey transcript. Use every available signal — direct scales,
cognitive test performance and specific error patterns, behavioral game
choices, free-response self-descriptions, somatic symptoms, social
desirability scores, word associations if present, and full demographic
context. Infer aggressively but always state your reasoning chain.

This file describes how the person reasons under uncertainty. It is
descriptive only — do NOT output any scalar prediction, social-influence
weight, or committed numeric forecast at the end. Any influenceability or
bias prediction should be expressed qualitatively in the prose.

Goal: produce the most complete, behaviorally predictive reasoning profile
for simulating this person on judgment-under-uncertainty tasks (framing,
anchoring, base-rate, conjunction, sunk cost, probability matching, myside
bias) and on numeric prediction tasks.

CRITICAL EXTRACTION PRINCIPLES:

1. STATED VS REVEALED GAP — surface any conflict between stated attitudes
   and behavioral choices. Do NOT average. The gap is the signal.

2. COGNITIVE ERROR PATTERNS MATTER MORE THAN TOTAL SCORES — name specific
   error types when they occur.

3. SOMATIC-VS-SELF-REPORT CALIBRATION — if anxiety symptoms are high but
   self-reported Neuroticism is low, note this as a self-report reliability
   issue.

4. FREE-RESPONSE TEXT IS HIGH-SIGNAL — quote or paraphrase when they
   illuminate reasoning style.

5. ACKNOWLEDGE PREDICTIVE LIMITS — per Stanovich & West (2008), most H&B
   biases are NOT reliably correlated with cognitive ability.

The survey covers: Big Five (44 items), Need for Cognition (18 items),
Need for Closure (15 items), Maximizer (6 items), Self-monitoring (13
items), Self-concept clarity (12 items), Values (24 items), Minimalism
(12 items), Empathy (20 items), consumerism/uniqueness scales, economic
games (dictator, ultimatum, trust), intertemporal choice tasks,
lottery/risk tasks, cognitive tests, word association chain, social
desirability, and demographics.

Output EXACTLY this format with no other text:

---DECISION_PROCEDURE---
Write prose in these sections, using markdown ## headers. No bullet lists in
the body prose.

## Effort regulation and cognitive style
Full synthesis of NFC score, cognitive test performance with named error
patterns, Conscientiousness from Big Five, and any free-response evidence
of thinking style. Name specific items and answers. Infer analytical depth
with reasoning.

## Closure and uncertainty tolerance
NFCC scale plus behavioral signals plus free-response evidence of how the
person handles ambiguity in daily life.

## Optimization strategy
Maximizer plus domain-specific evidence. Include evidence from consumer
choices, free-response text about indecision or completion, and any
observable pattern of re-evaluation vs. commitment in the survey itself.

## Self-concept stability
SCC scale, contradictions between self-descriptions, gaps between
aspirational-self and actual-self free responses, any Big Five items
contradicting each other.

## Behavioral vs. stated-preference gap
Thorough analysis. Compare economic games to stated values. Compare free-
response self-descriptions to personality items. Use social desirability
score to weight self-reports globally — if social desirability is high,
all self-reports should be discounted and behavioral data weighted more
heavily.

## Intertemporal patience
Full analysis from delayed-reward items, Conscientiousness, and free-
response evidence of planning vs. procrastination.

## Egocentric projection tendency
Predicts false-consensus magnitude. Use: self-concept clarity (low SCC →
more projection), empathy (low empathy → more projection), political
identity strength (stronger identity → more projection on policy items).

## Bias susceptibility predictions
For each bias, give a prediction and its confidence:

HIGH CONFIDENCE (predictable from cognitive signals):
- Probability matching vs. maximizing
- Dominator (denominator) neglect
- Belief bias
- Wason matching bias

LOW CONFIDENCE (not reliably predictable — acknowledge limits):
- Framing effects
- Anchoring
- Base-rate neglect
- Conjunction fallacy
- Outcome bias
- Sunk-cost effect
- Less-is-more effect
- Omission bias
- Allais / certainty effects
- Myside bias

MODERATE CONFIDENCE (predictable from belief content + projection
tendency):
- False consensus effect
- Risk-benefit nonseparability (affect heuristic)

## Self-report reliability weighting
Explicitly state how much the reader should trust this person's self-
reports given: social desirability score, stated-vs-revealed gaps
observed, somatic-vs-self-report calibration.

## Summary
Four to six sentences capturing the complete reasoning profile. Include
the most predictive single feature (often the stated-vs-revealed gap or a
specific cognitive error pattern).

TRANSCRIPT:
{transcript}"""


PROMPT_EVALUATION_V1 = """\
You are extracting a digital twin skill profile from a survey transcript.

STRICT RULE: Use ONLY questions that directly and explicitly measure the
preference and value constructs below. Do NOT infer from demographics. Do
NOT guess from related answers. If no direct evidence exists for a
component, say so explicitly rather than fabricating an answer.

This file describes what the person values and what she'd pay for. It
predicts preference-revelation behavior (pricing studies, product choices,
WTA/WTP gaps). Do NOT include reasoning style or bias susceptibility —
those belong in decision_procedure.txt.

The survey contains direct preference/value scales for:
- Values hierarchy (24 items)
- Minimalism scale (12 items)
- Consumerism / uniqueness-seeking scales
- Environmental concern (6 items)
- Direct income and employment items
- Lottery / risk preference tasks
- Intertemporal choice tasks
- Economic games (dictator, ultimatum, trust)

CRITICAL MEASUREMENT CAVEAT:
The values scale is vulnerable to acquiescence bias — many respondents rate
most values at 7 or 8 on a 9-point scale, producing a compressed
distribution. If you observe rating compression, name this explicitly and
extract only the relative structure (uniquely high or uniquely low values).
Do NOT treat a compressed ranking as a clear hierarchy.

Output EXACTLY this format with no other text:

---EVALUATION_PROFILE---
Write prose in these sections, using markdown ## headers. No bullet lists in
the body prose.

## Budget constraint
Direct income, employment, household size. No demographic inference about
spending style — just the budget envelope.

## Values hierarchy
Direct ratings from the 24-item values scale. Name the highest-rated and
lowest-rated values. If rating compression is present, name this explicitly
before extracting the relative structure. Cite specific item numbers.

## Minimalism and consumer orientation
Direct responses to the minimalism scale and any direct consumerism items.
Cite item numbers.

## Risk preferences
Direct responses to lottery/risk tasks. If not present, state this.

## Intertemporal patience
Direct responses to delayed-reward items. If not present, state this.

## Social preferences
Direct responses to economic games (dictator, ultimatum, trust). Name the
specific allocation choices. Do not infer family-vs-stranger context unless
the items directly test this.

## Summary
Three to five sentences capturing only directly measured preferences. Do
not predict specific product choices beyond what the measured scales
support.

TRANSCRIPT:
{transcript}"""


PROMPT_EVALUATION_V2 = """\
You are extracting a digital twin skill profile from a survey transcript.

RULES:
- Direct evidence: cite the scale name and item number
- Demographic inference: allowed but label it explicitly as:
  "Inferred from [demographic fact]: [conclusion] because [reasoning]"
- Cross-scale inference: allowed when two or more scales point in the same
  direction — note which scales agree
- No speculation beyond what the data can reasonably support

This file describes what the person values and what she'd pay for. It
predicts preference-revelation behavior (pricing studies, product choices,
WTA/WTP gaps). Do NOT include reasoning style or bias susceptibility.

CRITICAL EXTRACTION PRINCIPLES:

1. ACQUIESCENCE IN VALUES RATINGS — if the values scale shows compression
   (most values at 7-8 on the 9-point scale), name this explicitly and
   extract only relative structure. Do not treat compressed rankings as
   clean hierarchies.

2. STRANGER-VS-FAMILY DISTINCTION IN SOCIAL PREFERENCES — a persona can
   simultaneously show strong family-collectivism and selfish stranger-
   allocation behavior. Preserve this context dependence; do not average.

3. MINIMALISM OVERRIDES MANY OTHER SIGNALS FOR PRICING PREDICTIONS — a
   strong minimalist will skip purchases regardless of income, brand
   familiarity, or stated pleasure-seeking. Weight minimalism heavily if
   it's present.

The survey covers: Big Five, Values (24 items), Minimalism (12 items),
consumerism/uniqueness scales, environmental concern, economic games,
lottery/risk tasks, intertemporal choice, demographics, and free-response
items.

Output EXACTLY this format with no other text:

---EVALUATION_PROFILE---
Write prose in these sections, using markdown ## headers. No bullet lists in
the body prose.

## Budget constraint
Combine income, household size, employment status, and any free-response
signals about financial pressure or time constraints. Infer a spending
ceiling for everyday purchases with reasoning.

## Values hierarchy
Direct ratings with acquiescence caveat if applicable. Name uniquely high
and uniquely low values. Cross-reference with Big Five (e.g., high
Conscientiousness + high COMPETENCE rating = convergent signal). Note if
stated values seem consistent or inconsistent with other evidence.

## Minimalism and consumer orientation
Minimalism scale plus inferred brand orientation. Combine with income and
consumerism items to predict premium vs. mainstream vs. budget-brand
preferences.

## Intertemporal patience
From delayed-reward tasks. Cross-reference with Conscientiousness for
planning capacity.

## Risk preferences
From lottery tasks. Note gain/loss asymmetry if present. Cross-reference
with income (moderate-income risk aversion is the norm).

## Social preferences
Compare stated values on altruism/compassion/loyalty against behavioral
game allocations. Distinguish:
- Family-oriented social preferences (from collectivism items, family duty
  items)
- Stranger-oriented social preferences (from dictator/ultimatum/trust game
  behavior)
If there is a gap between these two (generous to family, selfish to
strangers), preserve it.

## Product-category predictions for the pricing study
Predict purchase likelihood patterns for these categories:
- Grocery staples (eggs, milk, bread, basic snacks)
- Premium branded items (Häagen-Dazs, premium chips)
- Bulk / party-size items (oversized formats)
- Indulgence / novelty items (candy, beer, ice cream)
- Household necessities (detergent, batteries)
For each, predict directional likelihood (more/less likely to purchase at
elevated prices) with reasoning from minimalism, income, values, and risk
preferences. Do not commit to specific per-product purchase decisions.

## Summary
Three to five sentences capturing the preference profile for
correspondence-rationality tasks (pricing, product choice, WTA/WTP).

TRANSCRIPT:
{transcript}"""


PROMPT_EVALUATION_V3 = """\
You are extracting the richest possible preference profile from a digital
twin survey transcript. Use every available signal — direct scales,
economic game allocations, lottery and intertemporal choice behavior, free-
response content, demographic context, and any behavioral gaps between
stated and revealed preferences. Infer aggressively but always state your
reasoning chain.

This file describes what the person values and what she'd pay for. Do NOT
include reasoning style or bias susceptibility.

Goal: produce the most complete, behaviorally predictive preference profile
for simulating this person on:
- Pricing study choices (40 products at varying prices)
- WTA/WTP gaps and endowment effects
- Value-laden policy items
- Correspondence-rationality tasks generally

CRITICAL EXTRACTION PRINCIPLES:

1. ACQUIESCENCE IN VALUES RATINGS — name it explicitly when present;
   extract only relative structure.

2. STRANGER-VS-FAMILY CONTEXT DEPENDENCE — preserve it; do not average
   family-collectivism with stranger-allocation behavior.

3. MINIMALISM AS DOMINANT PRICING SIGNAL — strong minimalism predicts
   purchase-skipping across categories even when income and pleasure-
   seeking scores would predict purchasing.

4. STATED-VS-REVEALED GAP IN SOCIAL PREFERENCES — if the person rates
   COMPASSION and ALTRUISM highly but offers selfish splits in economic
   games, the behavioral data wins for predicting actual allocation.

5. FREE-RESPONSE TEXT REVEALS TRUE DESIRES BETTER THAN SCALES —
   aspirational and actual-self descriptions often name what the person
   actually wants or is willing to trade off. Quote them when informative.

The survey covers: Big Five (44 items), Values (24 items), Minimalism (12
items), Empathy (20 items), consumerism/uniqueness scales, environmental
concern, economic games (dictator, ultimatum, trust), intertemporal choice,
lottery/risk tasks, word association chain, social desirability, full
demographics, and free-response items.

Output EXACTLY this format with no other text:

---EVALUATION_PROFILE---
Write prose in these sections, using markdown ## headers. No bullet lists in
the body prose.

## Budget constraint
Full synthesis: income, employment, household size, time poverty vs. money
poverty, free-response evidence of financial stress or abundance. Infer a
specific spending envelope for grocery-range purchases with reasoning.

## Values hierarchy
Direct ratings with acquiescence caveat. Relative structure. Cross-
reference with Big Five (achievement values × Conscientiousness, etc.),
minimalism (possession values × minimalism scores), and any behavioral
evidence of value enactment. Note whether stated values look achievement-
oriented, security-oriented, self-transcendence-oriented, or mixed.

## Minimalism and consumer orientation
Full minimalism profile plus inferred brand/category preferences. Predict:
- Premium vs. mainstream vs. budget brands
- Novelty vs. staple orientation
- Bulk/accumulation vs. just-in-time purchasing
- Lifestyle-brand shopping vs. functional shopping
State confidence for each inference.

## Risk preferences
Full analysis of lottery tasks including gain/loss asymmetry, certainty
preferences, and any Allais-like pattern. Cross-reference with
Conscientiousness and income. Infer how risk tolerance shapes product
choice (e.g., willingness to try unfamiliar brands).

## Intertemporal patience
Full analysis including exponential-vs-hyperbolic discounting signals if
detectable, Conscientiousness, and free-response evidence of planning vs.
procrastination. Distinguish stated patience from behavioral patience if
they diverge.

## Social preferences — detailed
Distinguish three context-dependent social preference zones:
- Toward strangers (from dictator, ultimatum, trust games) — use
  behavioral data, not stated values
- Toward family and close group (from collectivism items, family duty
  items, free-response) — use stated endorsement
- Toward abstract causes (environment, policy items, altruism ratings) —
  note stated-vs-revealed gap if economic games show selfish behavior
  while abstract-cause items show endorsement

If the person shows strong stranger-selfishness combined with strong
family-generosity, name this as an in-group cooperator profile.

## Policy and value-laden preferences
For any policy items in the transcript, state the person's direct
preferences. These predict myside bias direction on those items.

## Product-category predictions for the pricing study
Detailed predictions for purchase likelihood patterns:
- Grocery staples
- Premium branded items
- Bulk / party-size items
- Indulgence / novelty items
- Household necessities
- Specific brand tiers (generic, mainstream national brand, premium)
For each, give a directional prediction with confidence, reasoning from
the weighted combination of minimalism, income, values, risk tolerance,
and any behavioral evidence.

## Willingness-to-pay calibration
State the likely price ceiling for common grocery categories based on
income and minimalism. Give rough dollar ranges where reasonable.

## Self-report reliability for this file
Explicitly state how much to trust the person's stated values given
social desirability score, observed stated-vs-revealed gaps, and
acquiescence patterns.

## Summary
Four to six sentences capturing the complete preference profile. Include
the most predictive single feature for pricing behavior (often minimalism
strength, income, or the stranger-selfishness pattern).

TRANSCRIPT:
{transcript}"""


# ── Prompt lookup tables ───────────────────────────────────────────────────────
BACKGROUND_PROMPTS  = {"v1_direct": PROMPT_BACKGROUND_V1,
                       "v2_inferred": PROMPT_BACKGROUND_V2,
                       "v3_maximum": PROMPT_BACKGROUND_V3}
DECISION_PROMPTS    = {"v1_direct": PROMPT_DECISION_V1,
                       "v2_inferred": PROMPT_DECISION_V2,
                       "v3_maximum": PROMPT_DECISION_V3}
EVALUATION_PROMPTS  = {"v1_direct": PROMPT_EVALUATION_V1,
                       "v2_inferred": PROMPT_EVALUATION_V2,
                       "v3_maximum": PROMPT_EVALUATION_V3}

FILES = [
    ("background",         "background.txt",         BACKGROUND_PROMPTS,  "---BACKGROUND---"),
    ("decision_procedure", "decision_procedure.txt",  DECISION_PROMPTS,    "---DECISION_PROCEDURE---"),
    ("evaluation_profile", "evaluation_profile.txt",  EVALUATION_PROMPTS,  "---EVALUATION_PROFILE---"),
]


# ── Core logic ────────────────────────────────────────────────────────────────

def load_transcript(pid: str) -> str:
    for name in [f"pid_{pid}.txt", f"pid_{pid}_mega_persona.txt"]:
        path = PERSONA_DIR / name
        if path.exists():
            return path.read_text(encoding="utf-8")
    raise FileNotFoundError(f"Transcript not found for pid_{pid} in {PERSONA_DIR}")


def call_api(client: OpenAI, prompt: str, max_tokens: int) -> str:
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content.strip()


def parse_section(response_text: str, marker: str) -> str:
    if marker in response_text:
        return response_text.split(marker, 1)[1].strip()
    return response_text.strip()


def extract_for_persona(pid: str, versions: list, force: bool = False):
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY not set.")

    client     = OpenAI(api_key=api_key)
    transcript = load_transcript(pid)
    print(f"pid_{pid}: transcript loaded ({len(transcript)} chars)")

    for version in versions:
        output_dir = OUTPUT_BASE / f"pid_{pid}" / version
        max_tok    = MAX_TOKENS[version]

        # Skip if all 3 files already exist
        files_exist = all((output_dir / fname).exists()
                          for _, fname, _, _ in FILES)
        if not force and files_exist:
            print(f"  [{version}] already complete — skipping (--force to redo)")
            continue

        output_dir.mkdir(parents=True, exist_ok=True)

        for file_key, filename, prompt_dict, marker in FILES:
            out_path = output_dir / filename
            if not force and out_path.exists():
                print(f"  [{version}/{filename}] exists — skipping")
                continue

            prompt = prompt_dict[version].format(transcript=transcript)
            print(f"  [{version}/{filename}] calling API (max_tokens={max_tok})...")

            raw = call_api(client, prompt, max_tok)
            content = parse_section(raw, marker)

            # Retry with doubled tokens if Summary section is missing
            if "## Summary" not in content:
                print(f"  [{version}/{filename}] Summary missing — retrying with {max_tok*2} tokens")
                raw = call_api(client, prompt, min(max_tok * 2, 4000))
                content = parse_section(raw, marker)

            out_path.write_text(content, encoding="utf-8")
            print(f"  [{version}/{filename}] saved ({len(content)} chars)")

    print(f"Done: pid_{pid}")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract v2 skill profiles (background + decision + evaluation) for one persona.")
    parser.add_argument("--pid",     required=True, help="Persona ID (e.g. 1)")
    parser.add_argument("--version", choices=VERSIONS + ["all"], default="all",
                        help="Which version(s) to generate (default: all)")
    parser.add_argument("--force",   action="store_true",
                        help="Regenerate even if output files already exist")
    args = parser.parse_args()

    versions_to_run = VERSIONS if args.version == "all" else [args.version]
    extract_for_persona(args.pid, versions_to_run, force=args.force)
