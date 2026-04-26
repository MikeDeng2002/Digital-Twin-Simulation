# Extraction Prompts — Three-File Design (Single Version)
Twin-2K-500 Digital Twin Project

Three prompts, one per output file. Each prompt takes the full persona
transcript where `{transcript}` appears. No v1/v2/v3 split — one version
per file that extracts everything available, flags gaps explicitly, and
never invents content.

Module allocation is strict: each subsection of the survey appears in
exactly one file. See allocation table for the authoritative mapping.

---

## Output parsing

Each API response contains ONE section with ONE marker:

- Background prompt:          `---BACKGROUND---`
- Decision profile prompt:    `---DECISION_PROFILE---`
- Evaluation profile prompt:  `---EVALUATION_PROFILE---`

Parse by splitting on the marker and saving content to its respective `.txt` file.

```python
def parse_section(response_text: str, marker: str) -> str:
    if marker in response_text:
        return response_text.split(marker, 1)[1].strip()
    return response_text.strip()

background  = parse_section(response, "---BACKGROUND---")
decision    = parse_section(response, "---DECISION_PROFILE---")
evaluation  = parse_section(response, "---EVALUATION_PROFILE---")
```

---

## Design note — what each file is for

- **`background.txt`** — who they are: stable identity, personality traits,
  cognitive abilities, and emotional baseline. The "person" signal.
- **`decision_profile.txt`** — how they decide: reasoning style, time
  preferences, risk orientation, metacognitive accuracy, and decision
  search behavior.
- **`evaluation_profile.txt`** — what they value and how they evaluate
  outcomes: value priorities, fairness norms, social preferences, consumer
  orientation, and outcome weighting.

The three files map to a belief–desire–reasoning decomposition with strict
module boundaries and no redundancy across files.

---

## Token budget

- background:       `max_tokens=2000`
- decision_profile: `max_tokens=1800`
- evaluation_profile: `max_tokens=2200`

Retry with doubled `max_tokens` if the `## Summary` section is missing —
that is the most common truncation symptom.

---

## Output directory structure

```
text_simulation/skills/pid_{pid}/
├── background.txt
├── decision_profile.txt
└── evaluation_profile.txt
```

Three files per persona. One version only.

---

# BACKGROUND PROMPT

```
You are extracting a background profile for a digital twin from a survey
transcript. This file will be used by a downstream LLM to simulate how
this person responds to new questions. Accuracy and completeness are
critical.

This file covers: who the person is — their demographics, personality,
emotional baseline, cognitive abilities, and sense of self. It does NOT
include decision-making style (that is decision_profile.txt) or values and
preferences (that is evaluation_profile.txt).

MODULES THAT BELONG IN THIS FILE:
- A.2: All 14 demographic items
- A.3.1: Big Five (BFI) — 44 items
- A.3.2: Need for Cognition (NFC) — 18 items
- A.3.5: Empathy Scale (BES-A) — 20 items [if present in transcript]
- A.3.6: Green Values Scale — 6 items [if present in transcript]
- A.3.7: Social Desirability Scale — 13 items [reliability anchor only]
- A.3.9: Beck Anxiety Inventory (BAI) — 21 items
- A.3.10: Individualism vs. Collectivism — 16 items, 4 sub-scores
- A.3.11: Selves Questionnaire — 3 open-ended responses
- A.3.14: Beck Depression Inventory (BDI) — 22 items
- A.4.1: Cognitive Reflection Test (CRT) — 4 items
- A.4.2: Fluid Intelligence — 6 items
- A.4.3: Crystallized Intelligence — 20 items
- A.4.4: Syllogisms — 12 items
- A.4.7: Financial Literacy — 8 items
- A.4.8: Numeracy — 8 items
- A.4.9: Modus Ponens (Deductive Certainty) — 4 items
- A.4.11: Wason Selection Task — 1 item

Do NOT extract anything from: A.3.3, A.3.4, A.3.8, A.3.12, A.3.13,
A.3.15, A.3.16, A.3.17, A.3.18, A.3.19, A.4.5, A.4.6, A.4.10, A.5.x.
Those belong in decision_profile.txt or evaluation_profile.txt.

STRICT EXTRACTION RULES:

1. REPORT WHAT THE DATA SAYS. Do not soften, average, or normalize extreme
   scores. If a cognitive score is at the floor, say so explicitly. If
   anxiety is in the severe range, name it as severe.

2. COGNITIVE ERROR PATTERNS MATTER MORE THAN TOTAL SCORES. When a
   participant gives a wrong answer on a cognitive item, name the specific
   error type (e.g., "answered 27 on the hole question — volume-of-solid
   intuitive trap"). Do not just report the score.

3. SELVES QUESTIONNAIRE: quote the participant's own words verbatim for
   all three prompts (ideal self, ought self, actual self). Do not
   paraphrase. After quoting, note in one sentence whether the three selves
   are consistent or whether meaningful gaps exist between them.

4. SELF-REPORT RELIABILITY ANCHOR: the Social Desirability score
   calibrates how much to trust all other self-reports in this file. Report
   it first in the personality section and apply it as a caveat throughout.
   A high score (≥9/13) signals impression management. A low score (≤4/13)
   signals candid reporting.

5. SOMATIC-VS-SELF-REPORT CALIBRATION: if BAI or BDI shows moderate-to-
   severe loading but Big Five Neuroticism is low, flag this as a self-
   report reliability finding — not a contradiction to smooth over.

6. IF A MODULE IS ABSENT from the transcript, write one sentence saying so.
   Do not omit the section header.

7. Write each section as coherent prose. No bullet points inside sections.
   Use markdown ## headers exactly as specified below.

Output EXACTLY this format with no other text:

---BACKGROUND---

## Demographics
Report all 14 demographic items as stated: region, sex, age group,
education, race/ethnicity, citizenship, marital status, religion,
religious attendance, political party, household income, political
ideology, household size, employment status. Write as a single paragraph.

## Self-Report Reliability
Social Desirability Scale score: [X] / 13.
State in one sentence what this score implies for the trustworthiness of
self-reports throughout this file. Apply this caveat explicitly when
interpreting personality and emotional baseline sections below.

## Personality

### Big Five
Report all five dimension scores (extraversion, agreeableness,
conscientiousness, neuroticism, openness) on their 1–5 scales. Describe in
one paragraph what the profile reveals about interpersonal style, emotional
tendencies, and work habits. Explicitly name any score at or near a scale
extreme (≤1.5 or ≥4.5). Cross-reference neuroticism against the emotional
baseline section — flag discrepancies explicitly.

### Need for Cognition
Score: [X] / 5. One sentence: how much does this person enjoy and seek out
effortful thinking?

### Empathy (BES-A)
Score: [X] / 5. One sentence on tendency to share and recognize others'
emotions. If absent from transcript: "Empathy scale data not available in
this wave."

### Green Values
Score: [X] / 5. One sentence on environmental identity. If absent: "Green
values scale data not available in this wave."

### Individualism and Collectivism
Report all four sub-scores (horizontal individualism, vertical
individualism, horizontal collectivism, vertical collectivism) on 1–5
scales. One paragraph: orientation toward personal independence versus
group loyalty, and toward equality versus hierarchy within each domain.

## Emotional Baseline

### Beck Anxiety Inventory
Score: [X] / 63. Clinical benchmarks: 0–7 minimal, 8–15 mild, 16–25
moderate, 26–63 severe. State the clinical range and describe the symptom
pattern. Flag severe scores explicitly. Note whether the somatic symptom
pattern (physical symptoms: trembling, heart racing, sweating) versus
cognitive symptom pattern (fear of losing control, fear of dying, terrified)
is more pronounced, as this has implications for reliability of other
self-reports.

### Beck Depression Inventory
Score: [X] / 61. Clinical benchmarks: 0–9 minimal, 10–18 mild, 19–29
moderate, 30–63 severe. State the clinical range and describe the symptom
pattern. Flag severe scores explicitly. Note any specific item clusters
that stand out (e.g., physical symptoms only, cognitive distortion items,
motivational items).

## Sense of Self (Selves Questionnaire)
Quote verbatim:
- Ideal self (what they aspire to be): "[exact text]"
- Ought self (what they feel they should be): "[exact text]"
- Actual self (what they believe they currently are): "[exact text]"
Then in one sentence: are the three selves consistent, or are there
meaningful gaps (e.g., ideal contradicts actual, ought contradicts ideal)?

## Cognitive Ability Profile

For each test below, report: (a) raw score and scale maximum, (b) specific
items answered incorrectly with the error type named, (c) one sentence
interpretation.

### Cognitive Reflection Test (CRT)
Score: [X] / 4. For any wrong answer, name the intuitive trap the
participant fell into (e.g., "$0.10 → answered $1.00 — bat-ball anchoring
error").

### Fluid Intelligence
Score: [X] / 6. Note whether errors cluster on a particular item type
(matrix completion vs. cube rotation).

### Crystallized Intelligence
Score: [X] / 20. For any wrong answers, note whether the pattern suggests
a vocabulary gap or guessing (e.g., consistently choosing plausible-
sounding distractors).

### Syllogistic Reasoning
Score: [X] / 12. Note whether errors occur more on belief-biased items
(counterfactual or familiar content) versus abstract items. This indicates
whether reasoning is driven more by logic or prior belief.

### Financial Literacy
Score: [X] / 8. Name any incorrectly answered items — these reveal specific
knowledge gaps (e.g., missed the credit card minimum payment question →
does not know that minimum payments at 12% APR will never pay off the debt).

### Numeracy
Score: [X] / 8. Name any incorrectly answered items and the type of
numeric reasoning that failed (e.g., probability conversion, percentage
calculation).

### Modus Ponens (Deductive Certainty)
Score: [X] / 4. Note whether any item was answered "No" — this indicates
reluctance to accept logically valid conclusions that conflict with prior
beliefs.

### Wason Selection Task
Score: [X] / 4. State which cards were selected. If the participant chose
A and 3 (classic matching bias) instead of A and 7, name this explicitly.

## Cognitive Summary
One paragraph synthesizing the overall cognitive profile: where is this
participant strongest and weakest? Are there large discrepancies between
sub-scores that reveal a specific pattern (e.g., high CRT but low
crystallized intelligence suggests strong reflective reasoning but limited
verbal knowledge)? Are the cognitive ability scores consistent with the
financial literacy and numeracy results, or do they diverge?

## Summary
Four to six sentences capturing who this person is, their personality and
emotional state, and the shape of their cognitive abilities. Note the most
predictive single feature for downstream simulation (often a strong floor
or ceiling score, a notable stated-vs-somatic discrepancy, or a specific
cognitive error pattern).

TRANSCRIPT:
{transcript}
```

---

# DECISION PROFILE PROMPT

```
You are extracting a decision-making profile for a digital twin from a
survey transcript. This file will be used by a downstream LLM to simulate
how this person approaches choices under uncertainty, how they search for
options, and how they handle time trade-offs.

This file covers: how the person decides — their decision style, time
preferences, metacognitive accuracy, and spending mode. It does NOT include
who they are (that is background.txt) or what they value (that is
evaluation_profile.txt).

MODULES THAT BELONG IN THIS FILE:
- A.3.8: Conscientiousness Scale (IPIP, Wave 2) — 8 items, 9-pt scale
- A.3.12: Regulatory Focus Scale (RFS) — 10 items, 7-pt scale
- A.3.13: Tightwad–Spendthrift Scale — 4 questions across 3 scenarios
- A.3.18: Need for Cognitive Closure (NFCC) — 15 items, 5-pt scale
- A.3.19: Maximization Scale — 6 items, 5-pt scale
- A.4.5: Overconfidence — 1 item (predicted own score − actual score)
- A.4.6: Overplacement — 1 item (own prediction − average prediction)
- A.4.10: Forward Flow — 20-word free association chain [if present]
- A.5.3: Discount Rate — 3 multiple price lists
- A.5.4: Present Bias — 3 present discount price lists

Do NOT extract anything from: A.2, A.3.1–A.3.7, A.3.9–A.3.11,
A.3.14–A.3.17, A.4.1–A.4.4, A.4.7–A.4.9, A.4.11, A.5.1–A.5.2,
A.5.5–A.5.8. Those belong in background.txt or evaluation_profile.txt.

STRICT EXTRACTION RULES:

1. REPORT WHAT THE DATA SAYS. Do not soften extreme scores. A highly
   impatient discount rate is impatient — say so. A person who never
   switches in any price list is exhibiting extreme behavior — identify
   the direction (always prefers sooner, or always prefers later) and name
   it.

2. FOR PRICE LIST TASKS: the pipeline pre-computes annualized discount rates
   and present bias coefficients from switching points. Report the computed
   values and interpret their direction and magnitude. If no switching point
   was observed (participant chose one side consistently across all items in
   a list), say so explicitly and state which side they always chose.

3. REGULATORY FOCUS IS THE MOST PREDICTIVE DECISION-STYLE VARIABLE for
   framing effects and loss-related task behavior. Describe it thoroughly
   and note its directional implication for how this person will respond to
   gain-framed versus loss-framed choices.

4. NOTE INTERNAL CONSISTENCY. Where two measures are theoretically related
   (NFCC and Maximization; Regulatory Focus and Present Bias; Tightwad–
   Spendthrift and Discount Rate), check whether the scores are consistent
   or contradictory and state which.

5. FOR METACOGNITION: overconfidence and overplacement should be read
   together and cross-referenced against the cognitive ability scores in
   background.txt. A large overconfidence score combined with low cognitive
   ability scores is a more serious finding than the same overconfidence
   score with high ability scores.

6. IF A MODULE IS ABSENT from the transcript, write one sentence saying so.
   Do not omit the section header.

7. Write each section as coherent prose. No bullet points inside sections.
   Use markdown ## headers exactly as specified below.

Output EXACTLY this format with no other text:

---DECISION_PROFILE---

## Spending and Consumption Mode

### Tightwad–Spendthrift
Score: [X] (scale: 4 = extreme tightwad, 26 = extreme spendthrift,
midpoint ≈ 13). Describe in one paragraph this participant's default
spending mode. Reference the pattern across the four items: self-rating
slider, trouble-spending question, trouble-limiting question, and the mall
scenario. Note whether the items agree with each other or show within-
scale inconsistency.

## Decision Search and Closure

### Maximization vs. Satisficing
Score: [X] / 5. One sentence: does this participant search exhaustively for
the best possible option, or settle for good enough?

### Need for Cognitive Closure (NFCC)
Score: [X] / 5. One paragraph: tolerance for ambiguity, preference for
definite answers, decisiveness under uncertainty. Note whether high closure
need (≥4.0) or low closure need (≤2.0) shapes the expected decision speed
and commitment style.

### Regulatory Focus (RFS)
Score: [X] / 7 (higher = more promotion-focused; lower = more prevention-
focused; midpoint = 4). One paragraph: is this participant primarily
oriented toward achieving ideals and gains (promotion) or toward fulfilling
obligations and avoiding losses (prevention)? State the directional
implication explicitly: promotion-focused individuals are more responsive to
gain-framing and opportunity-framing; prevention-focused individuals are
more responsive to loss-framing and safety-framing. Note whether the RFS
score is consistent with the loss aversion coefficient in
evaluation_profile.txt.

### IPIP Conscientiousness (Wave 2)
Score: [X] / 8. One sentence: how organized, systematic, and efficient is
this participant in executing decisions? Note whether this score is
consistent with the Big Five Conscientiousness score in background.txt —
if they diverge, flag it.

## Metacognition

### Overconfidence
Score: [X] (predicted own score minus actual score on 42 cognitive items;
positive = overconfident, negative = underconfident, 0 = calibrated).
One sentence on direction and magnitude.

### Overplacement
Score: [X] (own predicted score minus predicted average score; positive =
believes they outperform peers, negative = believes they underperform).
One sentence on social comparison bias direction.

Cross-reference both scores against the cognitive ability profile in
background.txt: a large positive overconfidence combined with a low
ability score is a high-confidence finding of miscalibration. State this
explicitly if present.

### Forward Flow (Divergent Thinking)
Score: [X] (range approximately 0.70–0.95; higher = more semantically
distant word chains = more divergent / associative thinking).
One sentence on what this suggests about this participant's thinking style.
If absent from transcript: "Forward Flow data not available in this wave."

## Time Preferences

### Discount Rate
Annualized discount rate: [X].
If no switching point was observed across all three price lists, state
which side the participant always chose (always preferred sooner-smaller,
or always preferred later-larger) and what this implies about patience.
One paragraph: how patient or impatient is this participant when trading
off sooner versus later rewards? For reference, typical annualized discount
rates for US adults range from roughly 20% to 200%; situate this
participant accordingly.

### Present Bias
Present bias coefficient: [X] (positive = present-biased, prefers
immediate rewards more than the base discount rate predicts; negative =
future-biased; near zero = time-consistent).
One sentence: does this participant show present bias, and how strongly?
Note whether this is directionally consistent with the Tightwad–
Spendthrift score.

## Summary
Three to five sentences capturing the complete decision-making profile.
Name the most predictive single feature for simulating this person on
choice tasks (often the Regulatory Focus direction, the Discount Rate
magnitude, or the overconfidence calibration).

TRANSCRIPT:
{transcript}
```

---

# EVALUATION PROFILE PROMPT

```
You are extracting a preference and values profile for a digital twin from
a survey transcript. This file will be used by a downstream LLM to simulate
how this person evaluates outcomes, allocates resources, judges fairness,
and makes product and pricing decisions.

This file covers: what the person values and how they evaluate outcomes —
value priorities, consumer orientation, social preferences, fairness norms,
and risk/loss weighting. It does NOT include who they are (that is
background.txt) or how they decide (that is decision_profile.txt).

MODULES THAT BELONG IN THIS FILE:
- A.3.3: Agentic vs. Communal Values — 24 items, 9-pt scale
- A.3.4: Consumer Minimalism — 12 items, 5-pt scale
- A.3.15: Need for Uniqueness — 12 items, 5-pt scale
- A.3.16: Self-Monitoring Scale — 13 items, 6-pt scale
- A.3.17: Self-Concept Clarity (SCC) — 12 items, 5-pt scale
- A.5.1: Ultimatum Game — sender (1 item) + receiver (6 items)
- A.5.2: Mental Accounting — 4 binary-choice scenarios
- A.5.5: Risk Aversion — 3 gain lotteries (multiple price list)
- A.5.6: Loss Aversion — 3 loss lotteries + 1 mixed gamble
- A.5.7: Trust Game — sender (1 item) + receiver (5 items) + thought listing
- A.5.8: Dictator Game — sender (1 item) + thought listing

Do NOT extract anything from: A.2, A.3.1–A.3.2, A.3.5–A.3.14,
A.3.18–A.3.19, A.4.x, A.5.3–A.5.4. Those belong in background.txt or
decision_profile.txt.

STRICT EXTRACTION RULES:

1. REPORT WHAT THE DATA SAYS. Do not soften extreme behavioral choices.
   If the Dictator Game sender gives $0, write that they gave nothing — not
   that they were "somewhat less generous." If the Ultimatum sender
   offers $0, name it as the most selfish possible offer.

2. FOR MULTI-ITEM ECONOMIC GAMES: report the pattern across all items, not
   just the average. If the Trust Game receiver applies a consistent rule
   across all five amounts (e.g., always returns exactly one-third), name
   the rule explicitly. Do not compress item-level data into a summary
   statistic only.

3. PRESERVE THE STATED-VS-REVEALED GAP. If the stated Values scale (A.3.3)
   rates COMPASSION or ALTRUISM highly but the economic games show selfish
   allocations, surface this contradiction explicitly. Do NOT resolve it by
   favoring one source. Name both and let them coexist. The gap is
   predictive.

4. THOUGHT LISTINGS ARE HIGH-SIGNAL. Quote the participant's thoughts from
   the Trust Game and Dictator Game verbatim: Participant's thoughts:
   "[exact text]". Do not paraphrase. If thoughts reveal a specific
   reasoning frame (e.g., "fairness," "not greedy," "they can't reject
   anyway"), name that frame.

5. VALUES SCALE ACQUIESCENCE: the 24-item values scale uses a 9-pt scale
   and is vulnerable to acquiescence (most items rated 7–8). If the
   distribution is compressed, name this explicitly and extract only the
   relative structure — which values are uniquely high or uniquely low
   compared to this person's own average. Do not treat a compressed ranking
   as a clean hierarchy.

6. RISK AND LOSS AVERSION TOGETHER: always cross-reference risk aversion
   and loss aversion. If both are extreme in the same direction (very risk-
   averse and very loss-averse), name this as a consistent pattern. If they
   diverge, flag it.

7. IF A MODULE IS ABSENT from the transcript, write one sentence saying so.
   Do not omit the section header.

8. Write each section as coherent prose. No bullet points inside sections.
   Use markdown ## headers exactly as specified below.

Output EXACTLY this format with no other text:

---EVALUATION_PROFILE---

## Core Values

### Agentic vs. Communal Values
Agency score: [X] / 9. Communion score: [X] / 9.
Note first whether the ratings show acquiescence compression (most values
at 7–8). If so, extract only the relative structure. If not, treat the
absolute scores as informative.
Describe in one paragraph the balance between agentic values (wealth,
achievement, power, autonomy, status, recognition, competition) and
communal values (trust, loyalty, compassion, equality, harmony, humility,
civility). Name the 3–4 individual values rated highest and lowest if
item-level data is available.

## Consumer and Product Preferences

### Consumer Minimalism
Score: [X] / 5. One paragraph: does this participant actively restrict
possessions and prefer sparse aesthetics, or do they tend to accumulate?
Describe the pattern across the three sub-themes of the scale:
accumulation avoidance (items 1–4), aesthetic spareness (items 5–8), and
mindful curation (items 9–12). Note: strong minimalism (≥4.0) is the
dominant pricing-study signal — it predicts purchase-skipping across
categories even when income would predict purchasing.

### Need for Uniqueness
Score: [X] / 5. One sentence: how strongly does this participant seek
differentiation through consumption? Note whether the score is consistent
with minimalism — high uniqueness combined with high minimalism suggests
selective uniqueness (buying unusual items but few of them), while low
uniqueness with low minimalism suggests conventional accumulation.

## Social and Interpersonal Evaluation Standards

### Self-Monitoring
Score: [X] / 5. One sentence: how sensitive is this participant to others'
expectations, and how readily do they adjust their self-presentation?
Note the implication for social evaluation tasks — high self-monitors are
more responsive to social context cues.

### Self-Concept Clarity (SCC)
Score: [X] / 5. One sentence: how stable and consistent are this
participant's self-evaluation standards? Note: low SCC (≤2.5) predicts
more context-dependent and inconsistent judgments across tasks.

## Risk and Loss Evaluation

### Risk Aversion
CRRA coefficient: [X] (0 = risk-neutral; positive = risk-averse; negative
= risk-seeking; typical range −0.5 to 1.0 for US adults).
One paragraph: how does this participant evaluate uncertain outcomes in the
gain domain? If the participant never switched in any price list (always
chose one side), state this explicitly and name the direction.

### Loss Aversion
λ coefficient: [X] (1.0 = symmetric weighting; >1.0 = losses weighted more
than equivalent gains; <1.0 = gains weighted more; typical range 0.5 to
3.0 for US adults).
One paragraph: how asymmetric is this participant's weighting of losses
versus gains? Cross-reference with the Regulatory Focus score in
decision_profile.txt — a prevention-focused participant combined with high
loss aversion is a consistent pattern; a promotion-focused participant
combined with high loss aversion is a tension worth naming.

Note: cross-reference risk aversion and loss aversion. If both are extreme
in the same direction, name the pattern. If they diverge, flag it.

## Fairness and Social Preferences

### Mental Accounting
Score: [X]% choices consistent with mental accounting predictions (4
scenarios).
One paragraph: does this participant mentally segregate gains and losses
into separate accounts, or integrate them? High score (>75%) = strong
mental account separation (prefers two small wins over one combined win of
equal value). Low score (<25%) = strong integration. Describe what the
item-level pattern shows across the four scenarios (World Series lottery,
IRS letters, lottery-plus-rug, car damage-plus-pool win).

### Ultimatum Game
Sender: offered [X]% of $5 to the other person ([kept $X, sent $X]).
Receiver: state the accept/reject decision for each of the 6 possible
offers (keep $0/$1/$2/$3/$4/$5 for self) and identify the minimum
acceptable offer.
One paragraph: describe this participant's fairness norm in both giving and
receiving. For reference: a sender offer of 40–50% is typical; offers
below 20% are unusually low; offers of 50% or above signal strong fairness
motivation. A receiver who rejects offers below 20% enforces a meaningful
fairness floor. Note any asymmetry between sender generosity and receiver
tolerance — a person who offers little but accepts anything has a
different fairness structure than one who offers much and rejects unfair
offers.

### Trust Game
Sender: sent [X]% of $5 to the other person (amount tripled before
receiver sees it).
Receiver — report the return for each of the 5 possible received amounts:
  - Received $15 (sent $5 × 3): returned $[X] ([X]%)
  - Received $12 (sent $4 × 3): returned $[X] ([X]%)
  - Received $9  (sent $3 × 3): returned $[X] ([X]%)
  - Received $6  (sent $2 × 3): returned $[X] ([X]%)
  - Received $3  (sent $1 × 3): returned $[X] ([X]%)
If the participant applies a consistent return rule across amounts (e.g.,
always returns exactly one-third, or always returns a fixed dollar amount),
name the rule explicitly.
Participant's thoughts (sender): "[exact text from thought listing]"
Participant's thoughts (receiver): "[exact text from thought listing]"
One paragraph: (a) how much does this participant trust anonymous strangers
(sender behavior), (b) what is their reciprocity norm (receiver behavior),
(c) what do the thought listings reveal about the reasoning frame they
applied?

### Dictator Game
Sent [X]% of $5 ([kept $X, sent $X]). The other person had no power to
reject.
Participant's thoughts: "[exact text from thought listing]"
One paragraph: what does this allocation reveal about pure altruism or
fairness motivation when strategic incentives are absent? Compare to the
Ultimatum Game sender offer: if the Dictator amount is lower than the
Ultimatum offer, the difference reflects strategic giving (the person gives
more in Ultimatum because the recipient can punish them). If Dictator and
Ultimatum offers are equal, giving is driven by fairness preferences rather
than strategy.

## Self-Report Reliability for This File
State how much the stated Values scale (A.3.3) should be trusted relative
to behavioral game data (A.5.1, A.5.7, A.5.8), given: (a) the Social
Desirability score from background.txt, (b) any observed stated-vs-
revealed gaps (e.g., high stated altruism combined with low Dictator
giving), and (c) acquiescence patterns in the Values ratings. This
weighting tells the downstream simulation how much to lean on stated values
versus behavioral data when they conflict.

## Summary
Four to six sentences capturing the complete preference and values profile.
Include: the most predictive single feature for pricing behavior (usually
minimalism strength, income, or the stranger-selfishness pattern), the
dominant fairness norm revealed by the economic games, and any major
stated-vs-revealed gap that the simulation must navigate.

TRANSCRIPT:
{transcript}
```

---

# Integration notes for `extract_skills.py`

## Python variable names

```python
PROMPT_BACKGROUND    = """..."""
PROMPT_DECISION      = """..."""
PROMPT_EVALUATION    = """..."""
```

## Calls per persona

3 API calls per persona, one per file.

## Output parsing

```python
def parse_section(response_text: str, marker: str) -> str:
    if marker in response_text:
        return response_text.split(marker, 1)[1].strip()
    return response_text.strip()

background  = parse_section(response_bg,   "---BACKGROUND---")
decision    = parse_section(response_dp,   "---DECISION_PROFILE---")
evaluation  = parse_section(response_ep,   "---EVALUATION_PROFILE---")
```

## Output directory structure

```
text_simulation/skills/pid_{pid}/
├── background.txt
├── decision_profile.txt
└── evaluation_profile.txt
```
