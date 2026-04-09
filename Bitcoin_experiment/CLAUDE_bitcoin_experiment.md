e# Bitcoin Celebrity Influence Experiment

## Project overview

This project studies how celebrity opinions shift a center persona's
Bitcoin price prediction. It runs two parallel experiment types:

- **Type A — Number only**: center persona outputs a single dollar
  prediction for Bitcoin's price at end of 2025.
- **Type B — Number + reasoning**: center persona outputs a prediction
  AND a short paragraph explaining their reasoning, including any
  sources or ideas they would cite.

The key research question across both types: does knowing *who* said
a number (and seeing their reasoning) shift the center persona's
prediction more than seeing an anonymous number alone? And does the
shift vary by persona background and celebrity identity?

---

## Experiment types

### Type A — Predicted price only

Center persona sees neighbor predictions (numbers only) and outputs:
- A single dollar number

Simple, fast, easy to regress. Used to estimate w_social.

### Type B — Predicted price + reasoning

Center persona sees neighbor predictions WITH reasoning (numbers +
explanations + cited sources) and outputs:
- A single dollar number
- A short paragraph (3–5 sentences) explaining their reasoning
- Any sources, articles, X posts, or ideas they would cite to support
  their view

This captures the *mechanism* of influence — not just whether the
persona moved, but why. The reasoning text is analyzed to detect:
- Which celebrity arguments were absorbed
- Which sources were cited
- Whether the persona's own background knowledge surfaced
- Whether reasoning quality differs across skill versions

---

## Data sources

### Persona data
- Path: `text_simulation/text_personas/pid_{pid}.txt`
- Content: ~450 survey Q&A per persona (waves 1–3)
- Skills path: `text_simulation/skills/pid_{pid}/`
  - `v1_direct/` — direct evidence only
  - `v2_inferred/` — direct + demographic inference
  - `v3_maximum/` — all signals, maximum inference
  - Each version has: `background.txt`, `tools.txt`, `decision_procedure.txt`

### Celebrity data
- Path: `data/celebrities/`
- One folder per celebrity: `data/celebrities/{slug}/`
  - `profile.txt` — who they are, known stance, domain authority
  - `quotes.txt` — real documented quotes from X, interviews, public statements
  - `prediction.txt` — cached LLM-generated Bitcoin price prediction
  - `reasoning.txt` — cached LLM-generated reasoning for their prediction

Celebrity slugs:
  - `michael_saylor`
  - `elon_musk`
  - `cathie_wood`
  - `jack_dorsey`
  - `larry_fink`
  - `donald_trump`
  - `warren_buffett`
  - `peter_schiff`
  - `jamie_dimon`
  - `nouriel_roubini`

### X / social media data (optional enrichment)
- Path: `data/celebrities/{slug}/x_posts.txt`
- Content: real X posts relevant to Bitcoin predictions
- Used to ground celebrity reasoning in actual public statements
- If X API is available, fetch fresh posts before running experiment
- If not, use cached posts in the file

---

## Celebrity roster and documented positions

### Strong supporters (bullish)

**Michael Saylor** — Executive Chairman, MicroStrategy
- Stance: Extreme bull. Bitcoin maximalist.
- Price target: $500,000–$1,000,000+
- Key quotes:
  - "Bitcoin is digital energy — the apex property of the human race."
  - "Every dip is a buying opportunity. MicroStrategy will never sell."
- Evidence: MicroStrategy holds 500,000+ BTC. Tweets daily. Published
  long-term model projecting $13M per BTC by 2045.

**Elon Musk** — CEO, Tesla & SpaceX
- Stance: Volatile bull. Owns BTC personally. Moves markets with tweets.
- Price target: $150,000–$300,000
- Key quotes:
  - "I still own Bitcoin and won't sell."
  - Tesla bought $1.5B in BTC in February 2021.
- Evidence: Market-moving tweet history. Sold 75% of Tesla BTC in 2022
  for liquidity. Still holds personally. Prefers Dogecoin for transactions.

**Cathie Wood** — CEO, ARK Invest
- Stance: Strong institutional bull. Detailed public price model.
- Price target: $500,000 (2025), $1,500,000 (2030)
- Key quotes:
  - "Bitcoin is the most compelling store of value the world has ever seen."
  - "ARK's model suggests $1.5M is conservative for 2030."
- Evidence: Runs Bitcoin ETF. Buys on every dip. Published model public.

**Jack Dorsey** — CEO, Block (formerly Square)
- Stance: Bitcoin maximalist. Only crypto he supports.
- Price target: $150,000–$250,000
- Key quotes:
  - "Bitcoin changes everything. Nothing is more important to work on."
  - "Bitcoin will be the internet's native currency."
- Evidence: Block integrates BTC across all products. Funds open-source
  Bitcoin development. Publicly rejects all other crypto.

### Institutional moderates (cautiously positive)

**Larry Fink** — CEO, BlackRock
- Stance: Converted moderate. Former skeptic, now institutional legitimizer.
- Price target: $100,000–$200,000
- Key quotes:
  - "It's not much different from what gold has represented for thousands of years."
  - "I view Bitcoin as a legitimate financial tool."
- Evidence: Called it money laundering in 2017. Now runs world's largest
  BTC ETF (IBIT). Publicly changed position after personal research in 2024.

**Donald Trump** — US President
- Stance: Converted supporter. Created US strategic Bitcoin reserve in 2025.
- Price target: $150,000–$250,000
- Key quotes:
  - "I want the US to be the crypto capital of the planet."
  - "We will never let anyone shut down Bitcoin."
- Evidence: Called crypto a scam in 2021. Fully pro-crypto by 2024.
  Created national strategic Bitcoin reserve. Strong political signal.

### Strong critics (bearish)

**Warren Buffett** — CEO, Berkshire Hathaway
- Stance: Persistent critic. Value investor. Consistent 10+ year skeptic.
- Price target: $10,000–$30,000 (implies it will fall from current levels)
- Key quotes:
  - "Rat poison squared."
  - "I wouldn't pay $25 for all the Bitcoin in the world."
  - "It produces nothing. It's a gambling device."
- Evidence: Has never purchased Bitcoin. Consistent position since 2014.
  Multiple death declarations. On the 431-count Bitcoin obituary list.

**Peter Schiff** — CEO, Euro Pacific Capital
- Stance: Most vocal critic. Gold maximalist. 18 Bitcoin death declarations.
- Price target: Near $0 (predicts collapse)
- Key quotes:
  - "Bitcoin was born out of the 2008 crisis. The 2025 crisis will kill it."
  - "The only good digital currency is one backed by gold."
- Evidence: Leads the nocoiner list. Participated in Bitcoin 2025
  conference to present contrarian views. Never changed position despite
  consistently wrong price predictions.

**Jamie Dimon** — CEO, JPMorgan Chase
- Stance: Institutional critic publicly, profits from it institutionally.
- Price target: $20,000–$50,000 (implies significant downside)
- Key quotes:
  - "Bitcoin is a pet rock."
  - "I've always been deeply opposed to crypto, Bitcoin is a fraud."
- Evidence: Called it a fraud in 2017. JPMorgan simultaneously provides
  BTC ETF custody services. Stated vs actual behavior gap is scientifically
  interesting for the simulation.

**Nouriel Roubini** — Economist ("Dr. Doom")
- Stance: Academic critic. Testified to US Senate against crypto.
- Price target: Near $0 (predicts collapse to intrinsic value = zero)
- Key quotes:
  - "Crypto is the mother of all scams and bubbles."
  - "Bitcoin has no intrinsic value — it will go to zero."
- Evidence: Senate testimony. Predicted Bitcoin to zero repeatedly.
  Academic credibility distinguishes him from Schiff's ideological angle.

---

## Network structure

### Neighbor composition (default: mixed condition)
- Total neighbors: k = 5
- Celebrity neighbors: 2 (1 bull + 1 bear, randomly drawn from roster)
- Random 2k-500 neighbors: 3 (numbers drawn from dataset distribution)

### Neighbor conditions (tunable)
- `baseline`: 0 celebrity, 5 random — replicates original Exp 2
- `celebrity_only`: 5 celebrity (mixed bull/bear), 0 random
- `mixed` (default): 2 celebrity + 3 random
- `celebrity_heavy`: 4 celebrity + 1 random

### Celebrity number caching
Celebrity numbers are generated ONCE per celebrity and cached in
`data/celebrities/{slug}/prediction.txt`. They do not change across
trials. Only random neighbor numbers are redrawn each trial.

---

## Prompts

### Step 1: Generate celebrity prediction (run once per celebrity)

```
You are simulating {CELEBRITY_NAME}'s public prediction for Bitcoin's
price at the end of 2025.

Here is what {CELEBRITY_NAME} has actually said publicly:

{QUOTES}

Based on their documented positions above, what single dollar number
would {CELEBRITY_NAME} predict for Bitcoin's price at end of 2025?

Respond with a single number only. No explanation.
```

### Step 2: Generate celebrity reasoning (run once per celebrity)

```
You are simulating {CELEBRITY_NAME}'s explanation for their Bitcoin
price prediction of ${PREDICTED_PRICE}.

Here is what {CELEBRITY_NAME} has actually said publicly:

{QUOTES}

Write 3–5 sentences explaining WHY {CELEBRITY_NAME} would predict
${PREDICTED_PRICE}. Use their actual documented arguments.
Include any specific articles, data sources, or frameworks they
typically cite. Write in first person as {CELEBRITY_NAME}.
```

### Step 3A: Center persona prediction — Type A (number only)

```
{PERSONA_SYSTEM_PROMPT}

You are answering this question as the person described above.

Question: What do you think Bitcoin's price will be at the end of 2025?

Before answering, here is what people in your network have predicted:

{NEIGHBOR_BLOCK}

Given your own beliefs and what you have heard, what is your prediction?

Respond with a single dollar number only.
```

### Step 3B: Center persona prediction — Type B (number + reasoning)

```
{PERSONA_SYSTEM_PROMPT}

You are answering this question as the person described above.

Question: What do you think Bitcoin's price will be at the end of 2025?

Before answering, here is what people in your network have predicted:

{NEIGHBOR_BLOCK}

Given your own beliefs and what you have heard, provide:

1. Your prediction: a single dollar number
2. Your reasoning: 3–5 sentences explaining WHY you predict this
   price. Mention any specific sources, articles, news, or ideas
   that inform your view. Be specific to who you are — your
   background, what you typically read, and how you make decisions
   under uncertainty.

Format your response exactly as:
PREDICTION: $[number]
REASONING: [3-5 sentences]
```

### Neighbor block format — Type A (numbers only)

```
- {NAME or "Person (age X, background)"}: ${NUMBER}
- {NAME or "Person (age X, background)"}: ${NUMBER}
- {NAME or "Person (age X, background)"}: ${NUMBER}
- {NAME or "Person (age X, background)"}: ${NUMBER}
- {NAME or "Person (age X, background)"}: ${NUMBER}
```

### Neighbor block format — Type B (numbers + reasoning)

```
- {NAME or "Person (age X, background)"}: ${NUMBER}
  Reasoning: "{SHORT REASONING — 1-2 sentences}"

- {NAME or "Person (age X, background)"}: ${NUMBER}
  Reasoning: "{SHORT REASONING — 1-2 sentences}"

[... repeat for all neighbors]
```

### Persona system prompt — raw text condition

```
You are simulating a specific person based on their survey responses.
Here is their full survey transcript:

{FULL_QA_TRANSCRIPT}
```

### Persona system prompt — skill condition (v1/v2/v3)

```
You are simulating a specific person. Here is their profile:

## Background knowledge
{BACKGROUND_TXT}

## Information sources they use
{TOOLS_TXT}

## How they make decisions
{DECISION_PROCEDURE_TXT}
```

---

## Output parsing

### Type A output
Response is a single number. Strip $ signs and commas. Convert to float.
Store as `prediction_price`.

### Type B output
Parse by splitting on "PREDICTION:" and "REASONING:" markers.
- `prediction_price`: float extracted from PREDICTION line
- `prediction_reasoning`: string from REASONING line

Store both in the results JSON.

---

## File structure

```
bitcoin_experiment/
├── CLAUDE.md                          ← this file
├── data/
│   ├── celebrities/
│   │   ├── michael_saylor/
│   │   │   ├── profile.txt
│   │   │   ├── quotes.txt
│   │   │   ├── x_posts.txt            ← real X posts if available
│   │   │   ├── prediction.txt         ← cached: single number
│   │   │   └── reasoning.txt          ← cached: 3-5 sentence explanation
│   │   ├── elon_musk/
│   │   ├── cathie_wood/
│   │   ├── jack_dorsey/
│   │   ├── larry_fink/
│   │   ├── donald_trump/
│   │   ├── warren_buffett/
│   │   ├── peter_schiff/
│   │   ├── jamie_dimon/
│   │   └── nouriel_roubini/
│   └── neighbor_draws/
│       └── bitcoin_2025/
│           └── pid_{pid}_trial_{n}.json  ← cached random neighbor draws
│
├── results/
│   ├── type_a/                        ← number only
│   │   ├── raw_text/
│   │   │   └── pid_{pid}.json
│   │   ├── v1_direct/
│   │   ├── v2_inferred/
│   │   └── v3_maximum/
│   └── type_b/                        ← number + reasoning
│       ├── raw_text/
│       ├── v1_direct/
│       ├── v2_inferred/
│       └── v3_maximum/
│
├── analysis/
│   ├── w_social_regression.py         ← estimate influence weights
│   ├── reasoning_analysis.py          ← analyze Type B reasoning text
│   └── neural_network.py              ← predict w_social from features
│
└── scripts/
    ├── 01_generate_celebrity_data.py  ← Step 1: generate + cache celebrity predictions
    ├── 02_run_type_a.py               ← Step 2: run Type A experiment
    ├── 03_run_type_b.py               ← Step 3: run Type B experiment
    └── 04_analyze_results.py          ← Step 4: compute w_social, compare types
```

---

## Result JSON format

Each trial result is stored as JSON:

```json
{
  "pid": 1,
  "trial": 23,
  "representation": "v2_inferred",
  "experiment_type": "B",
  "condition": "mixed",
  "question": "bitcoin_price_2025",
  "neighbors": [
    {
      "type": "celebrity",
      "name": "Michael Saylor",
      "prediction": 750000,
      "reasoning": "Bitcoin is the apex asset of the human race..."
    },
    {
      "type": "celebrity",
      "name": "Warren Buffett",
      "prediction": 18000,
      "reasoning": "Bitcoin produces nothing and has no intrinsic value..."
    },
    {
      "type": "random",
      "pid": 847,
      "age": 34,
      "background": "Midwest, moderate politics",
      "prediction": 62000,
      "reasoning": null
    }
  ],
  "neighbor_mean": 187600,
  "celebrity_mean": 384000,
  "random_mean": 62000,
  "center_prediction": 95000,
  "center_reasoning": "I think Bitcoin has real value but the bulls are too optimistic...",
  "baseline_prediction": 78000
}
```

---

## Analysis pipeline

### Step 1: Estimate influence weights (per persona)

For each persona, across 50 trials:

```
center_prediction = w_baseline × baseline
                  + w_celeb × celebrity_mean
                  + w_random × random_mean
                  + error
```

Run OLS regression. Output per-persona:
- `w_baseline`, `w_celeb`, `w_random`
- `celebrity_premium` = w_celeb − w_random
- Standard errors and confidence intervals

### Step 2: Compare Type A vs Type B

For the same persona and trial conditions:
- Does Type B (reasoning included) produce higher or lower w_celeb
  than Type A (numbers only)?
- Does seeing celebrity reasoning make the persona MORE susceptible
  (argument absorption) or LESS susceptible (counter-argument triggered)?
- Which celebrities produce the largest A→B difference?

### Step 3: Analyze reasoning text (Type B only)

For each center persona reasoning output:
- Keyword detection: did they mention a celebrity by name?
- Source citation: did they cite any specific articles, data, or platforms?
- Sentiment: bullish, bearish, or neutral relative to their baseline?
- Argument absorption score: how many celebrity arguments appear in
  the persona's reasoning that did not appear in their baseline?

Use simple keyword matching first, then LLM-based analysis if needed:
```
Analyze this reasoning paragraph. Did the person absorb arguments from
[CELEBRITY] or [CELEBRITY]? What sources did they cite? Score their
argument absorption from 0 (none) to 1 (strong).

REASONING: {center_reasoning}
CELEBRITY ARGUMENTS PRESENTED: {celebrity_reasoning_texts}
```

### Step 4: Cross-representation comparison

For each condition, compare w_celeb across:
- Raw text vs v1_direct vs v2_inferred vs v3_maximum

Key hypothesis: v3_maximum produces higher celebrity-specific influence
(because richer tools.md captures who the persona listens to), while
raw text produces more generic anchoring.

---

## Running the experiment

### Quick start (5 personas, mixed condition, both types)

```
Run the Bitcoin celebrity influence experiment for pid 1 through 5.
Use the mixed neighbor condition (2 celebrity + 3 random).
Run both Type A and Type B.
Use v2_inferred skill representation.
Store results in results/type_a/ and results/type_b/.
Print a summary when done.
```

### Full run (all 2000 personas)

```
Run the full Bitcoin experiment for all 2000 personas.
Use the Batch API to minimize cost and avoid rate limits.
Run mixed condition only. Both Type A and Type B.
All four representations: raw_text, v1_direct, v2_inferred, v3_maximum.
Log progress every 100 personas. Resume from checkpoint if interrupted.
```

### Celebrity data generation (run first, once only)

```
Generate celebrity predictions and reasoning for all 10 celebrities.
Use the prompts in CLAUDE.md. Load quotes from data/celebrities/{slug}/quotes.txt.
Save prediction to data/celebrities/{slug}/prediction.txt.
Save reasoning to data/celebrities/{slug}/reasoning.txt.
Print each celebrity's prediction when done so I can review before running the main experiment.
```

### Analysis

```
Run the w_social regression analysis for all completed Type A results.
Compute w_baseline, w_celeb, w_random per persona.
Compare across representation methods.
Save results to analysis/w_social_results.csv.
Print top 10 personas with highest celebrity premium (w_celeb - w_random).
```

---

## Key research questions to answer

1. **Celebrity premium**: Is w_celeb > w_random across most personas?
   Does knowing it's Saylor or Buffett matter beyond the number itself?

2. **Directional asymmetry**: Do bull celebrities pull harder than bear
   celebrities, or vice versa? Does this depend on the persona's prior?

3. **Authority matching**: Do finance-expert personas defer more to
   Buffett/Fink (institutional authority) while tech-oriented personas
   defer more to Musk/Dorsey? Does domain expertise predict celebrity-
   specific influence?

4. **Type A vs Type B**: Does adding reasoning to the neighbor messages
   increase or decrease w_celeb? Is argument quality (Fink's reasoned
   conversion) more influential than raw assertion (Schiff's dogmatism)?

5. **Skill version effect**: Does v3_maximum produce more authentic
   celebrity-specific influence patterns than raw text? If yes, the
   skill method is capturing something behaviorally real about who
   each persona listens to.

6. **Reasoning quality**: In Type B, do personas with higher need for
   cognition produce richer reasoning that more explicitly engages with
   the celebrity arguments? Does their reasoning quality differ by
   celebrity match?

---

## Notes for Claude Code

- Always load celebrity predictions from cache before generating new ones.
  Only regenerate if the cache file is missing or explicitly requested.

- The 50 trials per persona use fixed random seeds for reproducibility.
  Seed = pid × 1000 + trial_number. Draw random neighbor PIDs and their
  answers from the dataset distribution using this seed.

- For Type B reasoning analysis, run LLM-based scoring in batches of
  100 to manage cost. Use the cheapest capable model (gpt-4.1-mini or
  claude-haiku) for scoring since it's a simple classification task.

- If X API access is available, fetch fresh posts for each celebrity
  before generating their predictions. Store in x_posts.txt and
  incorporate into the quotes context.

- Log every API call with timestamp, model, tokens used, and cost
  estimate to experiments.log for cost tracking.

- When printing progress, show: persona ID, trial number, predicted
  price, and (for Type B) first 100 chars of reasoning.
