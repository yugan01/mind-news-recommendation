# Slide 1: Neural Recommendation Models
**CP4285 Modern Recommendation Systems — Week 04**
CP4285 Instruction Team
01 Sep 2026

Scan for the slides: `soc-n.us/cp4285-t2610-w04`

# Slide 2: 01 From Evaluation to Model

# Slide 3: WEEK 03 RECAP

# Slide 4: LAST WEEK PRE-LECTURE EXERCISE
**What Does a Missing Interaction Mean?**
Canvas discussion board · post before Mon, 31 Aug 23:59 SGT

Post one concise response; then reply constructively to one classmate.
> A user has never clicked a recommended item. Give two plausible interpretations: one about preference and one about exposure or the interface.

Neural recommenders can learn richer interaction patterns, but they still learn only from the evidence the system records.

# Slide 5: LAST WEEK PRE-LECTURE EXERCISE · CANVAS VOICES
**Three careful readings of a missing click**

# Slide 6: PRE-LECTURE SYNTHESIS
**Capacity is not observability**

* **What a non-click might mean:**
  Disinterest, no exposure, poor placement, wrong timing, unavailable inventory, or inattention.
* **What today's model can change:**
  How it combines the information that was recorded - not whether the log distinguishes those explanations.

# Slide 7: Week 04: Neural Recommendation Models
Today: compare latent-factor and neural interaction functions, reason about implicit-feedback training, and ask when a model’s score is not yet an explanation.

> **Note**
> A neural recommender does not replace the evaluation contract. It replaces a fixed interaction rule with a learned function of the representations and evidence available to it.

# Slide 8: Learning outcomes
* **Build:** Describe a neural recommender as embeddings, an interaction function, and a training objective.
* **Compare:** Contrast a latent-factor dot product with a learned non-linear interaction function.
* **Interpret:** Identify what implicit feedback supports—and what an unobserved interaction cannot prove.
* **Critique:** Assess whether a performance claim, explanation, and safeguard are aligned.

# Slide 9: Transformer recap: query, key, value, attention
**Scaled dot-product attention**
input embeddings `X`
↓
`Q = XW_Q`, `K = XW_K`, `V = XW_V`
`QK^T / √d_k` (scaled compatibility scores)
↓
`H = softmax(QK^T / √d_k) V` (contextualised output embeddings)

**Compatibility**
`s_j = q^T k_j`
Queries score keys. Higher = closer match.

**Weighted information**
`h = ∑_j α_j v_j`, `α = softmax(s)`
Softmax gives weights; combine values.

> **To think about**
> Week 05 foreshadowing: Transformers natively deal with ordered interactions, which we don’t touch this week.

# Slide 10: Attention as a user–item matching mechanism
`user embedding e_u` → `user query q_u` ↔ `item key/value (k_i, v_i)`

**Learned projections**
`q_u = W_Q e_u`
`k_i = W_K e_i`, `v_i = W_V e_i`
The user supplies the query; each candidate item supplies a key and value.

**One user–item compatibility**
`q_u^T k_i`
A higher score means an item key better matches a user query. Across candidates, softmax turns scores into weights for values.

For this bridge, user = query and item = key/value. In a full Transformer, these are operational roles—not permanent identities.

# Slide 11: 02 From Matrix Factorisation to Neural CF

# Slide 12: Matrix factorisation: a dot product is an interaction
**Learn latent representations**
`p_u ∈ ℝ^d`, `q_i ∈ ℝ^d`
The axes are learned embeddings of one-hot users and items.

**Choose a fixed interaction**
`r^_ui = p_u^T q_i + b_u + b_i`
A dot product rewards aligned representations.

> **Note**
> Use `f_θ(p_u, q_i)` to capture combinations a dot product misses. It needs more data, compute, and overfitting control.

# Slide 13: Worked MF score: one user, one candidate
**Given learned coordinates**
`p_u = [0.8, 0.2]` (User: high on coordinate 1; low on coordinate 2.)
`q_i = [0.9, 0.1]` (Item: also high on coordinate 1; low on coordinate 2.)
These are learned coordinates, not named traits.

**Multiply, then add**
`r^_ui = 0.8(0.9) + 0.2(0.1) = 0.72 + 0.02 = 0.74`
* 0.72: dominant alignment on coordinate 1.
* 0.02: small alignment on coordinate 2.
* 0.74: relative ranking score—not probability or explanation.

# Slide 14: Key Point: NeuMF combines GMF + MLP
* **GMF**: `p_u ⊙ q_i`. Weights coordinate-wise matches, extending the dot-product path.
* **NeuMF**: 
  1 · retain the GMF path
  2 · add the MLP path
  3 · concatenate both paths.
* **MLP**: `f(p_u, q_i)`. Mixes all coordinates through learned non-linear layers.

# Slide 15: Worked interaction functions: same inputs, three scores
Shared toy inputs: `p_u = [.8, .2]`, `q_i = [.9, .1]`

**1 · GMF: Element-wise match**
`[.8, .2] ⊙ [.9, .1] = [.72, .02]`
Read out: `s_G = [1, 1]^T [.72, .02] = .74`

**2 · MLP: Build a hidden representation**
`h = ReLU([.8 + .1, .2 + .9]) = [.9, 1.1]`
Read out: `s_M = [.2, .4]^T [.9, 1.1] = .62`

**3 · NeuMF: Concatenate both paths**
`z = [.72, .02; .9, 1.1]`
Read out: `s_N = [.5, .5, .1, .1]^T z = .36 + .01 + .09 + .11 = .57`

> **Note**
> Navy vectors are toy readout weights. Each dot product produces one scalar: `s_G`, `s_M`, or `s_N`.

# Slide 16: Code lens: where each NeuMF component lives
**Annotated forward pass**
```python
p_g, q_g = user_gmf[u], item_gmf[i]
g = p_g * q_g # GMF path

p_m, q_m = user_mlp[u], item_mlp[i]
h = mlp(concat(p_m, q_m)) # MLP path

r_ui = readout(concat(g, h)) # scalar score
```
`GMF features g + MLP features h → score r_ui`

**Two representation paths**
Separate GMF/MLP embeddings create `g` and `h` on distinct paths.

**One final readout**
`r^_ui = w^T [g; h] + b`
`w` maps concatenated `g`, `h` to one ranking score.

# Slide 17: 03 Training for Ranked Recommendations

# Slide 18: Implicit feedback creates a sampling decision
**Observed positive**
Aisha clicked Botanic Gardens.
A signal of possible interest—not a rating of every alternative.

**Unobserved candidate**
Aisha did not click Riverside Food Court.
It may have been unseen, unavailable, badly timed, or unappealing.

`logged click → positive + sampled unobserved item → training comparison`

**Negative sampling is an assumption about missing feedback—not a discovered dislike.**

# Slide 19: Pointwise prediction loss vs. pairwise ranking loss
**Pointwise: predict one interaction label**
`L_pt = -[r_ui log σ(r^_ui) + (1 - r_ui) log(1 - σ(r^_ui))]`
Training claim: “Did Aisha interact with City Gardens?”

**Pairwise: predict one ordering**
`L_BPR = -log σ(r^(u, i^+) - r^(u, i^-))`
Training claim: “Should City Gardens rank above Riverside Food Court?”

The figure makes the choice visible: **pointwise calibrates a pair; pairwise orders a comparison.**
*(Rendle, S., Freudenthaler, C., Gantner, Z., & Schmidt-Thieme, L. (2009). BPR: Bayesian personalized ranking from implicit feedback.)*

# Slide 20: BPR example: make the observed item rank higher
**BPR = Bayesian Personalised Ranking.** It is a personalised ranking loss for tuning model parameters so that each user’s observed items score above sampled alternatives.

**Triplet:** `u` = Aisha; `i^+` = Botanic Gardens (clicked); `i^-` = Riverside Food Court (sampled).
`s(u, i^+) = 1.20`, `s(u, i^-) = 0.40`

**1 · Margin**
`Δ = 1.20 - 0.40 = 0.80` (Positive above sampled alternative.)

**2 · Pairwise probability**
`σ(Δ) = σ(0.80) = 0.690` (Model confidence in this ordering.)

**3 · Loss and update**
`L_BPR = -ln(0.690) = 0.371`
`∂L/∂Δ = -0.310`: gradient descent increases the positive score and/or lowers the sampled score.

**BPR ranks the click above this sampled comparison; it does not call the unclicked item “bad.”**

# Slide 21: Which negative is defensible?
Aisha clicked Botanic Gardens. Which sampled comparison is most defensible?

1. A restaurant that was never shown.
2. One shown after it closed.
3. One shown beside Botanic Gardens at the same time.

Groups of 3–4: choose one; give one reason and one remaining uncertainty. Prepare two sentences.

▶ Start Timer (03:00)

# Slide 22: Bandit Time: What does BPR optimise?
For Aisha’s clicked Botanic Gardens and sampled Riverside Food Court, which update most directly reduces the BPR loss?

* **A.** Widen the margin: raise Botanic Gardens’ score and/or lower Riverside Food Court’s score for Aisha.
* **B.** Raise both item scores by the same amount because both may be relevant.
* **C.** Optimise Botanic Gardens’ standalone click probability without using the sampled alternative.
* **D.** Set Riverside Food Court to a zero target for everyone who has not clicked it.

# Slide 23: Break — 5 minutes
WHILE YOU PAUSE: ☕ TriRank
TriRank extends the user–item graph with review aspects, modelling users, items, and the aspects mentioned in reviews as a tripartite graph. That extra evidence can support explanations such as “recommended because you value battery life and display quality.”

TriRank was one of the final precursor works to Neural Collaborative Filtering (NeuCF) pioneered here at NUS.
*(He, Chen, Kan & Chen (2015), “TriRank: Review-aware Explainable Recommendation by Modeling Aspects,” CIKM 2015)*

▶ Start Timer (05:00)

# Slide 24: 04 Practical Comparison: Baseline First

# Slide 25: The experiment contract
| Hold fixed | Why | Inspect |
| :--- | :--- | :--- |
| **Data + split** | Both models face the same evidence. | Is test later than training? |
| **Negative sampler** | Sampling changes the ranking task. | Which unseen items become negatives? |
| **Metrics + K** | Metrics answer stated list-quality questions. | Does K match the product surface? |
| **Budget + reporting** | A gain needs cost and reproducibility context. | Time, parameters, seeds, examples. |

> **Warning**
> Changing the model and the split, sampler, or metric together does not identify what produced an apparent improvement.

# Slide 26: Practical walkthrough: BPR-MF in RecBole
**w04-ranking.yaml**
```yaml
# reproducible random choices
seed: 2026
# ranking evaluation protocol
eval_args:
 group_by: user
 order: TO
 split: {RS: [0.8, 0.1, 0.1]}
 mode: full
# one unobserved comparison per positive
train_neg_sample_args:
 distribution: uniform
 sample_num: 1
# report top-k list quality
metrics: [Recall, NDCG]
topk: [10]
```

**Illustrative run**
BPR learns a pairwise objective with latent-factor, dot-product scoring.
`run_recbole(model="BPR", dataset="ml-100k", config_file_list=["w04-ranking.yaml"])`

**Keep fixed:** split · sampler · metrics · seed policy → **Change:** model = BPR → **Record:** NDCG@10 · Recall@10 · time · parameters

# Slide 27: Practical walkthrough: change only the interaction function
**S26 reference · BPR-MF**
`run_recbole(model="BPR", dataset="ml-100k", config_file_list=["w04-ranking.yaml"])`
Same runner from Slide 26, kept deliberately muted.

**S27 implementation · NeuMF**
`run_recbole(model="NeuMF", dataset="ml-100k", config_file_list=["w04-ranking.yaml"])`
Only the interaction model changes.

**What changes inside the scorer:** NeuMF combines GMF and MLP paths. More parameters do not make implicit feedback less ambiguous.

> **Note**
> Locate the embedding tables, MLP layers, and prediction layer. Which is absent from a dot-product baseline?
> Hint: See the pictorial of NeuMF on Slide 14.

# Slide 28: Deployment decision clinic
Should this service replace BPR-MF with NeuMF?
* **Quality signal:** NDCG@10 rises from 0.214 (BPR-MF) to 0.221 (NeuMF).
* **Operational cost:** runtime rises from 2.1 to 7.8 minutes; parameters from 1.2M to 5.9M.

Group outcome: submit **one decision**—deploy NeuMF, retain BPR-MF, or collect more evidence—and justify it with **one reason that weighs both conditions**.

▶ Start Timer (03:00)

# Slide 29: Bandit Time: Which evidence supports NeuMF?
Which evidence best supports replacing BPR-MF with NeuMF in this service?

* **A.** NeuMF wins on one seed, although the latency measurement uses a different candidate set.
* **B.** A post-hoc explanation is persuasive to reviewers, but quality and operational cost are not compared.
* **C.** NeuMF gains after the time split and negative sampler both change from the BPR-MF run.
* **D.** Under the same split, sampler, and seeds, NeuMF has a stable meaningful gain, fits the latency budget, and documents limits.

# Slide 30: Debrief: read the result as a trade-off
* **Quality**: Stable across seeds, periods, and segments?
* **Operations**: Fits latency, memory, and maintenance budget?
* **Evidence**: Does the protocol justify the user-facing claim?

# Slide 31: 05 Revisiting our Card Lenses

# Slide 32: ♠ Spades: Requirements need policy
`stated requirement` → `eligibility + constraint policy` → `BPR-MF or NeuMF ranking` → `faithful explanation`

* **BPR-MF**: It learns an observed-versus-sampled ordering. It does not guarantee that an important attribute or rule is satisfied.
* **NeuMF**: A flexible interaction score neither makes an attribute a hard condition nor proves that its rationale is truthful.
* **Overclaim**: “This item is recommended because the model understands your preferences.”
* **Accountable claim**: “Among items meeting the stated requirements, this item ranked highly under our declared data, objective, and policy.”

> **Warning**
> Use the model to rank eligible options; do not use its score as evidence that a critical requirement has been met.

# Slide 33: ♥ Hearts: Neither score identifies free preference
`social cue or peer activity` → `prominent exposure` → `logged click`

* **BPR-MF · pairwise claim**: The clicked item moves above a sampled alternative. The loss does not identify independent interest, a friend’s cue, or ranking-induced visibility.
* **NeuMF · richer patterns**: It may learn complex user, item, and context co-occurrences. Flexibility cannot turn observational social traces into causal preference evidence.
* **Evidence + protection**: Record exposure and relevant social context; compare contexts; give users controls over social cues and sharing.

What in the data lets us say this reflects preference rather than system or peer influence?

> **Warning**
> A high score may predict behaviour after influence—not explain what the person independently wants.

# Slide 34: ♦ Diamonds: Better ranking can reduce discovery
`sparse long-tail evidence` → `BPR-MF or NeuMF ranking` → `top-K exposure`

* **BPR-MF · tail evidence**: Tail items have few observed positives. Sampled unobserved items become training alternatives despite weak evidence that they were seen.
* **NeuMF · no discovery objective**: Richer signals may help where they exist, but a small aggregate NDCG gain can still concentrate exposure on familiar items.
* **Report beside NDCG@K**: Catalogue coverage, long-tail share, new-item exposure, and relevant segment outcomes—plus candidate policy and sampler.

Would this result still count as an improvement if the top-K list became less discoverable?

> **Warning**
> BPR and NeuMF optimise the ranking objective supplied to them; neither automatically protects long-tail opportunity.

# Slide 35: ♣ Clubs: BPR and NeuMF can both learn an attack
**A pairwise win is not a robustness guarantee.**

* **BPR-MF · loss sees**: Observed positive > sampled unobserved item. Coordinated fake positives can create favourable training pairs for a target item.
* **NeuMF · capacity changes**: It can fit richer interaction patterns—including an attack pattern if that pattern appears in the log. More parameters are not an abuse defence.
* **Before a deployment claim**: Stress-test plausible shill or coordinated behaviour. Report ranking shifts, affected users, and target-item exposure.

If suspicious accounts or concentrated bursts are removed, does the claimed gain still hold?

> **Warning**
> Neither BPR-MF nor NeuMF is robust merely because it ranks held-out interactions well.

# Slide 36: Summary
**Architecture and objective are separate choices.**

* **1 · Represent**: Embeddings provide inputs.
* **2 · Interact**: MF dot product or neural function.
* **3 · Optimise**: BPR trains a relative ranking claim.
* **4 · Justify**: Keep costs, safeguards, and limits visible.

**Next week: order, recency, and session context enter the model.**

# Slide 37: NEXT WEEK PRE-LECTURE EXERCISE
**Where Does One Session End?**
Canvas discussion board · post before Mon, 7 Sep 23:59 SGT
Post one 100–140 word decision memo. No classmate reply is required.

> A person asks Ask Maps for a 2pm lunch route, explores nearby cultural sites, then later asks for a halal eatery open after 10pm. Does the late-evening query begin a new session? Answer yes or no and cite two trace details.

**Next week: sequences make order, recency, and local context part of the recommendation signal.**
*(Optional reading: The Straits Times, 11 Aug 2026)*

# Slide 38: NEXT WEEK PREVIEW — WEEK 05
**Sequential and Session-Based Recommendation**
A static user profile treats interactions as a collection. A sequence model asks whether **order, recency, and session context** change the next useful recommendation.

**In Week 5:**
* model temporal preferences;
* predict the next item; and
* critically assess engagement-driven objectives.
