# Slide 1: Evaluation of Recommendation Systems
**CP4285 Modern Recommendation Systems — Week 03**
CP4285 Instruction Team
25 Aug 2026

Scan for the slides: `soc-n.us/cp4285-t2610-w03`

# Slide 2: 01 Announcements and Context
**CHANGELOG**
* 25 Aug 2026 — Added the Week 02 Canvas Voices review and guardrail discussion.
* 25 Aug 2026 — Added the Week 04 pre-lecture activity and linked preview image.

# Slide 3: WEEK 02 RECAP

# Slide 4: Recap: Yesterday’s deadlines
**MON, 24 AUG · 23:59 SGT**

* ✓ Essay e1
* ✓ Mini-team formation
* ✓ Personal ID
* ✓ Pre-lecture exercise

# Slide 5: Essay e1 Peer Review
**YOUR TASK**
TA Sahaj will manually assign you another student’s essay that is on a different recommendation platform.
1. Read the assigned essay.
2. Review and critique that essay only for the card suit lens.

**FOCUS OF CRITIQUE**
Assess the quality of the essay’s card suit lens' quality.
* How well does it match the recommendation surface?
* How useful is the insight?

The two marks for the peer review are for the completion of your peer review and its quality. Your essay’s quality will be assessed by the eight marks given by the instruction staff.

Due: **Mon, 31 Aug 2026 · 23:59 SGT**

> **AI-use note.** As per the `e1` specification, remember AI should not be used in completing the peer review.

# Slide 6: LAST WEEK PRE-LECTURE EXERCISE
**What Should Evaluation Be Responsible For?**

**Read:** “S’pore studying restrictions on direct messaging, auto-play on social media to protect kids” (PHOTO: ST FILE)

**Canvas discussion board · post one individual comment before Mon, 23:59 SGT**
> A recommender may be evaluated using whether people watch, click, or continue interacting with suggested content. What is one responsibility that evaluation should include for younger users, beyond predicting continued engagement?

In 120–160 words, choose and name **one card-suit lens**, cite one article detail, and propose one piece of evidence, safeguard, or user control the platform should examine.

**No classmate reply this week. Today: protocols, ranking metrics, and what a score leaves out.**

# Slide 7: LAST WEEK PRE-LECTURE EXERCISE · CANVAS VOICES
**Diamond and Clubs guardrails students proposed ♦ ♣**

# Slide 8: Evaluation makes a claim about success
Today: design the protocol, read ranking metrics, and ask what the metrics leave out.

> **Note**
> A model can score user–item pairs. An evaluation protocol decides whether those scores support a useful, defensible decision.

# Slide 9: Key Point: Evaluation Is More Than a Score
Defensible evaluation connects a protocol, a ranking metric, and the limits of the evidence.

1. **Design** evaluation protocols
2. **Compute** ranking metrics
3. **Critique** metric selection
4. **Explain** limitations of offline evaluation

Ask what the recommender is being credited for—and what the evidence cannot establish.

# Slide 10: Bandit Time: What did the model learn?
A user has never clicked a recommended film. Which conclusion is justified?

* **A.** The user dislikes the film. 
* **B.** The film was shown and ignored for a known reason.
* **C.** The log alone cannot separate preference from exposure or interface effects.
* **D.** The latent-factor model has no prediction for this user.

▶ Start Timer (00:30)

# Slide 11: TempoQuiz debrief: absence is not a label
**Tempting shortcut**
“No click” means dislike, so treat it as a negative label.

**Evaluation-aware view**
No click may mean no exposure, poor placement, wrong time, inattention, or lack of interest. The protocol must say which interpretation is defensible.

# Slide 12: 02 Evaluation Design

# Slide 13: The evaluation contract
**Scenario**
Who is deciding, and on which surface?
→
**Candidates**
Which items could have appeared?
→
**Relevance**
What counts as a useful outcome?
→
**Decision**
What is shown at top K?

<u>Do not choose a metric until these four terms are explicit.</u>

# Slide 14: Bandit Time: Which split supports this claim?
You claim a news recommender will work next week. Which test design best supports that claim?

* **A.** Train and test on the same logged clicks. 
* **B.** Randomly split all interactions, including future events and features.
* **C.** Train on earlier interactions; tune on a later validation period; test on an untouched future period.
* **D.** Choose the split after seeing which one gives the highest NDCG.

▶ Start Timer (00:30)

# Slide 15: Training, validation, and testing serve different decisions
**Train**
Fit the recommender using only information available at the training cutoff.

**Validate**
Choose hyperparameters, thresholds, and model variants without touching the final test period.

**Test once**
Estimate performance of the chosen system on an untouched period; do not tune again after seeing it.

# Slide 16: 03 Ranked Lists, Not Just Scores

# Slide 17: Complete the Metric Lens Table
In pairs, complete the four orange cells. Be ready to justify one choice from the ranking behaviour it measures.

| Metric / signal | Question answered | Useful when |
| :--- | :--- | :--- |
| **Precision@K** | Of the K shown items, how many were relevant? | ________ |
| **Recall@K** | How much of the relevant set was retrieved? | ________ |
| **Mean Average Precision (MAP)** | Are relevant items repeatedly placed early? | ________ |
| **Normalised Discounted Cumulative Gain (NDCG)** | Are the most valuable items at the top? | ________ |

# Slide 18: Offline ranking metrics
| Metric / signal | Question answered | Useful when |
| :--- | :--- | :--- |
| **Precision@K** | Of the shown K, how many were relevant? | Visible space is scarce. |
| **Recall@K** | How much relevant content was retrieved? | Missing useful items is costly. |
| **MAP** | Are binary-relevant items repeatedly early? | Several relevant items may matter. |
| **NDCG** | Are higher-value items early? | Relevance is graded; rank matters. |

> **To think about**
> How do precision and recall evolve as k increases? When is P@K or R@K more useful as an evaluation lens?

# Slide 19: Which recommender wins?
For a five-item music shelf, choose the better recommender and defend one metric.

**A: early high-value hit**
1. ★★★★★ 2. ○ 3. ○ 4. ★★ 5. ○

**B: more binary hits**
1. ★★ 2. ★★ 3. ○ 4. ★★ 5. ○

In groups of 3–4: name the surface assumption, the metric, and one reason the other system might still be preferred.

▶ Start Timer (03:00)

# Slide 20: Worked MAP@5: binary relevance
**A** · 1. ★★★★★ · 2. ○ · 3. ○ · 4. ★★ · 5. ○ 
**B** · 1. ★★ · 2. ★★ · 3. ○ · 4. ★★ · 5. ○

For MAP, ★★ and ★★★★★ are both binary hits.

`AP@5 = 1/R ∑_{i=1}^{5} P@i × rel_i`

**A: hits at ranks 1 and 4**
Relevant ranks: 1, 4
`AP@5 = (P@1+P@4)/2 = (1+2/4)/2 = .750`

**B: hits at ranks 1, 2, and 4**
Relevant ranks: 1, 2, 4
`AP@5 = (P@1+P@2+P@4)/3 = (1+1+3/4)/3 = .917`

> **Would graded relevance change the result?**
> Yes—but then this is no longer standard MAP. A different relevance threshold changes AP; inserting rating grades directly creates a different graded metric and may favour A.

# Slide 21: Worked NDCG@5: graded relevance
**A** · 1. ★★★★★ · 2. ○ · 3. ○ · 4. ★★ · 5. ○ 
**B** · 1. ★★ · 2. ★★ · 3. ○ · 4. ★★ · 5. ○

Grades: ○ = 0, ★★ = 2, ★★★★★ = 5. The gain term preserves their difference.

`NDCG@5 = DCG@5 / IDCG@5`
`DCG@5 = ∑_{i=1}^{5} (2^{rel_i} - 1) / log_2(i + 1)`

**A: ★★★★★ at 1; ★★ at 4**
`DCG = 31 + 3/log_2 5 = 32.292`
`IDCG = 31 + 3/log_2 3 = 32.893`
`nDCG@5 = .982`

**B: ★★ at 1, 2, and 4**
`DCG = 3 + 3/log_2 3 + 3/log_2 5 = 6.185`
`IDCG = 3 + 3/log_2 3 + 3/2 = 6.393`
`nDCG@5 = .968`

> **Note**
> IDCG == Idealised Discounted Cumulative Gain. An oracular value that simulates the best possible outcome.

# Slide 22: Click-through rate (CTR): an online metric
**Calculation**
`CTR = clicks / impressions × 100%`

* **Impression:** a visible recommendation opportunity.
* **Click:** an opening or action that your instrumentation counts.

**Worked example**
`28 clicks / 400 impressions × 100% = 7%`

CTR asks whether people acted after seeing the recommendation. It is measured online, after a deployed surface creates impressions.

> **To think about**
> How do you apply CTR to our two use cases? What assumptions should you make?

# Slide 23: Key Point: A Metric Orders Priorities
A system can be better at the top, across many relevant items, or at avoiding wasted slots.

* **Top of the list**: Reward especially valuable items that appear early.
* **Repeated relevance**: Reward several relevant items placed early.
* **Wasted slots**: Penalise irrelevant items in scarce visible space.

**Your metric makes one of these priorities count most.**

# Slide 24: Break — 5 minutes
**BREAK · CTR DEPENDS ON THE SURFACE**
The same recommendation can earn different click-through rates because the surface changes the user’s intent, the competing choices, and the meaning of an impression.

* **Paid search ads**: 6.64% (Overall CTR across 13,000 U.S. campaigns. WordStream 2026)
* **Social traffic ads**: 1.71% (Average CTR for Facebook traffic campaigns. WordStream 2025)
* **Email**: Context matters (Audience, message, and timing shape the response. Mailchimp)

**Rule:** compare CTR only across like-for-like surfaces, placements, audiences, and impression definitions.

▶ Start Timer (05:00)

# Slide 25: 04 Beyond Accuracy

# Slide 26: Accuracy is a proxy, not a verdict
> **Warning**
> An observed click can be a useful signal without being a complete account of benefit, consent, satisfaction, or long-term welfare.

# Slide 27: Same precision, different catalogue
**System P: popular repeat**
Every user sees familiar blockbusters. Precision@5 is high, but the catalogue is barely touched.
*Likely risk: repeated exposure and narrow discovery.*

**System D: relevant discovery**
Each user sees useful, less familiar options across the catalogue. Precision@5 is the same.
*Potential value: novelty, diversity, and long-tail exposure.*

# Slide 28: Metric design clinic: choose the guardrails
Your group is assigned one card lens. Design a minimal evaluation plan for a recommendation surface.

* ♣ **Clubs — adversarial / shill**: Can coordinated activity manipulate the list? Consider attack impact, anomaly detection, and stability.
* ♥ **Hearts — social**: Does social evidence help without unwanted exposure or conformity? Consider privacy, welfare, and subgroup outcomes.
* ♦ **Diamonds — novelty seeker**: Does the list enable discovery? Consider novelty, diversity, serendipity, and catalogue coverage.
* ♠ **Spades — loyalist / metadata-driven**: Are important requirements stable and visible? Consider constraints, consistency, calibration, and explanation.

Name: (1) one primary metric; (2) two guardrails; and (3) one evidence gap that offline logs cannot resolve. Prepare a two-sentence report.

▶ Start Timer (03:00)

# Slide 29: Debrief: metric choices distribute attention
| Lens | Possible primary measure | Example guardrail |
| :--- | :--- | :--- |
| ♣ | Ranking utility under attack | Attack impact / stability |
| ♥ | Task completion or satisfaction | Privacy, opt-outs, welfare signal |
| ♦ | NDCG or utility | Novelty, diversity, coverage |
| ♠ | Constraint-satisfied relevance | Consistency / explanation quality |

# Slide 30: Evaluation criteria map
* **USEFUL NOW: Relevance & utility** (Precision, recall, MAP, NDCG, task completion)
* **BEYOND RELEVANCE: Engagement, exposure & discovery** (CTR, dwell time, exposure distribution, coverage, intra-list diversity, novelty, serendipity)
* **SAFE AND DEPENDABLE: Trust & resilience** (Confidence, robustness, stability, scalability)

> **To think about**
> Where should the computation budget sit—candidate generation, ranking, or re-ranking? What end-to-end latency can this surface tolerate before a useful recommendation arrives too late?

> **Note**
> Measure the three ideas separately. *Dwell time* captures depth after an interaction; *exposure distribution* records who and what receives impressions; *intra-list diversity* asks how varied one ranked list is. None is interchangeable with relevance.

# Slide 31: Beyond relevance: measure what receives attention
**Engagement (CTR · dwell time)**
Did people act after a visible recommendation opportunity?

**Exposure (Impression share · coverage)**
Who and what receives a place in the ranked list?

**Discovery (Diversity · novelty · serendipity)**
Does the list expand useful options beyond familiar repeats?

> **Note**
> These measures answer different questions. A system can earn clicks while concentrating exposure, or improve discovery while leaving CTR unchanged.

# Slide 32: Intra List Diversity (ILD): how varied is one ranked list?
**A three-item recommendation list** (Item A, Item B, Item C)
Use a content representation to estimate each pair’s cosine similarity.

* **A–B:** similarity = 0.80, distance = 0.20
* **A–C:** similarity = 0.20, distance = 0.80
* **B–C:** similarity = 0.30, distance = 0.70

**Average pairwise dissimilarity**
`ILD@3 = (0.20+0.80+0.70)/3 = 0.57`

Interpretation: this list has moderate intra-list diversity on a 0–1 distance scale. A higher score means the items are less alike under the chosen representation.

> **To think about**
> What item representation would make two recommendations look more—or less—similar: genre labels, text embeddings, creator, or user interaction patterns? What is the complexity of this calculation? Is it important to consider in practical service?

# Slide 33: Safe and dependable: measure whether the system holds up
**Trust: Calibration · explanation coverage**
Are confidence and explanations proportionate to what the system knows?

**Resilience: Attack impact · ranking stability**
How much does the list change under manipulation or small perturbations?

**Scalability: p95 latency · throughput**
Can the surface respond reliably under its expected traffic and catalogue size?

> **Warning**
> There is no universal safety score. Select operational and resilience checks that match the surface, the users affected, and the failure mode at stake.

# Slide 34: 95% Percent Latency (p95): Protect the slow tail
**20 sorted recommendation-response latencies (ms)**
120 · 125 · 128 · 130 · 135 · 140 · 145 · 148 · 150 · 152 155 · 160 · 165 · 170 · 175 · 180 · 190 · 205 · 240 · 800
*(Use the nearest-rank convention for this worked example.)*

**Find the 95th percentile**
`rank = ⌈0.95 × 20⌉ = 19`
`p95 = L_19 = 240 ms`

Interpretation: 95% of requests completed in 240 ms or less; the slowest 5% took longer. The single 800 ms request still matters, but it is not p95.

> **To think about**
> What are appropriate interventions to address long latency as a result of a high p95?

# Slide 35: The feedback loop hidden in a log
* **Exposure**: What the system displayed
  →
* **Interaction**: Click, play, skip, save
  →
* **Log**: Observed, selected data
  →
* **Model**: Learns from the log
  →
* **Future exposure**: Changes the next log

> **Warning**
> Unobserved is not random. The system’s own past choices influence what can be learned and evaluated.

# Slide 36: Four offline failure modes
**MNAR / selection bias**
Logged feedback reflects what users chose and what the system exposed. Mitigate with temporal realism, exposure data, and careful claims.

**Temporal drift**
Preferences and catalogues change. Use forward-looking splits and report the evaluation window.

**Leakage**
Future features or test-set tuning inflate results. Enforce a feature cutoff and final untouched test period.

**Cold start and long tail**
Averages can hide new users, new items, and rare items. Report slices and coverage, not just one aggregate score.

# Slide 37: Offline → online → user impact
**Offline benchmark**
Compare candidates under a declared protocol; inspect coverage and intra-list diversity
→
**Guarded online test**
Randomise exposure; monitor CTR, dwell time, exposure distribution, and safeguards
→
**User / product impact**
Interpret retention, satisfaction, complaints, and long-term effects

# Slide 38: Summary
**Evaluation choices define what counts as a good recommendation.**

**1 · Begin with the claim**
Name the surface, decision, candidates, and relevant outcome before choosing a split or metric.

**2 · Judge ranked lists**
Match Precision@K, Recall@K, MAP, or NDCG to the ranking behaviour that matters.

**3 · Add guardrails**
Pair utility with discovery, exposure, trust, robustness, and constraint checks that fit the card lens.

**4 · Keep claims proportional**
Offline logs are a simulation: test forward in time, report evidence gaps, then validate with guarded online evidence.

**A metric orders priorities; an evaluation plan makes those priorities defensible.**

# Slide 39: NEXT WEEK PRE-LECTURE EXERCISE
**What Does a Missing Interaction Mean?**
Canvas discussion board · post before Mon, 31 Aug 23:59 SGT

Post one concise response; then reply constructively to one classmate.

> A user has never clicked a recommended item. Give two plausible interpretations: one about preference and one about exposure or the interface.

Next week: neural recommenders can learn richer interaction patterns, but they still learn only from the evidence the system records.

# Slide 40: NEXT WEEK PREVIEW — WEEK 04
**Neural Recommendation Models**
A neural recommender can learn richer patterns from interactions. That does not make an unclicked item self-explanatory.

**In Week 4:**
* compare neural and latent-factor recommenders;
* examine deep ranking and representation learning; and
* ask what model complexity can—and cannot—explain.

*(Illustration: Google DeepMind / Pexels)*
