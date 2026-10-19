# Slide 1: Latent Factor Models
**CP4285 Modern Recommendation Systems — Week 02**
CP4285 Instruction Team
18 Aug 2026

Scan for the slides: `soc-n.us/cp4285-t2610-w02`

# Slide 2: 01 Announcements and Context

# Slide 3: WEEK 01 RECAP

# Slide 4: Pre-Flight Survey · Metadata Summary
**PRE-FLIGHT SURVEY · 26 SUBMISSIONS** — cohort signals

* **COMPUTER SCIENCES**: 85% (22 of 26)
* **BALANCE / PRACTICE**: 81% (lecture preference)
* **IR KNOWLEDGE**: 15% (4 of 26)
* **EMBEDDINGS EXPOSURE**: 73% (19 of 26)
* **DSA KNOWLEDGE**: 92% (strong / decent)
* **AI KNOWLEDGE**: 100% (strong / decent)
* **INDUSTRY CASES**: 42% (10 of 24 responses)
* **RETRIEVAL / ARCHITECTURES**: 38% (9 of 24 responses)
* **BUILD / DEPLOY**: 33% (8 of 24 responses)

Source: Canvas pre-flight survey. Programme selections were multi-select; learning-expectation themes were coded from 24 substantive answers.

# Slide 5: Pre-Flight Survey · Voices in the Room
**OPEN RESPONSES · REPRESENTATIVE ANONYMOUS PHRASES** — larger phrases indicate central themes, not frequency

* “retrieval and ranking architectures”
* “more application based”
* “deployment and productionization”
* “TikTok / Instagram … real-world case studies”
* “YouTube and Spotify”
* “modern search alongside recommendation”
* “recommendation algorithms … real-world datasets”
* “end-to-end recommender … project”
* “Two Tower Model”
* “LLMs for recommendation … RAG or fine-tuning”
* “replicate papers”
* “create and deploy MCP server”
* “Heavy workload”
* “system design … production”
* “PyTorch … building and training”
* “latest research and progress”
* “form our own groups”
* “ethics … history and context”

● Systems and concepts ● Building and deployment ● Expectations and concerns
Source: Canvas pre-flight survey, anonymous open responses. Phrases are abbreviated from representative excerpts for slide readability.

# Slide 6: Essay 1: Micro Case Study: Designing a Production Recommender Surface
**Focus**
One approved user-facing recommendation surface.
→
**Reason**
One or two consequential Week 01–02 choices.
→
**Defend**
Evidence, limits, and a substantive suit lens.

⚠️ (Short) **1,500 words** · Due **Mon, 24 Aug 2026, 23:59 SGT**.

* Use no more than **two external sources**; distinguish platform facts from your reasoned proposal.
* Feature a card-suit lens in the technical design, objective, constraint, or evaluation criterion.
* Include an AI-use declaration; AI may assist inquiry but you need to own the argument.

# Slide 7: 02 From Neighbours to Features to Latent Structure

# Slide 8: LAST WEEK PRE-LECTURE EXERCISE
**Look for Hidden Structure**
Canvas discussion board · post before Mon, 23:59 SGT

Post a short, concrete response to this question; then reply constructively to one classmate.

> **Name one latent dimension that could explain several ratings—but is not an item metadata label.**

Next week: turn this intuition into latent-factor representations of the user-item matrix.

# Slide 9: LAST WEEK PRE-LECTURE EXERCISE · CANVAS VOICES
**Latent dimensions students noticed**

# Slide 10: Three Routes to a Recommendation
**Collaborative filtering (CF; last week)**
* **Evidence:** User–item interactions
* **Mechanism:** Borrow from similar users or items
* **Watch:** Thin local overlap can leave a 

**Content-based filtering (CBF)**
* **Evidence:** Explicit item, user, or context features
* **IR mechanism:** Encode and match genres, tags, and text
* **Watch:** Feature coverage and encoding

**Latent-factor model (LFM)**
* **Evidence:** User–item interactions
* **Mechanism:** Learn shared hidden coordinates from many interactions
* **Watch:** A factor is a learned pattern, not a personal fact.

# Slide 11: Explicit Features Can Be Weighted
**Observed or stated fields** can enter a recommender score before any latent factors are learned.

**User / context fields**
student · Singapore · vegetarian · weekday · evening · budget: low
*(Supplied, measured, or declared—not inferred)*

+

**Item fields**
cuisine: Thai · price: $$ · genre: comedy · duration: 90 min
*(Observed descriptions of a candidate item)*

→

**Weighted score**
`score(u, i, c) = w^T x`
Vector `x` represents an item or a user. Use a similarity metric (Week 01) to define similarities.

# Slide 12: Text Is One Possible Feature Source
Text can be an item field, user field, or contextual need.

**User / context text**
“animated family film”
“a light comedy tonight”
*Profiles, reviews, queries, and session descriptions*

↔︎

**Movie item text**
Toy Story (1995)
Animation · Children’s · Comedy
*Titles, genres, tags, synopses, and reviews can describe a candidate.*

→

**Encoded feature**
`x_text = encode(text)`
*Encoding maps words to coordinates for comparison or combination.*

# Slide 13: A Bag of Words Is Sparse and Literal
| text / term | animated | family | light | comedy | animation | children |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **need: animated family film** | 1 | 1 | 0 | 0 | 0 | 0 |
| **need: light comedy tonight** | 0 | 0 | 1 | 1 | 0 | 0 |
| **item: Toy Story (1995)** | 0 | 0 | 0 | 1 | 1 | 1 |

* One coordinate per vocabulary term; most coordinates are zero.
* The second need shares `comedy`; the first shares no exact coordinate with the listed movie genres.
* `animated` and `animation`, and `family` and `children`, remain separate until another mechanism relates them.

# Slide 14: TF and IDF Signal Different Things
**TF: local evidence**
`tf(t, d) = count of term t in document d`
Repeated occurrence can strengthen a term’s contribution **within one document**.

**IDF: collection discrimination**
`idf(t) = log (N / df_t)`
A term found in almost every document contributes less to distinguishing documents.

**tf-idf(t, d) = tf(t, d) × idf(t)**

# Slide 15: Worked TF–IDF: Two Titles, One Interest Profile
**User interest:** animated family comedy
**A:** Toy Story · Animation · Children’s · Comedy
**B:** Mrs. Doubtfire · Family · Comedy

**Vocabulary coordinates**
| term | IDF | U | A | B |
| :--- | :--- | :--- | :--- | :--- |
| animated | 1.8 | 1 | 0 | 0 |
| animation | 1.8 | 0 | 1 | 0 |
| family | 1.4 | 1 | 0 | 1 |
| children | 1.4 | 0 | 1 | 0 |
| comedy | 1.2 | 1 | 1 | 1 |
*animated ≠ animation; family ≠ children.*

**Cosine = dot product ÷ (two vector norms)**
`cos(u, t) = (u · t) / (‖u‖ ‖t‖)`

* **A · Toy Story:** 0.217
  * dot: 1.2 × 1.2 = 1.44
  * norms: ‖U‖ = 2.577; ‖A‖ = 2.577
  * 1.44 ÷ (2.577 × 2.577)

* **B · Mrs. Doubtfire:** 0.716
  * dot: 1.4² + 1.2² = 3.40
  * norms: ‖U‖ = 2.577; ‖B‖ = 1.844
  * 3.40 ÷ (2.577 × 1.844)

# Slide 16: Dense Word Embeddings Learn Neighbourhoods
**Training contexts**
Words that occur in related contexts provide a learning signal.
→
**Dense vectors**
Each word receives a short, continuous learned representation.
→
**Neighbourhoods**
Related usage can place vectors closer together.

* A sparse document–term vector has dimension (V): one coordinate per vocabulary term, with most coordinates equal to zero.
  * **Typical scale:** V =∼ 50,000–100,000 terms after tokenisation and filtering.
* Dense word embeddings use a much smaller learned dimension `d ≪ V`, while retaining continuous coordinates for semantic similarity.
* Their coordinates are learned patterns, **not human-readable facts or ground truth**.

# Slide 17: Sparse and Dense Make Different Trade-offs
| Question | TF–IDF | Dense word embedding |
| :--- | :--- | :--- |
| **What is represented?** | Exact vocabulary-term evidence | Learned continuous coordinates |
| **Main strength** | Transparent lexical matching | Can generalise across related usage |
| **Main limitation** | Synonyms remain separate terms | Learned proximity can be opaque or misleading |
| **Typical score** | Cosine similarity over sparse vectors | Cosine/dot-product similarity over dense vectors |

> **Bridge:** IR turns *text into comparable vectors*. LFM will turn *interaction history into comparable user and item vectors*.

# Slide 18: Break — 5 minutes
**IR HISTORY · 1958: Keyword search predates the Web**
At IBM, **Hans Peter Luhn** demonstrated **Key Word in Context (KWIC)**: an automatic index that rearranged keywords while preserving their textual context.

By the early 1960s, KWIC had become central to hundreds of computerised indexing systems—a useful reminder that retrieval began with finding useful words, not with the Web.

▶ Start Timer (05:00)

# Slide 19: 03 Latent Factor Models

# Slide 20: One Vector-Space Idea, Two Evidence Sources
| Retrieval | Latent-factor recommendation |
| :--- | :--- |
| Query and document vectors | User and item vectors |
| Evidence: words in text | Evidence: observed interactions |
| Score: lexical/semantic match | Score: predicted affinity |
| Goal: rank relevant documents | Goal: rank plausible items |

**Analogy, not equivalence:** a latent factor is not a word, a recorded metadata field, or a fact about a person.

# Slide 21: Video: SVD for Movie Recommendations

# Slide 22: Two Factorisation Lenses: Signed or Additive?
Both methods approximate a matrix with fewer factors, but their **constraints change the meaning** of those factors.

**SVD-style lens**
`R ≈ U_k Σ_k V_k^⊤`
* Factors may be **positive or negative**.
* Natural for centred residuals: above- and below-baseline evidence can offset.
* The singular values in (_k) scale the retained directions.

**NMF lens**
`X ≈ WH, W, H ≥ 0`
* Factors are **nonnegative and additive**.
* Natural for counts, intensities, or “parts” interpretations.
* Components combine; they do not cancel one another.

**Shared goal:** compress matrix structure. **Different question:** may a factor subtract?

# Slide 23: Worked SVD Lens: Signed Residual Directions
**Toy centred residual matrix** — positive means “above this user’s baseline”; negative means “below it.”

**Read the signs**
* User 1: Item A above baseline; Item B below.
* User 2: the reverse pattern.
* One signed direction captures the contrast.

**Why SVD fits this view**
The negative entries are meaningful **residual directions**, not impossible ratings. A nonnegative-only factorisation could not reproduce this signed residual matrix directly.

# Slide 24: Worked NMF Lens: Additive Intensity Components
**Toy nonnegative interaction matrix** — every entry is an intensity, so components are required to add rather than subtract.

**Read the factors**
* (W): User 2 has twice the component intensity of User 1.
* (H): Item A expresses twice as much of the component as Item B.
* Their product reconstructs every entry without negative values.

**Why NMF fits this view**
For counts or accumulated activity, a factor can be interpreted as an **additive amount**. There is no “minus 3 clicks” component to cancel another one.

# Slide 25: Bandit Time: What Does Factorisation Learn?
A restaurant app has a sparse matrix of diner ratings. Which description best matches latent-factor factorisation?

* **A.** It sorts restaurants only by their recorded cuisine label.
* **B.** It learns compact vectors for diners and restaurants from many observed interactions, then uses their alignment to estimate affinity.
* **C.** It treats every unvisited restaurant as a one-star rating.
* **D.** It copies the rating of the geographically nearest diner.

# Slide 26: A Dot Product Turns Factors into a Score
`r^_ui = u_u^⊤ v_i`

**User vector**
`u_u = [u_{u1}, u_{u2}, ... , u_{uk}]`
A learned location in the model’s preference space.

**Item vector**
`v_i = [v_{i1}, v_{i2}, ... , v_{ik}]`
How an item aligns with the learned interaction patterns.

The dot product produces a **model score**. A production recommender can still filter, diversify, or re-rank the resulting candidates.

# Slide 27: Worked Calculation: Two Items, One Score
`u_u = [0.8, 0.2]`, `v_A = [0.9, 0.1]`, `v_B = [0.4, 0.8]`

**Item A**
`r^_uA = 0.8(0.9) + 0.2(0.1) = 0.74`

**Item B**
`r^_uB = 0.8(0.4) + 0.2(0.8) = 0.48`

**Current model verdict:** Item A scores higher. **Still unresolved:** diversity, constraints, uncertainty, exposure, and the rest of the candidate list.

# Slide 28: Training Fits Evidence, Regularisation Limits It
`min_{U,V} ∑_{(u,i)∈Ω} (r_{ui} - u_u^⊤ v_i)^2 + λ(‖U‖_F^2 + ‖V‖_F^2)`

* **Observed interaction set Ω:** explicit ratings or implicit events select the evidence the objective fits.
* **User factors U and their row vectors:** are adjusted together with the item factors to explain the selected evidence.
* **Item factors V and their row vectors:** align with user factors to produce the model score.
* **Regularisation strength λ:** limits convenient memorisation by penalising excessively large factor values.

# Slide 29: Group Discussion · Choose the Lens
In groups of 3–4, assign a lens, state the evidence, and name one caveat. **Time: 5 minutes.**

* **Case A · Centred ratings**
  Each value is a deviation from the user’s own average rating: some are positive, some negative. *Choose SVD or NMF. Why?*
* **Case B · Watch counts**
  Entries are nonnegative counts of completed views across topic groups. *Choose SVD or NMF. Why?*
* **Case C · Missing ratings**
  Most user–item pairs were never observed; they are not zero ratings. *What must both approaches avoid?*

**Report back:** one choice, one data property that justifies it, and one interpretation or missing-data caveat.

▶ Start Timer (05:00)

# Slide 30: NEXT WEEK · PREVIEW
**Week 03 — Evaluation of Recommendation Systems**
An offline metric tells us how a list performed. We ask with our four-suit lenses whether it performs well under a scenario.

* ♣ **Robustness:** Can the protocol resist distorted or manipulated evidence? 
* ♥ **Social impact:** Does the ranking inform without unwanted exposure or conformity?
* ♦ **Discovery:** Does the list surface genuinely new, long-tail, or emerging choices? 
* ♠ **Constraints & explanation:** Can users hold important attributes fixed and understand why an item appears?
