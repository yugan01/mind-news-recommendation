# Slide 1: Graph-Based Recommendation
**CP4285 Modern Recommendation Systems — Week 09**  
CP4285 Instruction Team  
13 Oct 2026  

Scan for the slides: `soc-n.us/cp4285-t2610-w09`

---

# Slide 2: 🎰 Bandit Time: Start TempoQuiz
**Week 08 recap**  
Join using the live host-dashboard QR or room code.  
Five minutes including joining and explanations.

**What does a preference pair tell us?**  
Recall BPR, ranked-list utility and exposure.  
The instructor will release the questions in TempoQuiz.

---

# Slide 3: WEEK 08 RECAP

### One ranked list, three learning targets
* **Pointwise**: Is this item relevant? Learn a score or label for each user–item pair independently. Useful when explicit and calibrated scores are given.
* **Pairwise**: Which of two items comes first? Learn that an observed or preferred item should outrank a comparison item. Useful when relative preference is clearer than a rating.
* **Listwise**: How should the candidate group rank? Define a loss using the candidate group, such as a target distribution. Position-sensitive metric alignment depends on the training loss and update weighting rules.

### BPR: pairwise ranking from implicit feedback
* $T = \{(u, i, j) : i \in I_u, j \notin I_u\}$, $\Delta_{uij} = s(u, i) - s(u, j)$
* Here $I_u$ contains the items user $u$ interacted with. $T$ defines assumed training preferences, not proven dislikes.
* **Objective versus model**: BPR specifies the training criterion. BPR-MF uses matrix factorisation to supply the score. Other differentiable scoring models can use the same pairwise loss.
* **Observed does not mean universally preferred**: A sampled unobserved item may be unseen, not disliked. The sampler determines which comparisons training encounters.

### Listwise quality: measure the displayed order
Relative order, not position weighting: basic BPR has no explicit NDCG swap weights.
1. **Convert grades to gains**: $g = 2^r - 1$
2. **Discount gains by position**: $\text{DCG}@3 = \sum \frac{g_i}{\log_2(i+1)}$
3. **Normalize**: $\text{NDCG}@3 = \frac{\text{DCG}@3}{\text{IDCG}@3}$

### BPR in the learning-to-rank map
| Method | Training signal | Correction |
| :--- | :--- | :--- |
| **Pointwise** | One item and its label | Item-level prediction error |
| **RankNet** | Relevance preference pair | Logistic pairwise error |
| **BPR** | Observed / unobserved user pair | Logistic error + pair sampling |
| **LambdaRank** | Pair in the current ranking | Pairwise signal $\times$ swap impact |
| **ListNet** | Candidate-group distribution | Group cross-entropy |

> **BPR-MF = a matrix-factorisation scorer trained with BPR.** Neither a neural scorer nor BPR alone despises fair exposure.

---

# Slide 4: 01 Announcements and Context
**Changelog**
* **13 Oct 2026** — Added graph walks, priors, learned propagation, worked music examples and two activity cycles. Replaced S10 with six verified Canvas quotations and author avatars grouped by three evidence considerations.
* **6 Oct 2026** — Added TempoQuiz, exercise review and recommendation-history break.
* **5 Oct 2026** — Added the next-week prelecture exercise and preview.

---

# Slide 5: Course checkpoints
**ANNOUNCEMENTS**
* **Essay 2**: deadline was Mon, 12 Oct · 23:59 SGT.
* **TEAMMATES**: indicative feedback has been released. This feedback is not counted towards your grade.
* **STePS**: all teams except 3 and 6 submitted that they will present.
* **Weeks 10 and 11**: optional 15-minute sessions for additional project feedback.
* **Presentation deck critiques**: to be released later this week.

---

# Slide 6: Make experiments explain
**PROJECT ADVICE**
* **Justify your choices**: why this model, dataset and evaluation metric?
* **Explain successes and failures**: understanding matters more than complexity or state-of-the-art results.
* **Look beyond an aggregate score**: inspect individual examples and meaningful subgroups.
* **Build on prior work**: read, cite and, where feasible, replicate it. Contribute your own analysis.

---

# Slide 7: Ablations and experiment scale
**EXPERIMENTATION ADVICE**

### Test a component's contribution
* Predict the effect before running the experiment.
* Remove a feature group or component; additive comparisons also help.
* Retrain each variant. Keep the split and evaluation protocol fixed.
* Explain where results change and why.

### Validate before scaling
* **1–3 min · Toy**: check correctness on tiny data.
* **30–60 min · Medium**: test promising ideas and inspect results.
* **Overnight · Full**: run validated settings at scale.
* Estimate runtime from initial runs; sample data to fit your budget.

---

# Slide 8: Learning outcomes
* Represent recommendations as graph problems.
* Explain graph propagation.
* Build graph-based recommenders.
* Analyze risks of graph-driven feedback loops.

---

# Slide 9: LAST WEEK PRE-LECTURE EXERCISE · W08
**Popularity or connection? Fix the surface**  
Canvas · Solo · 15 minutes · no peer reply  
Complete by Mon, 12 Oct 2026 · 23:59 SGT  

Choose one surface and keep its user, task, and constraints fixed throughout your answer:

| A · Music autoplay | B · Jobs search | C · Grocery checkout |
| :--- | :--- | :--- |
| Next track after an acoustic-folk playlist. Keep the genre transition gentle. | First search-result card for a graduate seeking entry-level data work in Singapore. | One “Complete your meal” checkout item for a vegetarian dinner basket. It must be available. |

Both candidates are eligible: $X$ has 1,000 interactions but no known overlap with this user’s interests; $Y$ has 20 interactions, several from users with overlapping histories. Counts come from the same period and interaction definition.

**Post 120–160 words**: name your surface and concrete X/Y items; choose which to investigate first; state one assumption, one check of exposure, and evidence that would reverse your choice.

> **Today’s lecture**: ask what the pattern of connections reveals beyond an item’s total count.

---

# Slide 10: LAST WEEK PRE-LECTURE EXERCISE · CANVAS VOICES
**What evidence lies behind the count?**

### Relevant connections
*Does overlap reflect the same needs?*
* “The overlapping history would not be useful as a connection if the overlapping histories were about something else like toiletries.”  
  — *Context of the overlap*
* “My assumption is that similar baskets reflect similar meal needs, and that both products meet the shopper’s dietary requirements.”  
  — *Preferences and eligibility*

### Exposure & visibility
*Did placement produce the count?*
* “I would check exposure by looking at how often each job was shown to similar users in comparable search-result positions, then compare their interaction rates.”  
  — *Comparable opportunities*
* “X may have accumulated more interactions simply because it received greater exposure.”  
  — *Visibility behind popularity*

### Interaction meaning
*What does an observed edge mean?*
* “Assumption: Interactions are positive signals (full plays, saves), not skips or accidental plays.”  
  — *Positive interest or incidental activity?*
* “What would reverse my choice is if Y’s 20 interactions turn out to be mostly apply-then-reject or quick bounces, or if X’s clickers include lots of similar grads after all.”  
  — *Evidence that reverses a choice*

> **Guiding Question**: Before propagating an edge, what evidence does that edge carry?

---

# Slide 11: 02 Ranking Through Graphs
A shared example for walks, priors and learned propagation.

---

# Slide 12: Four listeners, four tracks
**Interaction cycle: four users $\times$ four tracks**

| Listener | Observed tracks |
| :--- | :--- |
| **A · Anjali** | R, S |
| **B · Bala** | S, T |
| **C · Cheryl** | T, U |
| **D · Danish** | U, R |

**Track Catalog:**
* **R (Roots)** · world fusion
* **S (Sunshower)** · dance pop
* **T (Tides)** · instrumental surf rock
* **U (Uptown)** · funk pop

**Anjali’s unseen candidates**: T, U.

* **Graph Structure**:
  * Users (teal circles): A, B, C, D
  * Tracks (orange rounded rectangles): R, S, T, U
  * Regular cycle structure: A — R — D — U — C — T — B — S — A
  * Every node has degree 2.

---

# Slide 13: Degrees and paths
* Music interactions are undirected: either direction is allowed.
* Every node has degree 2.
* Each direction uses its source degree:
  $$P_{ij} = \frac{1}{d_i} = \frac{1}{2}$$

**Paths from Anjali to unseen candidates:**
* $\text{Anjali} \to \text{Sunshower} \to \text{Bala} \to \text{Tides}$
* $\text{Anjali} \to \text{Roots} \to \text{Danish} \to \text{Uptown}$

**Asymmetric transition example:**
If Anjali had degree 2 and Roots degree 3:
* $\text{Anjali} \to \text{Roots}: 1/2$
* $\text{Roots} \to \text{Anjali}: 1/3$

---

# Slide 14: 🎥 Video: PageRank’s random surfer
Visual demonstration of PageRank's random surfer traversing web graph nodes (Pages 1 through 5).

---

# Slide 15: Markov state and PageRank
Same people, new relation: arrows mean follows.

* **Directed follow relations**:
  * $A \to B$ ($1/2$), $A \to C$ ($1/2$)
  * $B \to C$ ($1$)
  * $D \to C$ ($1$)
  * $C \to A$ ($1$)

* Start at Anjali (A): $r^{(0)} = (1, 0, 0, 0)^\top$; node order: A, B, C, D.
  $$r^{(k+1)} = P^\top r^{(k)}$$

* **One step**:
  $$r^{(1)} = \left(0, \frac{1}{2}, \frac{1}{2}, 0\right)^\top$$

* **Two steps**:
  $$r^{(2)} = \left(\frac{1}{2}, 0, \frac{1}{2}, 0\right)^\top$$

> *Reference*: Brin, S., & Page, L. (1998). *The anatomy of a large-scale hypertextual Web search engine*. Computer Networks and ISDN Systems, 30(1-7), 107–117. https://doi.org/10.1016/S0169-7552(98)00110-X

---

# Slide 16: Stationarity and periodicity
Return to the undirected music cycle.

* Starting at Anjali, total user-side mass across steps is:
  $$1, 0, 1, 0, \dots$$
* A return to a user takes an even number of steps. The chain has **period 2** (bipartite).
* Yet a stationary distribution exists:
  $$\pi_j = \frac{d_j}{\sum_i d_i} = \frac{2}{16} = \frac{1}{8}$$
* **Stationarity does not imply convergence from every starting state.**

---

# Slide 17: Personalised PageRank: compare one step
Back to social follows, order A, B, C, D.

* **One-hot start**: $r^{(0)} = \mathbf{1}_A = (1, 0, 0, 0)^\top$. All initial mass is at Anjali.
* **Restart**: probability $\tau = 1/2$; destination distribution $q$.
  $$r^{(1)} = (1 - \tau) P^\top r^{(0)} + \tau q$$
* **Global PR**: $q = (1/4, 1/4, 1/4, 1/4)^\top$
* **PPR**: $q = \mathbf{1}_A$; restart only at Anjali.

### One-step masses comparison:
| Method | A | B | C | D |
| :--- | :---: | :---: | :---: | :---: |
| **Walk only** | 0 | 1/2 | 1/2 | 0 |
| **Global PR** | 1/8 | 3/8 | 3/8 | 1/8 |
| **PPR at A** | 1/2 | 1/4 | 1/4 | 0 |

* **Global at B**: $\frac{1}{2} \times \frac{1}{2} + \frac{1}{2} \times \frac{1}{4} = \frac{3}{8}$.
* **PPR at A**: $\frac{1}{2} \times 0 + \frac{1}{2} \times 1 = \frac{1}{2}$.
* Each row sums to 1. These are one-step masses.

> *Reference*: Jeh, G., & Widom, J. (2003). *Scaling personalized web search*. In Proceedings of the 12th International Conference on World Wide Web (pp. 271–279). ACM.

---

# Slide 18: Teleportation and long-term probabilities
$$r^* = \frac{1}{2} P^\top r^* + \frac{1}{2} q, \quad r^* = (a, b, c, d)^\top$$

### Global PR · $q = (1/4, 1/4, 1/4, 1/4)^\top$
1. **Balance** · each restart contribution $g = 1/8$:
   $$d = g$$
   $$b = a/4 + g$$
   $$c = a/4 + b/2 + d/2 + g$$
   $$a = c/2 + g$$
2. **Substitute**:
   $$c = 3a/8 + 2g$$
   $$a = 3a/16 + 2g$$
   $$\text{Hence } 13a/16 = 1/4 \implies a = 4/13$$
3. **Long-term mass** · order A, B, C, D:
   $$r^*_{\text{PR}} = \frac{1}{104} (32, 21, 38, 13)^\top$$
   **Cheryl > Anjali > Bala > Danish** · $\text{sum} = 1$.

### PPR at Anjali · $q = (1, 0, 0, 0)^\top$
1. **Balance** · restart contributes $1/2$ only to A:
   $$d = 0$$
   $$b = a/4$$
   $$c = a/4 + b/2$$
   $$a = c/2 + 1/2$$
2. **Substitute**:
   $$c = 3a/8$$
   $$a = 3a/16 + 1/2$$
   $$\text{Hence } 13a/16 = 1/2 \implies a = 8/13$$
3. **Long-term mass** · order A, B, C, D:
   $$r^*_{\text{PPR}} = \frac{1}{13} (8, 2, 3, 0)^\top$$
   **Anjali > Cheryl > Bala > Danish** · $\text{sum} = 1$.

---

# Slide 19: 👥 Which seed changes the recommendation?
**Back to music: item $\to$ user $\to$ item**  
Groups of 2–3: start once at Roots and once at Sunshower. Split mass equally at each node.

* **Edges**: Anjali–R,S; Bala–S,T; Cheryl–T,U; Danish–U,R.
* **Output**: two distributions over R,S,T,U and Anjali’s best unseen track.
* **Question**: Average both distributions. What happens to T versus U?  
*(Timer: 04:00)*

---

# Slide 20: Seed choice and a structural tie
| Seed | Roots R | Sunshower S | Tides T | Uptown U |
| :--- | :---: | :---: | :---: | :---: |
| **R** | 1/2 | 1/4 | 0 | 1/4 |
| **S** | 1/4 | 1/2 | 1/4 | 0 |

* **Equal seed**: $(3/8, 3/8, 1/8, 1/8)$
* **Filter observed R, S**: T and U tie because this graph and seed are symmetric.
* *A finite-depth zero does not establish irrelevance.*

---

# Slide 21: One graph, three ranking ideas
**Anjali’s music graph: same listening links, different ranking mechanisms.**

| Method | Ranking idea | Output |
| :--- | :--- | :--- |
| **PageRank / PPR** | Walk along links. Global or personal $q$ selects where probability mass restarts. | Probability mass; $\text{Sum} = 1$ |
| **BiRank** | Couple user and item scores, with separate priors and symmetric normalization. | Two ranking-score vectors |
| **TriRank** | Add explicit aspects and a third coupled score vector, using review-derived evidence or supplied metadata. | Ranking scores and aspect routes |

> **Takeaway**: An aspect route offers an explanation, not proof of relevance.

---

# Slide 22: BiRank: two coupled score vectors
Users $u$ and items $v$ have separate priors:
$$S = D_u^{-1/2} W D_i^{-1/2}$$
$$v^{(k+1)} = \alpha S^\top u^{(k)} + (1 - \alpha) v^0$$
$$u^{(k+1)} = \beta S v^{(k+1)} + (1 - \beta) u^0$$

For $u^{(0)} = u^0 = \mathbf{1}_A$, $v^0 = \mathbf{0}$, $\alpha = 1/2$:
$$v_R^{(1)} = v_S^{(1)} = \frac{1}{2} \times \frac{1}{2} = \frac{1}{4}$$

> *Reference*: He, X., Gao, M., Kan, M.-Y., & Wang, D. (2017). *BiRank: Towards ranking on bipartite graphs*. IEEE Transactions on Knowledge and Data Engineering, 29(1), 57–71.

---

# Slide 23: TriRank: explicit aspects join the graph
Alternate user, item and aspect scores through degree-normalized links, anchored by query priors.

* **Constructed music metadata**:
  * Roots, Uptown have Guitar, Vocals, Rhythm.
  * Sunshower has Vocals, Rhythm.
  * Tides has Guitar, Rhythm.
* **Aspect nodes**:
  * G · Guitar
  * V · Vocals
  * Y · Rhythm
* **Query Prior**:
  * An explicitly supplied Guitar prior ($A \to \text{Guitar}$) offers a route to Tides alongside listening evidence.
  * Explicit metadata links + optional query prior determine the reachable routes.
  * *Constructed example; no numeric TriRank solution is claimed.*

> *Reference*: He, X., Chen, T., Kan, M.-Y., & Chen, X. (2015). *TriRank: Review-aware explainable recommendation by modeling aspects*. In Proceedings of the 24th ACM International Conference on Information and Knowledge Management (pp. 1661–1670). ACM.

---

# Slide 24: Break — 5 minutes
**WHILE YOU PAUSE · RECOMMENDATION HISTORY**  
### 2017 · Pinterest’s Pixie
Pixie walked Pinterest’s pin–board graph to find related items in real time. Shared boards supplied paths between pins—not just a popularity count (Eksombatchai et al., 2017).

> *When does a shared graph path reveal relevance—and when might it amplify a community’s existing tastes?*

> *Reference*: Eksombatchai, C., Jindal, P., Liu, J. Z., Liu, Y., Sharma, R., Sugnet, C., Ulrich, M., & Leskovec, J. (2017). *Pixie: A system for recommending 3+ billion items to 200+ million users in real-time*. arXiv. https://doi.org/10.48550/arXiv.1711.07601

---

# Slide 25: 03 Learned Propagation
Graph structure becomes a shared embedding operator.

---

# Slide 26: LightGCN: graph-aware collaborative filtering
* **Matrix factorization**: learn user and item vectors, then score their dot product.
* **NGCF bridge**: graph neighbours enrich representations with learned transformations and nonlinearities.
* **LightGCN**: retain normalized neighbour aggregation. Remove feature transforms and nonlinear activations.
* Learn the initial embeddings; combine layers and train for ranking.

> *Reference*: He, X., Deng, K., Wang, X., Li, Y., Zhang, Y., & Wang, M. (2020). *LightGCN: Simplifying and powering graph convolution network for recommendation*. In Proceedings of the 43rd International ACM SIGIR Conference on Research and Development in Information Retrieval (pp. 639–648). ACM.

---

# Slide 27: LightGCN: normalized neighbour aggregation
$$e_u^{(k+1)} = \sum_{i \in \mathcal{N}_u} \frac{1}{\sqrt{d_u d_i}} e_i^{(k)}$$

On the original graph:
$$\frac{1}{\sqrt{2 \cdot 2}} = \frac{1}{2}$$
$$e_A^{(k+1)} = \frac{1}{2} e_R^{(k)} + \frac{1}{2} e_S^{(k)}$$

* Items use the analogous update from user embeddings.
* Every node reads the **previous layer**.

---

# Slide 28: One synchronous layer

### Illustrative 1D initial embeddings
| Node | A | B | C | D |
| :--- | :---: | :---: | :---: | :---: |
| $e_u^{(0)}$ | 1 | 3 | 5 | -1 |

| Node | R | S | T | U |
| :--- | :---: | :---: | :---: | :---: |
| $e_i^{(0)}$ | 2 | 4 | 6 | -2 |

### Computations for Layer 1:
$$e_A^{(1)} = \frac{1}{2} \left( \underbrace{2}_{e_R^{(0)}} + \underbrace{4}_{e_S^{(0)}} \right) = 3$$
$$e_S^{(1)} = \frac{1}{2} \left( \underbrace{1}_{e_A^{(0)}} + \underbrace{3}_{e_B^{(0)}} \right) = 2$$

### Layer 1, all nodes:
* **Users A, B, C, D**: $3, 5, 2, 0$
  * $e_A^{(1)} = (2 + 4)/2 = 3$
  * $e_B^{(1)} = (4 + 6)/2 = 5$
  * $e_C^{(1)} = (6 + (-2))/2 = 2$
  * $e_D^{(1)} = (-2 + 2)/2 = 0$
* **Items R, S, T, U**: $0, 2, 4, 2$
  * $e_R^{(1)} = (1 + (-1))/2 = 0$
  * $e_S^{(1)} = (1 + 3)/2 = 2$
  * $e_T^{(1)} = (3 + 5)/2 = 4$
  * $e_U^{(1)} = (5 + (-1))/2 = 2$

### Next layer uses layer 1:
$$e_A^{(2)} = \frac{1}{2} \left( \underbrace{0}_{e_R^{(1)}} + \underbrace{2}_{e_S^{(1)}} \right) = 1$$

---

# Slide 29: Layer combination and unseen-item scores
**Why combine layers? Different information.**
* **Layer 0**: Anjali’s learned initial embedding — identity and interaction signal from training.
* **Layer 1**: one-hop neighbourhood information through Roots and Sunshower.

For this $K = 1$ example, retain both:
$$e_A = \frac{1}{2} \left( \underbrace{1}_{e_A^{(0)}} + \underbrace{3}_{e_A^{(1)}} \right) = 2$$
Using only layer 1 drops the layer-0 contribution.

With $K = 3$, combine all four layers:
$$e_u = \sum_{k=0}^3 \alpha_k e_u^{(k)}$$
* Layers 0, 1, 2, 3: original, one-, two-, three-hop signals — not just layers 2 and 3.
* Uniform choice: $\alpha_k = 1/(K + 1)$; $K = 1 \implies 1/2$, $K = 3 \implies 1/4$.

### Combine items the same way ($K = 1$):
$$e_T = \frac{1}{2} (6 + 4) = 5, \quad e_U = \frac{1}{2} (-2 + 2) = 0$$

### Score unseen items using combined embeddings:
$$\hat{y}_{AT} = 2 \times 5 = 10, \quad \hat{y}_{AU} = 2 \times 0 = 0$$
* **Tides ranks above Uptown** for these inputs. Filter observed Roots and Sunshower.
* *The equal-seed walk tied T, U. These illustrative embeddings are not evidence that LightGCN is better.*

---

# Slide 30: Propagate, combine, score, train
```python
# Pseudocode: sparse normalized adjacency S_hat
layers = [E0]  # trainable user + item embeddings
for k in range(K):
    layers.append(S_hat @ layers[-1])
E = sum(layers) / (K + 1)
users, items = split_users_items(E)
scores = users @ items.T
```
* All nodes update together through the sparse matrix.
* Training gradients flow back to $E^{(0)}$.
* For Anjali: positive $S$, sampled unobserved $U$:
  $$\Delta = \hat{y}_{AS} - \hat{y}_{AU}, \quad L = -\log \sigma(\Delta) + \lambda \|E^{(0)}\|^2$$
* Held-out interactions stay outside the training graph.
* Filter observed items at recommendation time.

---

# Slide 31: 🎰 Bandit Time: Graph-learning check
**Fix the training graph and layer weights. In this LightGCN model, which parameters does BPR update?**
* **A.** Only the final combined embeddings; the initial embeddings stay fixed after initialization.
* **B. [Correct]** The initial user and item embeddings, through gradients from all propagated layers.
* **C.** A separate embedding table at each layer, updated independently before layer averaging.
* **D.** The initial embeddings and the degree-normalized edge coefficients, jointly optimized by BPR.

*Answer in TempoQuiz and explain where the gradient flows.*

---

# Slide 32: 04 Feedback and Personalization
Longer propagation changes whose signal survives.

---

# Slide 33: Mixing can forget the starting user
* For this regular graph, the ordinary stationary mass is $1/8$ per node.
* **Global teleportation** makes convergence possible, but the same global operator has the same stationary ranking for every start.
* A **finite walk** retains depth-dependent seed information.
* A **personalised restart** keeps a user-specific prior.
* **Deep embedding propagation** can also blur distinctions. This is related to, but not identical with, probability mixing.

---

# Slide 34: 👥 One new edge, whose exposure changes?
**Add Cheryl–Roots to the music graph**  
Groups of 2–3: recompute the symmetric normalized weights for C–T, C–U and the new C–R.

* **Constructed scenario, not a paper result**:  
  NDCG rises $0.30 \to 0.34$; long-tail exposure falls $25\% \to 10\%$.
* **Output**: three weights, one affected route, and one measurable guardrail. Would you deploy?  
*(Timer: 05:00)*

---

# Slide 35: New degrees change existing messages
**Adding Cheryl–Roots changes five normalized edge weights**  
After C–R: $d_C = d_R = 3$.

$$w_{CT} = w_{CU} = \frac{1}{\sqrt{3 \cdot 2}} = \frac{1}{\sqrt{6}}$$
$$w_{CR} = \frac{1}{\sqrt{3 \cdot 3}} = \frac{1}{3}$$

* R–A and R–D also become $1/\sqrt{6}$. Existing messages change even where no new edge appeared.
* *After addition: $\text{deg}(C) = \text{deg}(R) = 3$. Audit changed exposure and relevance separately.*
* **Example guardrail**: long-tail share of eligible top-$K$ impressions, with relevance monitored separately.

---

# Slide 36: Community amplification and feedback
Observed interactions reflect what users had an opportunity to hear.

> [!WARNING]
> More exposure can produce more interactions, stronger graph signal and still more exposure. Similar communities may become harder to escape.

* **Log exposure** as well as interactions.
* **Audit coverage** and subgroup effects.
* **Preserve a route** for unfamiliar but eligible tracks.

---

# Slide 37: Summary
Graph evidence needs an explicit operator and an evaluation goal.

---

# Slide 38: 🔑 Connections can reveal and reinforce taste
* **Representation**: what do nodes and edges mean?
* **Propagation**: probability, coupled score or embedding?
* **Personalization**: seed, prior or learned parameters?
* **Evaluation**: relevance gain with which exposure guardrail?
* **Next week**: multi-objective recommendation.

---

# Slide 39: NEXT WEEK PRE-LECTURE EXERCISE
**Challenge an improvement through one lens**  
Canvas · Post + peer critique · 15 minutes  
Complete by Mon, 19 Oct 2026 · 23:59 SGT  

**Your post · 100–140 words**: name a surface, propose a graph-recommender change, and state a primary metric and benefit. Choose one primary lens below:
* ♠ **Stable requirements**: Preserve eligibility, important item attributes, consistency, or an understandable reason for inclusion.
* ♦ **Discovery**: Help users find relevant unfamiliar items; examine variety and long-tail exposure.
* ♥ **Social influence**: Use social evidence helpfully while limiting conformity pressure or unwanted disclosure.

**Peer reply · 50–70 words**: quote a classmate’s claim. Through their lens, give a case where the primary metric improves but the intended outcome worsens. Name a measurable guardrail to detect this.

*Canvas: initial post by Sun, 18 Oct · 23:59 SGT; peer reply by the Monday deadline. These are design lenses, not labels for people.*  
*Next lecture: turn “better” into explicit objectives and defensible trade-offs.*

---

# Slide 40: NEXT WEEK PREVIEW — WEEK 10
**Multi-Objective Recommendation**  
*(Image generated with GPT)*
