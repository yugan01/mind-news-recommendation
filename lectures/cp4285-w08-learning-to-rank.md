# Slide 1: Learning-to-Rank
**CP4285 Modern Recommendation Systems — Week 08**
CP4285 Instruction Team
06 Oct 2026

Scan for the slides: `soc-n.us/cp4285-t2610-w08`

# Slide 2: 01 Announcements and Context
**CHANGELOG**
* 6 Oct 2026 — Added feedback and Canvas Voices; retrofitted activity templates; introduced original LTR papers and a compact McAuley BPR summary; developed varied calculation builds and a timed Kendall’s tau exercise; removed the objective-clinic activity.
* 5 Oct 2026 — Added the next-week prelecture exercise, submission requirements, and lecture preview.
* 24 Aug 2026 — Refreshed the course slide deck and changelog.

# Slide 3: 🎰 Bandit Time: Start TempoQuiz
**TEMPOQUIZ HOST LAUNCH**
**Open the room before Question 1**
Open the host dashboard, then ask students to join with the live QR code or room code shown there.

**LIVE QR + ROOM CODE**
Use the host dashboard — do not add a static code here.

**HOST DASHBOARD CUE**
Week 07 Recap -> Release question 1
Wait for the room to settle, then release the first recap question.

# Slide 4: WEEKS 06 + 07 RECAP

# Slide 5: Week 8: Learning-to-Rank
**Learning outcomes**
* Formulate ranking objectives.
* Compare ranking approaches.
* Analyze position bias.
* Discuss fairness implications of ranking.

# Slide 6: Midterm feedback: survey ratings
23 responses per item · Bar segments show counts · A + SA = agree + strongly agree

### Module
| Criterion | Counts | A + SA |
| :--- | :--- | :---: |
| Organisation / understanding | 2 Disagree, 5 Neutral, 14 Agree, 2 Strongly agree | 16/23 |
| Appropriate workload | 2 Neutral, 16 Agree, 5 Strongly agree | 21/23 |
| Sufficient communication | 2 Neutral, 13 Agree, 8 Strongly agree | 21/23 |
| Regular attendance (self-report) | 4 Agree, 19 Strongly agree | 23/23 |
| Sufficient Canvas site | 3 Disagree, 13 Agree, 6 Strongly agree | 19/23 |
| Would recommend the module | 4 Neutral, 18 Strongly agree | 18/23 |

### Lecturer
| Criterion | Counts | A + SA |
| :--- | :--- | :---: |
| Prepared for class | 2 Neutral, 11 Agree, 10 Strongly agree | 21/23 |
| Clear and well-paced | 2 Neutral, 10 Agree, 10 Strongly agree | 20/23 |
| Encourages participation | 7 Agree, 15 Strongly agree | 22/23 |
| Responsive to feedback | 8 Agree, 14 Strongly agree | 22/23 |
| Passion for the subject | 9 Agree, 14 Strongly agree | 23/23 |

### Lecture materials
| Criterion | Counts | A + SA |
| :--- | :--- | :---: |
| Comprehensible content | 5 Disagree, 2 Neutral, 10 Agree, 6 Strongly agree | 16/23 |
| Interesting and engaging | 3 Neutral, 14 Agree, 6 Strongly agree | 20/23 |
| Lecture–assignment fit | 2 Disagree, 2 Neutral, 14 Agree, 5 Strongly agree | 19/23 |
| Valuable materials | 4 Disagree, 12 Agree, 6 Strongly agree | 18/23 |
| Available in good time | 2 Neutral, 11 Agree, 10 Strongly agree | 21/23 |

*Scale: Strongly disagree, Disagree, Neutral, Agree, Strongly agree*

* **What should we preserve?**
  * Participation and responsiveness: each has 22/23 agree or strongly agree.
* **What would make learning clearer?**
  * Content comprehensibility: 5/23 disagree. Materials’ value: 4/23 disagree. Which concrete change would help?

*CP4285 teaching team. (2026). Midterm survey [Course survey]. Canvas. Respondents only; not a claim about the whole class.*

# Slide 7: Midterm feedback: learning from lectures
Anonymous student excerpts · Theme labels, colour and normal/italic type distinguish clusters

* **Structure & clarity**
  * “The point being brought across each slide is not clear, nor is the link or connection between the different content.”
  * “vague sentences that sound legit but don't really help with learning.”
* **Worked examples & practice**
  * “lecturer can use whiteboard more to explain concepts, especially the mathematical explanations”
  * “more relevant practice examples so that we can apply the concepts more effectively.”
* **Technical depth & deployments**
  * “Some concepts taught are still a little high-level for my taste”
  * “a short skim or walkthrough of more papers / real world deployments such as FAISS”
* **Peer discussion**
  * “having more opportunities to discuss in-class with other group members around us can help us retain what has been taught in class better.”

**Discuss:** Which change would help most: clearer sequencing, worked examples, or optional deeper walkthroughs?

*CP4285 teaching team. (2026). Midterm survey [Course survey]. Canvas. Selected verbatim phrases; theme headings are instructor summaries.*

# Slide 8: Midterm feedback: assessment & course design
Anonymous student excerpts · Theme labels, colour and normal/italic type distinguish clusters

* **Quiz timing & question quality**
  * “The midterm quiz timing is confusing (25min actual vs 30min time limit in Canvas)”
  * “Questions could have been made clearer I feel.”
* **Essay scope & grading criteria**
  * “There could also be clearer rubrics or deliverables to the essays.”
  * “I think the essay was too open ended in general to write a good response”
* **Participation-mark allocation**
  * “maybe can convert the admin items that weighs 1% to be assigned for the Tempo quizzes.”
* **Team formation**
  * “I think its better to let students form their whole team.”
  * “groupmates do not have the same baseline knowledge of the subject.”

**Discuss:** What can we clarify now—and which proposals need a course-level decision?

*CP4285 teaching team. (2026). Midterm survey [Course survey]. Canvas. Selected verbatim phrases; theme headings are instructor summaries.*

# Slide 9: LAST WEEK PRE-LECTURE EXERCISE · W07
**One list, two plausible orders**
Canvas discussion board · complete by **Mon, 5 Oct 2026 · 23:59 SGT**

> Choose five items from a recommendation domain you used this week. Order the same set twice for different reasons. Explain the reason by identifying one item whose position changes, and explaining why.

**Your post:** Name the domain, list both complete orders, and explain the changed position.
Then post a constructive reply to one classmate’s post.
Name and apply one treatment lens in your comment: ♠ ♦ ♥ ♣.

# Slide 10: LAST WEEK PRE-LECTURE EXERCISE · CANVAS VOICES

### Travel & transport
* **Travel · cost vs novelty**
  * **POST:** “If simply travelling is the goal i.e just leave Singapore then Malaysia is the most cost-effective destination to go to.”
  * **REPLY · ♠ SPADES:** “Your two orders are almost exact reverses, so "novelty" ends up just meaning "far and expensive."”
* **Transport · cost vs effort**
  * **POST:** “… walking ranks last as it requires the most exertion, especially over longer distances or in hot and hazy weather.”
  * **REPLY · ♦ DIAMONDS QUESTION:** “… it would be interesting to see how the Diamonds lens can be factored in …”

### TikTok & YouTube feeds
* **TikTok · engagement vs discovery**
  * **POST:** “In order B, ranked for diversity and discovery, the ramen video moves up to the top.”
  * **REPLY · ♦ DIAMONDS:** “This can make the FYP feel less repetitive and improve discovery.”
* **YouTube · personal interest vs trends**
  * **POST:** “If I am trying to relax and watch my favourite YouTuber, then I would like to see Order A.”
  * **REPLY · ♥ SOCIAL LENS:** “… although relying heavily on trends could also create conformity where users mainly see what is already popular.”

# Slide 11: 02 Ranking Objectives

# Slide 12: One ranked list, three learning targets

* **Pointwise**
  * **Is this item relevant?**
  * Learn a score or label for each user–item pair independently.
  * Useful when explicit and calibrated scores are given.
* **Pairwise**
  * **Which of two items comes first?**
  * Learn that an observed or preferred item should outrank a comparison item.
  * Useful when relative preference is clearer than a rating.
* **Listwise**
  * **How should the candidate group rank?**
  * Define a loss using the candidate group, such as a target distribution.
  * Position-sensitive metric alignment depends on the training loss and update weighting rules.

# Slide 13: Sourcing Internships for Maya
All four roles satisfy the student’s hard constraints. Grades are independent assessments of fit, constructed for this example; urgency is held fixed.

| Company | Akari | Banyan | Chandra | Dempsey |
| :--- | :---: | :---: | :---: | :---: |
| **Current interface position** | 1 | 2 | 3 | 4 · hidden |
| **Assessed relevance grade $y_i$** | 1 | 2 | 0 | 3 |
| **Current model score $s_i$** | 2.5 | 2 | 1.5 | 1 |
| **Alternative model score $s_i^{\text{alt}}$** | 2.5 | 3.5 | 1.5 | 4.5 |
| **Alternative interface position** | 3 | 2 | 4 · hidden | 1 |

Two hypothetical models: alternative scores are the assessed grades plus 1.5. Sorting them gives **Dempsey → Banyan → Akari → Chandra**.

> **Note**
> Only the three highest-scoring roles are shown. The gray Dempsey column marks its exclusion under the current model; the alternative displays Dempsey first and excludes Chandra.

# Slide 14: Pointwise: estimate relevance one item at a time
$$\hat{y}_{ui} = f(u, i, x_{ui})$$

* **Training question**
  * “How relevant is this internship to this student?”
  * Compare a predicted score with a label, rating, click, or conversion proxy.
* **Ranking limitation**
  * Good independent scores do not directly say which swap in the final list matters most.
  * A small score error at rank 1 can be more consequential than a larger one far below the fold.

# Slide 15: Lower prediction error can give the worse list
As a contrast to ranking quality, assume we measure relevance-prediction error:
$$L_{\text{point}} = \frac{1}{4} \sum_{i \in I} (s_i - y_i)^2$$

Company order: Akari, Banyan, Chandra, Dempsey. Fixed grades $y = [1, 2, 0, 3]$. The current model gives $s = [2.5, 2, 1.5, 1]$; the alternative gives $s^{\text{alt}} = [2.5, 3.5, 1.5, 4.5]$.

$$L_{\text{current}} = \frac{(2.5 - 1)^2 + (2 - 2)^2 + (1.5 - 0)^2 + (1 - 3)^2}{4} = \frac{2.25 + 0 + 2.25 + 4}{4} = 2.125 \approx 2.12$$

$$L_{\text{alternative}} = \frac{(2.5 - 1)^2 + (3.5 - 2)^2 + (1.5 - 0)^2 + (4.5 - 3)^2}{4} = \frac{4(1.5^2)}{4} = 2.25$$

**Smaller MSE, worse top three:** current scores exclude Dempsey. The alternative has a uniform +1.5 score error, yet gives the ideal order **Dempsey, Banyan, Akari, Chandra**.

# Slide 16: Why not differentiate the ranking metric?
True ratings: Nero = 0, Gladiator = 1. Recommend only the higher-scoring movie.

* **(a) Smooth RMSE objective** vs **(b) Non-smooth hit rate**
* Figure 13.1: Ranking objective is not a smooth function of predicted ratings (and underlying model parameters)
* *Source: Charu C. Aggarwal, Recommender Systems: The Textbook (Springer, 2016), Figure 13.1, p. 414. Figure reproduced from the textbook; © Springer International Publishing Switzerland 2016.*

**Flat regions give no useful gradient; the ordering boundary gives a jump.** A smooth pairwise surrogate supplies a learning direction without differentiating the discrete metric.

# Slide 17: Foundational paper: learning from relative evidence
Joachims (KDD 2002): [Optimizing Search Engines using Clickthrough Data](https://doi.org/10.1145/775047.775067) 🔗

* **Training evidence**
  * Derive query-specific document preferences from clickthrough logs.
  * Clicks supply relative evidence under assumptions about the displayed ranking.
* **Clicked below a skipped link**
  * Clicking link **3** after passing link **2** suggests **3** should rank above **2**.
  * Clicking link **7** suggests it should rank above the skipped links **2, 4, 5 and 6**.
  * Assumes top-to-bottom scanning and informative abstracts. These are partial, potentially noisy preferences—not absolute relevance labels.

> **Why no preference pair from link 1?**
> Link 1 was clicked, but there are no skipped links above it. This rule therefore extracts no pair from that click. It also gives no ordering between the clicked links 1, 3 and 7.

# Slide 18: 👥 From preference pairs to a training metric
You have five inferred preferences: 3 above 2; 7 above 2, 4, 5 and 6. How would you reward a model for respecting them?

* **Compare two proposed rankings**
  * **Ranker X:** 1 → 2 → 3 → 4 → 5 → 6 → 7
  * **Ranker Y:** 1 → 7 → 3 → 2 → 4 → 5 → 6
  * Count respected and reversed preference pairs for each ranker. Ignore pairs for which you have no evidence.
* **Design the metric**
  * Propose a normalized score—or a loss to minimize. State its best and worst values.

**Groups · 5 minutes:** write the two scores and your metric definition, including its best and worst values. Then: 1 minute to report and discuss.

> **Hint · Kendall’s tau (complete rankings, no ties)**
> $\tau = (P - Q) / (P + Q)$: $P$ counts concordant pairs (same order); $Q$ counts discordant pairs (reversed order).
> For $m$ items, $P + Q = \binom{m}{2}$; $+1$ is perfect agreement, $-1$ complete reversal. Adapt the idea to the five observed pairs; unknown preferences are not errors. This is not full-list tau.

▶ Start Timer (05:00)

# Slide 19: Original RankNet paper: learning to rank by gradient descent
Burges et al. (ICML 2005): [Learning to Rank using Gradient Descent](https://doi.org/10.1145/1102351.1102363) 🔗

* Score each query–document feature vector with a shared neural network.
* Turn score differences into pairwise preference probabilities.
* Train with probabilistic pairwise loss and backpropagation.

> **Note**
> Training compares pairs. Serving still scores each candidate and sorts the scores.

# Slide 20: Pairwise: learn the direction of a comparison
Compare items within the same user or query.

* **Desired preference**
  * Grades: Dempsey = 3; Akari = 1.
  * Dempsey should rank above Akari.
* **Current model disagrees**
  * Scores: Dempsey = 1; Akari = 2.5.
  * $\Delta = s_{\text{Dempsey}} - s_{\text{Akari}} = 1 - 2.5 = -1.5$
  * Training should increase this margin.

| Training family | Pair evidence | Training question |
| :--- | :--- | :--- |
| **Pairwise ranking** | Grades or explicit preferences | Does Dempsey score above Akari? |
| **BPR** | Observed interaction vs sampled unobserved item | Does the observed item score higher? |

**BPR is a method in the pairwise family.** Grades give six strict pairs: Dempsey > Banyan > Akari > Chandra.

# Slide 21: Worked pairwise example: Dempsey vs Akari
Return to the assessed-grade pair Dempsey > Akari. Scores are $s_{\text{Dempsey}} = 1$, $s_{\text{Akari}} = 2.5$:

**Calculate:**
$$\Delta = s_{\text{Dempsey}} - s_{\text{Akari}}, \quad \sigma(\Delta) = 1/(1 + e^{-\Delta}), \quad L = -\ln \sigma(\Delta)$$

$$\Delta = 1 - 2.5 = -1.5, \quad \sigma(-1.5) = \frac{1}{1 + e^{1.5}} = 0.1824, \quad L = -\ln \sigma(-1.5) \approx 1.701$$

$$\frac{\partial L}{\partial \Delta} = \sigma(\Delta) - 1 = -0.8176, \quad \frac{\partial L}{\partial s_{\text{Dempsey}}} = -0.8176, \quad \frac{\partial L}{\partial s_{\text{Akari}}} = +0.8176$$

For independent score parameters and $\eta = 0.1$:
$$s'_{\text{Dempsey}} = 1 - 0.1(-0.8176) = 1.0818, \quad s'_{\text{Akari}} = 2.5 - 0.1(+0.8176) = 2.4182$$

Shared models propagate these signals through their parameters; these are illustrative score updates.

# Slide 22: RankNet: a probabilistic pairwise loss
For one query/context, define the score margin $\Delta = s_i - s_j$. Target $t_{ij}$ expresses whether item $i$ should outrank item $j$.

$$P_{ij} = \sigma(s_i - s_j), \quad L_{ij} = -t_{ij} \ln P_{ij} - (1 - t_{ij}) \ln(1 - P_{ij})$$

* **Preference probability $P = \sigma(\Delta)$**
  * Score margin $\Delta = s_i - s_j$
  * Point $(0, 0.5)$
  * Dempsey–Akari: $P = 0.182$
  * Negative margin: $j$ scores higher · Positive margin: $i$ scores higher
* **Training loss**
  * Strict preference: $t = 1$
  * Tie target: $t = 0.5$ (minimum at $\Delta = 0$)
  * Soft preferences: $t = 0.6, 0.75, 0.9$
  * Dempsey–Akari: $L = 1.701$
  * Training increases this margin

> **Tie target ($t = 0.5$):** equal scores minimize loss. Our internship examples use strict preferences ($t = 1$).

# Slide 23: BPR: pairwise ranking from implicit feedback
$$T = \{(u, i, j) : i \in I_u, \, j \notin I_u\}, \quad \Delta_{uij} = s(u, i) - s(u, j)$$

Here $I_u$ contains the items user $u$ interacted with. $T$ defines assumed training preferences, not proven dislikes.

* **Objective versus model**
  * BPR specifies the training criterion.
  * BPR-MF uses matrix factorisation to supply the score.
  * Other differentiable scoring models can use the same pairwise loss.
* **Observed does not mean universally preferred**
  * A sampled unobserved item may be unseen, not disliked. The sampler determines which comparisons training encounters.

# Slide 24: BPR: turn preferred orderings into learning updates
**McAuley §5.2.2:** learn preferred orderings, not absolute ratings.

* **1 · Define success**
  * $x_{uij} = s(u, i) - s(u, j)$
  * Positive $\to i$ ranks above $j$.
  * Observed $i$ vs sampled unseen $j$. AUC counts correct pairs; the hard count has no useful gradient.
* **2 · Use a smooth loss**
  * $P = \sigma(x_{uij})$: predicted pair preference.
  * BPR assumes the strict target $t = 1$.
  * $L_{\text{pair}} = -\ln P = -\ln \sigma(x_{uij})$
  * Same cross-entropy as S22 with $t = 1$. Sum pair losses + L2 regularization; minimize $L$.
* **3 · Update the model**
  * $\partial L / \partial x = P - t = \sigma(x_{uij}) - 1$
  * Upward correction: $t - P = 1 - \sigma(x_{uij})$.
  * With $t = 1$, lower predicted preference means a stronger correction. The chain rule updates the model parameters.
* **4 · See the direction**
  * Dempsey > Akari: scores 1 vs 2.5.
  * Margin $-1.5 \to$ probability 0.1824.
  * Loss $1.701 \to$ push Dempsey up, Akari down.
  * Grade-based illustration of the loss; BPR uses implicit-feedback pairs.

# Slide 25: Break — 5 minutes
**WHILE YOU PAUSE · RECOMMENDATION HISTORY**
### 2010 · Yahoo’s ranking challenge
An ensemble of LambdaMART models won Track 1 of the Yahoo! Learning to Rank Challenge. Search ranking made a practical case for learning from errors that matter to the ranked list (Burges et al., 2011).

*Why should a swap near rank 1 carry more learning pressure than a swap near rank 40?*

*Burges, C., Svore, K., Bennett, P., Pastusiak, A., & Wu, Q. (2011). [Learning to rank using an ensemble of lambda-gradient models](https://proceedings.mlr.press/v14/burges11a.html) 🔗. In O. Chapelle, Y. Chang, & T.-Y. Liu (Eds.), Proceedings of the Learning to Rank Challenge (Vol. 14, pp. 25–35). PMLR.*

▶ Start Timer (05:00)

# Slide 26: 03 Listwise Ranking & LambdaRank

# Slide 27: Listwise quality: measure the displayed order
Relative order, not position weighting: basic BPR has no explicit NDCG swap weights.

### 1 · Convert grades to gains
$$g_i = 2^{y_i} - 1$$

| Company | Grade | Gain computation |
| :--- | :---: | :--- |
| Akari | 1 | $2^1 - 1 = 1$ |
| Banyan | 2 | $2^2 - 1 = 3$ |
| Chandra | 0 | $2^0 - 1 = 0$ |
| Dempsey | 3 | $2^3 - 1 = 7$ |

Gains come from assessed grades. Dempsey contributes zero to the current DCG@3 because it is at rank 4.

### 2 · Discount gains by position
$$\text{DCG}@3 = \sum_{r=1}^3 \frac{g_{\pi_r}}{\log_2(r + 1)}$$
Rank 1/2/3 discounts: 1, 0.631, 0.5.

* **Current:** Akari → Banyan → Chandra
  $$\text{DCG} = 1(1) + 3(0.631) + 0(0.5) = 2.893$$
* **Ideal:** Dempsey → Banyan → Akari
  $$\text{IDCG} = 7(1) + 3(0.631) + 1(0.5) = 9.393$$

### 3 · Normalize
$$\text{NDCG}@3 = \frac{2.893}{9.393} \approx 0.308 \cdot \text{Ideal} = 1.$$

# Slide 28: ListNet: learn from a candidate-group distribution
Cao et al. (ICML 2007): [Learning to Rank: From Pairwise Approach to Listwise Approach](https://doi.org/10.1145/1273496.1273513) 🔗

* **One training example = one candidate group**
  * Fix a query or user/context, then take its candidates and relevance grades together.
  * For Maya: all four internships form one example. Compute one loss across the group, rather than a separate loss for each item or pair.
* **“Top-one” gives every item a probability**
  * The model scores every candidate. Softmax turns these scores into a distribution: each value expresses that item's probability of occupying first place under the model.
  * Sort all probabilities to rank the whole group. This gives exactly the same order as sorting the original scores.

$$p_i = \frac{e^{y_i}}{\sum_j e^{y_j}}, \quad q_i = \frac{e^{s_i}}{\sum_j e^{s_j}}, \quad L_{\text{list}} = -\sum_i p_i \log q_i$$

$p_i =$ target top-one probability, derived from item $i$’s assessed relevance grade. $q_i =$ predicted top-one probability, derived from its model score. Each distribution sums to 1 across the candidate group.

*Training: bring $q$ closer to $p$. Serving: sort model scores; softmax preserves their order.*

# Slide 29: ListNet: turn four scores into one group loss
Company order: Akari, Banyan, Chandra, Dempsey · Grades $y = [1, 2, 0, 3]$ · Scores $s = [2.5, 2, 1.5, 1]$

### 1 · Exponentiate, then normalize
$$Z_y = e^1 + e^2 + e^0 + e^3 = 31.193$$
$$Z_s = e^{2.5} + e^2 + e^{1.5} + e^1 = 26.772$$

Divide each exponential by its group total:
* Akari: $p_{\text{Akari}} = e^1 / 31.193 = 0.0871$
* Dempsey: $p_{\text{Dempsey}} = e^3 / 31.193 = 0.6439$
* Akari: $q_{\text{Akari}} = e^{2.5} / 26.772 = 0.4551$
* Dempsey: $q_{\text{Dempsey}} = e^1 / 26.772 = 0.1015$

### 2 · Compare the distributions
$$p_{\text{target}} \approx [0.0871, \, 0.2369, \, 0.0321, \, 0.6439]$$
$$q_{\text{predicted}} \approx [0.4551, \, 0.2760, \, 0.1674, \, 0.1015]$$
Each sums to 1. The target favours Dempsey; the model favours Akari.

### 3 · Cross-entropy (natural logs)
$$L = -\sum_i p_i \ln q_i$$
$$L \approx -[0.0871 \ln(0.4551) + 0.2369 \ln(0.2760) + 0.0321 \ln(0.1674) + 0.6439 \ln(0.1015)] \approx 1.904$$

* **Wrong order · Dempsey vs Akari**
  * Grades: $3 > 1$ · Scores: $1 < 2.5$
  * Item gradients $(q - p)$:
    * Dempsey: $0.1015 - 0.6439 = -0.5424$
    * Akari: $0.4551 - 0.0871 = +0.3680$
  * Update: Dempsey ↑; Akari ↓. Large corrections.
* **Correct order · Banyan vs Chandra**
  * Grades: $2 > 0$ · Scores: $2 > 1.5$
  * Item gradients $(q - p)$:
    * Banyan: $0.2760 - 0.2369 = +0.0391$
    * Chandra: $0.1674 - 0.0321 = +0.1353$
  * Update: Banyan ↓ slightly; Chandra ↓ more.

# Slide 30: LambdaRank: metric-aware pairwise learning
Burges, Ragno and Le (NeurIPS 2006): [Learning to Rank with Nonsmooth Cost Functions](https://papers.nips.cc/paper/2006/hash/4a2820f65b6df20430985a70106b8396-Abstract.html) 🔗

* **RankNet does not account for NDCG**
  * Its pairwise loss uses score margins, without NDCG’s rank discounts, top-K cutoff or swap-impact weights.
  * Equal margins give equal strict-pair corrections—even when one swap would improve NDCG much more.
* **LambdaRank adds NDCG sensitivity**
  * Scale each pairwise learning signal by the absolute change in NDCG caused by swapping those items.
  * With the same margin, a larger NDCG improvement produces a stronger correction.

> **Note**
> LambdaRank specifies score-update signals and propagates them through a scorer. LambdaMART later uses such signals with boosted trees.

# Slide 31: LambdaRank: weight a comparison by its list impact
Same RankNet variables: $\Delta = s_i - s_j$, $P_{ij} = \sigma(\Delta)$, strict target $t_{ij} = 1$. New weight: $w_{ij} = |\Delta\text{NDCG}@3_{ij}|$.

$$\lambda_{ij} = w_{ij}(t_{ij} - P_{ij}) = w_{ij}(1 - P_{ij}) = \frac{w_{ij}}{1 + e^\Delta} \quad (t_{ij} = 1)$$

* **Swap-impact weight $w$ (ranking held fixed)**
  * Dempsey–Akari: $w = 0.639$
  * Dempsey–Chandra: $w = 0.373$
  * Recompute $w$ when the ranking changes.
* **Update magnitude $\lambda = w / (1 + e^\Delta)$**
  * Curves for $w = 0.1, 0.25, 0.5, 1$ over score margin $\Delta = s_i - s_j$
  * Akari pair: $(-1.5, \, 0.522)$
  * Chandra pair: $(-0.5, \, 0.232)$

Swap impact scales the curve; margin selects the point. Positive $\lambda$ pushes the preferred item’s score up and the other down; sum pair signals through the model.

# Slide 32: Two swaps: work out their top-three utility
Current order Akari, Banyan, Chandra, Dempsey; gains 1, 3, 0, 7. Current DCG = 2.893, IDCG = 9.393. Beyond rank 3, the discount is zero.

### 1 · Dempsey ↔ Akari: new top three Dempsey, Banyan, Chandra.
$$\text{DCG}' = 7(1) + 3(0.631) + 0(0.5) \approx 8.893, \quad \text{NDCG}' \approx 8.893 / 9.393 = 0.947$$
$$w_{\text{Dempsey},\text{Akari}} \approx |0.947 - 0.308| = 0.639$$

### 2 · Dempsey ↔ Chandra: new top three Akari, Banyan, Dempsey.
$$\text{DCG}' = 1(1) + 3(0.631) + 7(0.5) \approx 6.393, \quad \text{NDCG}' \approx 6.393 / 9.393 = 0.681$$
$$w_{\text{Dempsey},\text{Chandra}} \approx |0.681 - 0.308| = 0.373$$

# Slide 33: BPR in the learning-to-rank map

| Method | Training signal | Correction |
| :--- | :--- | :--- |
| **Pointwise** | One item and its label | Item-level prediction error |
| **RankNet** | Relevance preference pair | Logistic pairwise error |
| **BPR** | Observed / unobserved user pair | Logistic error + pair sampling |
| **LambdaRank** | Pair in the current ranking | Pairwise signal × swap impact |
| **ListNet** | Candidate-group distribution | Group cross-entropy |

**BPR-MF = a matrix-factorisation scorer trained with BPR.** Neither a neural scorer nor BPR alone specifies fair exposure.

# Slide 34: 04 Position Bias & Fair Ranking

# Slide 35: Click counts need exposure context
A job card gets **60 clicks at rank 1** and **18 clicks at rank 6**.

Without impression counts or evidence about examination, we cannot tell whether this reflects relevance or simply more opportunity to be seen.

# Slide 36: 👥 Evidence audit: what can you conclude?
Same job card, comparable audiences. Data are constructed; examination probabilities are supplied assumptions.

| Rank | Impressions | Clicks | Assumed $P(E)$ |
| :---: | :---: | :---: | :---: |
| 1 | 1,000 | 60 | 0.60 |
| 6 | 1,000 | 18 | 0.18 |

* Raw CTR = clicks / impressions
* Adjusted estimate = raw CTR / assumed $P(E)$

**Groups · 5 minutes · one deliverable:** write a two-sentence verdict on the claim “The card is more relevant at rank 1.” Support your verdict with the exposure-adjusted estimates and one limitation of that evidence.

▶ Start Timer (05:00)

# Slide 37: Position bias: clicks mix relevance with opportunity
Clicks combine examination opportunity and response after examination.

$$P(C = 1 \mid i, r) = P(E = 1 \mid r) \, P(C = 1 \mid E = 1, i)$$

**Model assumptions:** clicks require examination; examination depends on rank; response after examination is rank-invariant. Clicks remain a proxy for relevance.

* **Log exposure**
  * Record impressions, rank, and context.
  * Unseen is not rejected.
* **Test carefully**
  * Use randomisation or defensible propensity-aware methods.
* **Audit allocation**
  * Check who receives exposure; specify a fairness guardrail.

# Slide 38: Summary

# Slide 39: 🔑 A ranking objective encodes which errors matter most.

* **1 · Decide what counts as evidence.**
  Clicks can supply partial, noisy preference pairs; relevance grades express an assessed ordering. Declare the assumptions behind either signal.
* **2 · Choose what training should correct.**
  Pointwise reduces item prediction error; RankNet learns pair preferences; BPR constructs strict pairs from implicit feedback; ListNet matches candidate-group probability distributions.
* **3 · Make position matter when the goal requires it.**
  Lower prediction error can still give a worse top-three list. RankNet has no explicit NDCG weighting; LambdaRank scales pair corrections by NDCG swap impact.
* **4 · Separate relevance from opportunity.**
  Exposure changes clicks. Adjusted estimates depend on examination assumptions; equal estimates alone do not establish fairness.

Next week: use graph connections to explore evidence beyond an item’s interaction count.

# Slide 40: NEXT WEEK PRE-LECTURE EXERCISE
**Popularity or connection? Fix the surface**
Canvas · Solo · 15 minutes · no peer reply
Complete by **Mon, 12 Oct 2026 · 23:59 SGT**

Choose one surface and keep its user, task, and constraints fixed throughout your answer.

* **A · Music autoplay**
  Next track after an acoustic-folk playlist. Keep the genre transition gentle.
* **B · Jobs search**
  First search-result card for a graduate seeking entry-level data work in Singapore.
* **C · Grocery checkout**
  One “Complete your meal” checkout item for a vegetarian dinner basket. It must be available.

Both candidates are eligible: **X** has 1,000 interactions but no known overlap with this user’s interests; **Y** has 20 interactions, several from users with overlapping histories. Counts come from the same period and interaction definition.

**Post 120–160 words:** name your surface and concrete X/Y items; choose which to investigate first; state one assumption, one check of exposure, and evidence that would reverse your choice.

Next lecture: ask what the pattern of connections reveals beyond an item’s total count.

# Slide 41: NEXT WEEK PREVIEW — WEEK 09
**Graph-Based Recommendation**
*(Image generated with GPT)*
