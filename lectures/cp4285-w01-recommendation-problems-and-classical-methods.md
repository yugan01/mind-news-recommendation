# Slide 1: Recommendation Problems and Classical Methods
**CP4285 Modern Recommendation Systems — Week 01**
CP4285 Instruction Team
11 Aug 2026

Scan for the slides: `soc-n.us/cp4285-t2610-w01`

# Slide 2: 01 Overview
**CHANGELOG**
* 10 Aug 2026 — Default assignment deadlines are Mon, 23:59 SGT.
* 10 Aug 2026 — Project teams are 2–4 students; Min may adjust teams to support diversity.
* 10 Aug 2026 — Added essay and Project Mini-team Declaration deadlines.

# Slide 3: Video: Spotify Recommendations

# Slide 4: Recommendations Everywhere
**Everyday surfaces**
Video, music, shopping, news, jobs, dating, maps, social feeds, and course resources.

* **Inputs** — behaviour, context, item attributes, social signals
* **Outputs** — a shortlist, ranking, or exposure decision

> **Warning**
> A recommender does not merely retrieve information. It helps decide what receives attention — and what remains unseen.

# Slide 5: CP Pilot Course
CP4285 is a pilot in the CP428x Emerging Topics in Computer Science series.

* We test and improve content, assessment, labs, tutorials, and delivery for future cohorts.
* Students help by self-learning, building labs and tutorials, slide notes for more comprehensive study.
* And documenting processes, gaps, and improvements.

> **Warning**
> This means that your class notes will rarely be ready before class. We have to create the materials from scratch.

# Slide 6: Bandit Time: Objectives
**Same data, different objective**
Two systems use the same user data but optimise different objectives. What should we expect?

* **A.** They should agree if they observe the same users and items.
* **B.** They may rank different items because the objective changes what gets exposed.
* **C.** They can differ only when one system has more user history.
* **D.** Different objectives change scores, but not the ordering of items.

**Choose an answer; justify it in one sentence.**

▶ Start Timer (00:20)

# Slide 7: AI Voice Mode
We are using AI Voice Mode experimentally to investigate the **ethics of recommenders**: how an interactive system explains trade-offs, frames uncertainty, and responds when we question who receives attention.

It is a conversation partner for inquiry, not an authority or a production recommender. Listen for what it assumes, what it leaves out, and how its wording could shape our judgement.

# Slide 8: Find the Recommender
**Audit one recommendation**

1. Name one recommendation you received recently.
2. What signals might the system have used?
3. What objective might it be optimising?
4. Who or what might be left unseen?

**Compare with a partner; share one observation.**

▶ Start Timer (03:00)

# Slide 9: Ethics Thread
* Recommendation systems allocate **attention, visibility, and opportunity**.
* Historical data and engagement objectives can reproduce or amplify **bias**.
* Personalisation raises questions about **privacy, transparency, exposure, fairness, and stakeholder impact**.

# Slide 10: 02 Schedule

# Slide 11: Roadmap: Course Learning Outcomes
By the end of CP4285, you should be able to:

* **CLO 1:** Compare classical, content-based, collaborative-filtering, and hybrid methods.
* **CLO 2:** Implement matrix-factorization and neural recommenders.
* **CLO 3:** Design rigorous offline evaluation using ranking metrics.
* **CLO 4:** Analyse cold start and propose appropriate remedies.
* **CLO 5:** Critique systems through fairness, privacy, transparency, and stakeholder impact.
* **CLO 6:** Explain sequential, graph-based, multi-objective, and LLM-enhanced systems.
* **CLO 7:** Design and justify an end-to-end recommender pipeline.
* **CLO 8:** Communicate and defend technical designs and results.

# Slide 12: Course Library
**Core textbook**
Charu C. Aggarwal, *Recommender Systems: The Textbook*
Focus of the first third of the course.

**A course library for going deeper**
*(Includes various books on Recommender Systems from Springer)*

# Slide 13: Weekly Topics
| Week | Date | Topic |
| :--- | :--- | :--- |
| 01 | 11 Aug | Recommendation Problems and Classical Methods |
| 02 | 18 Aug | Latent Factor Models |
| 03 | 25 Aug | Evaluation of Recommendation Systems |
| 04 | 1 Sep | Neural Recommendation Models |
| 05 | 8 Sep | Sequential and Session-Based Recommendation |
| 06 | 15 Sep | Retrieval and Ranking Architectures |

# Slide 14: Weekly Topics (cont.)
| Week | Date | Topic |
| :--- | :--- | :--- |
| — | 22 Sep | Recess Week — No Class |
| 07 | 29 Sep | Project Design Critique Workshop |
| 08 | 6 Oct | Learning-to-Rank |
| 09 | 13 Oct | Graph-Based Recommendation |
| 10 | 20 Oct | Multi-Objective Recommendation |
| 11 | 27 Oct | Exploration and Online Learning |
| 12 | 3 Nov | LLMs, Generative Recommendation, Research |

# Slide 15: Class Meetings
* **Time:** Every Tue, 10:00–12:00 SGT
* **Venue:** Seminar Room 12, COM3-01-21

Lectures will be recorded where technology permits — recordings are for **revision only**, not a substitute for attendance.

There are **no tutorial sessions** for this course — We’ll be building the materials for the next cohort.

# Slide 16: 03 Assignments & Grading

# Slide 17: Grading Overview
| Component | Weight |
| :--- | :--- |
| Class Participation | 10% |
| Essays | 20% |
| Group Project | 30% |
| In-Class Quizzes | 10% |
| Final Exam (Mon, 23 Nov 2026, 13:00–15:00) | 30% |

# Slide 18: Class Participation (10%)
1. **In-lecture activities**
2. **Pre-lecture activities**
3. **Pre-Flight Survey and Midterm Survey**
   * *Pre-Flight Survey* (due Week 02; Mon, 17 Aug 2026 23:59 SGT)
   * *Midterm Survey* (due Week 08; Mon, 5 Oct 2026 23:59 SGT)

Complete these on time — they are **free marks**.

> **Warning**
> Missing the Pre-Flight or Midterm Survey will directly reduce your participation grade. Don’t forget to do them!

# Slide 19: Essays (20%)
**Two individual take-home essays**, each worth **10%**

* **Essay 1:** Due Week 03; Mon, 24 Aug 2026 23:59 SGT
* **Essay 2:** Due Week 08; Mon 5 Oct 2026 23:59 SGT

Short written analyses of recommender-system methods, evaluation choices, and social implications (8%). Randomised peer review counts towards the remaining 2%.

**AI tools are permitted as a resource** — but an AI declaration is **mandatory** for each submission.

> **Note**
> Essays assess your own critical thinking. AI may assist but must not substitute your analysis. So do them on your own, and don’t let AI do the hard work of understanding; the process is more important than the outcome.

# Slide 20: Project (30%)
* Groups have **2–4 students**
* Define a clear problem anchored on a selected **recommendation system dataset**

**Milestones:**
* **Week 03:** Project Mini-team Declaration — Mon, 24 Aug 2026 23:59 SGT
  * Submit a self-formed team or indicate that you need instructor placement
  * Min has the final decision on group formation (may override student-formed groups)
* **Week 07:** Project Design Critique Workshop
* **Week 13:** Final Project Presentation at 29th STePS: Wed, 11 Nov
  * Teams presenting at 29th STePS do **not** submit a project report; other teams submit a report instead, due same Wed, 11 Nov 2026 23:59 SGT

# Slide 21: Recommendation Dataset Catalogue
Click the screenshot to open the live dataset catalogue.
RUCAIBox. (n.d.). RecSysDatasets [GitHub repository]. `github.com/RUCAIBox/RecSysDatasets`

# Slide 22: Quizzes (10%)
* Quizzes are **jointly created by peers and instructors**
* Peers contribute candidate questions, answers, and plausible distractors from their self-learning
  * Min will assign peers randomly to a week and section responsible.
  * Min will curate, verify, and finalise questions so that the assessments remain fair, accurate, and aligned with the course learning outcomes.
* Combined worth **10%** of total marks
* **AI tools must not be used during in-class quizzes.**

> **Note**
> Co-creation makes quizzes part of the pilot course: students help build better learning materials for this cohort and future cohorts.

# Slide 23: Authority Chain
Our slides will be the most authoritative source of truth.

While we will try to get the slides synced with the website or Canvas, use information introduced in lecture as **the definitive source**.

There will likely be errors in the lecture, so check any doubts or questions with Min.

Unless a slide or Canvas announcement specifies otherwise, assignment deadlines are **Mondays, 23:59 SGT**.

> **Note**
> The slide deck is built in Quarto and rendered through reveal.js. The website is also built in Hugo and served using Github Pages. This means any student, member of the public, or AI can inspect all details of the course. Use this to your advantage.

# Slide 24: BREAK · THE RECOMMENDATION DESK
*Mrs. Arabella Murray, librarian, works at her desk while five girls read at a nearby table.*
Digital Public Library of America. (n.d.). Photograph of a librarian at the Dunbar Branch Library, Athens, Georgia, 1944–1948 [Photograph].

▶ Start Timer (05:00)

# Slide 25: 04 Recommendation Problems and Decision Goals
* User–item matrices, feedback, and missingness
* Prediction, top-k ranking, and useful lists
* Signals and classical recommendation strategies
* Cold start and heavy-head/long-tail exposure
* Treatment lenses and safeguards

# Slide 26: The User–Item Matrix
* `u` ∈ `U`: user; `i` ∈ `I`: item
* `r_ui`: observed feedback
* `Ω`: observed `(u, i)` pairs

> **Warning**
> A `?` is usually **unobserved**, not a negative rating.

> **To think about**
> What user types might be represented in the matrix?
> What other forms can “ratings” take?

# Slide 27: From a Score to a Useful List
**Prediction**
Estimate a value such as `r^_ui` for one user–item pair.
*Question:* “Will this user rate this film highly?”

**Top-k ranking**
Order candidate items and present a short list.
*Question:* “Which five films should appear first, now?”

<u>Accurate scores do not automatically make a useful list.</u>

> **To think about**
> Does top-k apply to users as well as items?

# Slide 28: Signals and Usefulness
**Interaction signals**
ratings · clicks · saves · purchases

**Item signals**
metadata · text · images

**Context**
time · device · place

**Social and operational signals**
friends · availability · constraints

**A useful recommendation can be...**
* **relevant** to the task
* **novel** or serendipitous
* **diverse** across the list
* **well covered** across a catalogue
* **explainable** and constraint-respecting

> **Note**
> The right evaluation follows the intended usefulness — not just the available data.

# Slide 29: Signal Hunt
What else could a recommender capture?
1. Name **three** additional signals.
2. Place each in a box at left — or argue for a new box.
3. For one signal, name a benefit and a privacy, fairness, or manipulation risk.

**Pairs:** prepare one signal to report to critique ethical standpoint.

▶ Start Timer (02:00)

# Slide 30: Complete the Classical Methods Table
In pairs, complete the four orange cells. Be ready to justify one answer from the evidence each method uses.

| Approach | Main evidence | Useful when | Early limitation |
| :--- | :--- | :--- | :--- |
| **Popularity** | aggregate interactions | ________ | reinforces the head |
| **Content-based** | item attributes + one user | features are meaningful | ________ |
| **Collaborative filtering** | patterns across users | ________ | sparse or cold data |
| **Knowledge-based** | requirements and constraints | history is weak | ________ |

# Slide 31: Four Classical Starting Points
| Approach | Main evidence | Useful when | Early limitation |
| :--- | :--- | :--- | :--- |
| **Popularity** | aggregate interactions | little personal data | reinforces the head |
| **Content-based** | item attributes + one user | features are meaningful | narrow, familiar lists |
| **Collaborative filtering** | patterns across users | history is available | sparse or cold data |
| **Knowledge-based** | requirements and constraints | history is weak | needs elicitation |

# Slide 32: When the Matrix Has No Answer
**Cold start**
* **New user:** no observed preferences
* **New item:** no interaction evidence
* **New system:** few observations overall

**A response is a design choice**
* ask for requirements or examples
* use item metadata
* use a transparent popularity fallback
* reserve exposure for new or long-tail items

# Slide 33: Bandit Time: What should this system optimise?
A campus library has sparse borrowing history and a student asks for “a short, accessible book on climate migration.”

* **A.** Maximise predicted rating from the sparse matrix only.
* **B.** Use stated requirements and explain the match.
* **C.** Show the globally most-borrowed books. 
* **D.** Remove every item with little interaction history.

▶ Start Timer (00:20)

# Slide 34: “What types of users are there?”

# Slide 35: Key Users to Consider
We’ll use **card suits** as *User Archetypes* (treatment lenses) to reason about the outcomes. These are not demographic labels or permanent personalities. They are **situational lenses** for asking what a recommender should protect or provide.

| User Archetype | Ask the system to consider |
| :--- | :--- |
| ♣ **Clubs** — adversarial / shill | manipulation, coordinated abuse, robustness |
| ♥ **Hearts** — social | social evidence, influence, conformity risk |
| ♦ **Diamonds** — novelty seeker | freshness, diversity, serendipity, new-item exposure |
| ♠ **Spades** — loyalist / metadata-driven | stable attributes, filters, consistency, explanation |

# Slide 36: Four Different Safeguards
* ♣ **Clubs**: Can coordinated ratings force a target item into the list?
* ♥ **Hearts**: Does a social cue inform the user without creating unwanted exposure or conformity?
* ♦ **Diamonds**: Does the list include genuinely new, long tail, or emerging options?
* ♠ **Spades**: Can the user keep an important attribute fixed and understand why an item appears?

“Good recommendation” changes when the risk, task, and user situation change.

# Slide 37: Reprise: Find the Recommender
Return to the recommendation you audited in Section 01.
1. Name the likely signal and objective.
2. Is it predicting a response or ranking a shortlist?
3. Choose one user archetype to critique: a requirement, diversity/coverage target, explanation, or abuse check.

**Pairs: write a four-line audit.** critique one omitted stakeholder or assumption; challenge its answer.

▶ Start Timer (03:00)

# Slide 38: 05 Neighborhood-Based Collaborative Filtering (CF)
* Rating types, sparse matrices, and long-tail evidence
* User–user and item–item neighbourhoods
* Similarity measures: Pearson, raw cosine, adjusted cosine, and confidence
* Worked examples: from shared evidence to a similarity score
* From a chosen neighbourhood to a prediction

# Slide 39: Rating Types
* Continuous ratings
* Forced-choice ratings
* Ordinal ratings
* Binary ratings
* Unary ratings
* Explicit ratings
* Implicit ratings

> **Note**
> **Missing-value imputation:** do not treat an unobserved explicit rating as a default score; pre-substitution can introduce bias.

> **To think about**
> How do missing values impact training and testing in recommendation systems?

# Slide 40: The Long Tail
* A small heavy head receives most ratings.
* The long tail holds many items with little evidence.
* Thin overlap makes neighbourhood similarity uncertain.
* Head-only ranking can reinforce what is already visible.

**The tail is under-observed, not empty.**

# Slide 41: From a Sparse Matrix to a Neighborhood
**Standard notation**
* `I_u`: items observed for user `u`
* `U_i`: users who observed item `i`
* `I_u ∩ I_v`: shared evidence for a user pair
* `P_u(j)`: the neighbours of `u` who rated target item `j`

**User-based CF**
Find users who rate overlapping items similarly.

**Item-based CF**
Transpose the question: find items that receive similar patterns of ratings.

> **Warning**
> **Similarity is not a fact**
> It is an estimate from a partial overlap. With sparse data, two users may have no common ratings at all.

# Slide 42: Similarity I: Pearson Correlation
* **Overlap:** only mutually rated items contribute.
* **Centred Rating:** rating minus that user’s mean, so generous and harsh raters are put on a comparable baseline.
* **Normalisation:** turns aligned variation into a bounded correlation-like score.

**Use it when**
Rating-scale differences matter.

**Watch for**
Very small overlap can create an unstable correlation.

# Slide 43: Worked Example I: Pearson Correlation
**User–user example: users 1 and 6**

* **Step 1 — begin with the same overlap**
  `I_{u_1} ∩ I_{u_6} = {i_1, i_3, i_4}`
* **Step 2 — subtract each user’s own mean**
  `μ_1 = 11/3`, `μ_6 = 4`
  `s_1 = (4/3, -5/3, 1/3)`, `s_6 = (1, 1, -2)`
* **Step 3 — correlate the centred vectors**
  `Pearson(u_1, u_6) = -1 / (√(14/3)√6) = -0.189`

# Slide 44: Similarity II: Raw Cosine
* **Overlap:** compare only entries both users supplied.
* **Dot Product:** large ratings in the same positions increase similarity.
* **Vector Length:** corrects for the overall magnitude of each observed rating vector.

**Use it when**
Raw magnitude is meaningful or ratings are already on a comparable scale.

**Watch for**
It does not remove a user’s tendency to rate high or low.

# Slide 45: Worked Example II: Raw Cosine
**Users 1 and 6**

* **Step 1 — keep only shared ratings**
  `I_{u_1} ∩ I_{u_6} = {i_1, i_3, i_4}`
* **Step 2 — form the two rating vectors**
  `r_1 = (5, 2, 4)`, `r_6 = (5, 5, 2)`
* **Step 3 — normalise their dot product**
  `raw-cosine(u_1, u_6) = 43 / (√45√54) = 0.872`

> **To think about**
> What are the differences between cosine and Pearson here?

# Slide 46: Similarity III: Adjusted Cosine for Items
* **Shared Raters:** users who rated both items.
* **User-Centred Rating `S_ui`:** remove each rater’s personal baseline before comparing item columns.
* **Normalisation:** compare the centred item vectors fairly.

**Use it when**
Building item-to-item neighbourhoods from explicit ratings.

**Watch for**
It still needs sufficient shared raters; centring does not solve sparse overlap.

> **To think about**
> What makes this an “adjusted” metric? Why don’t we need a similar adjustment for user-based cosine?

# Slide 47: Worked Example III: Item–Item
**Items `i_1` and `i_5` after user mean-centring**

`adjusted-cosine(s_{i_1}, s_{i_5}) = 2.56 / (√3.56√3.56) = 0.719`

> **Note**
> The item vectors are `s_{i_1}` and `s_{i_5}`; their entries are user-centred ratings for the shared raters. This is the same cosine operation, but across item columns rather than user rows.

# Slide 48: From Item Neighbours to a Prediction
* `Q_t(u)` — valid item neighbours of target `t` that user `u` rated
* **Similarity `sim(t, i)`**: adjusted-cosine similarity between target and neighbour item
* **Observed rating `r_ui`**: the user’s rating of that neighbour item

**Target: recommend `i_5` to `u_1`**
* **Valid evidence:** `Q_{i_5}(u_1) = {i_1, i_3, i_4}`
* These are items that **both** resemble target item `i_5` and have an observed rating from `u_1`.
* **Weight `u_1`’s rating of each item by its similarity to `i_5`.**

> **Note**
> The neighbourhood changes with the target item and with the ratings that the user has actually supplied.

# Slide 49: Similarity IV: Significance Weighting
* **Base Similarity:** Pearson, cosine, or another chosen score.
* **Evidence Count:** how many items both users rated.
* **Threshold (β):** the overlap at which no further discount applies.

**Interpretation**
One striking shared rating should not carry the same weight as twenty.

**Role**
This is a confidence modifier, not a replacement for the underlying similarity measure.

> **To think about**
> This is a discount factor for users. Does it make sense for items?

# Slide 50: Worked Example IV: Significance Weighting
**User–user example: users 1 and 6**

* **Step 1 — start from the user-user Pearson score**
  `Pearson(u_1, u_6) = -0.189`
* **Step 2 — count the overlap and set a threshold**
  `|I_{u_1} ∩ I_{u_6}| = 3`, `β = 10`
* **Step 3 — apply the confidence discount**
  `discounted-sim_{β=10} = -0.189 (3/10) = -0.057`

# Slide 51: User–User KNN in RecBole
**After training: score user 196 on item 242**
```python
1 from recbole.quick_start import run_recbole
2
3 run_recbole(
4   model="ItemKNN",
5   dataset="ml-100k",
6   config_dict={
7     "knn_method": "user",
8     "k": 50,
9     "shrink": 100,
10  },
11 )
```
* `knn_method: user` - Compare user rows, instead of its default items.
* `k: 50` - Retain 50 neighbours.
* `shrink: 100` - Regularise thin overlap.

`model.predict(query)` returns the estimated score for that one user–item pair. A recommender would repeat this over candidate items, then rank them.

> **Note**
> 196 and 242 are MovieLens 100K raw IDs. Convert them to RecBole’s internal IDs before prediction.

# Slide 52: Discussion: Which Similarity Axis?
**When would user-based or item-based similarity be more effective?**

* **User-based: compare rows** - When might a person's nearest neighbours be reliable enough to recommend from?
* **Item-based: compare columns** - When might relationships between items be more stable or useful?

**Pairs:** choose a setting, make a claim for one axis, then give one condition that could reverse your choice.
Optional: Consider User Archetypes (♣ · ♥ · ♦ · ♠)

▶ Start Timer (03:00)

# Slide 53: Key Point: Similarity Is a Modelling Choice
Choose similarity for the evidence you have—not because one metric is universally best.
* **Pearson**: Corrects for rating generosity; thin overlap is still unstable.
* **Raw cosine**: Keeps raw magnitude; it also keeps scale bias.
* **Adjusted cosine + weighting**: Changes the comparison axis or discounts weak overlap.

Ask which entries, baselines, and confidence assumptions shape the neighbours.

> **To think about**
> We can use similarity thresholds to calculate a neighbourhood instead of truncating to top `k`. When is that a good idea?

# Slide 54: Neighbourhood Choices
**Who enters the neighbourhood?**
* **Top-k**: Keep the closest fixed number of valid neighbours.
* **Similarity threshold**: Keep everyone above a meaningful score; the neighbourhood size can vary.
* **Filter weak or negative scores**: Avoid letting low-confidence or oppositional evidence dominate an estimate.

**How should their evidence count?**
* **Mean-centred prediction**: Account for different user rating baselines.
* **Z-score normalisation (not covered)**: Also account for how widely each user uses the rating scale.
* **Other variants**: Amplify stronger similarities or aggregate votes for categorical ratings.

> **To think about**
> Which choice changes *who* is eligible, and which changes *how* their evidence is aggregated?

# Slide 55: From Neighbours to a Prediction
* **Neighbourhood `P_u(j)`**: the top-`k` similar users who rated item `j`.
* **Similarity weight**: closer neighbours contribute more.
* **Baseline and deviation**: predict a deviation, then add the target user’s own mean back.

**Pipeline**
1. choose a similarity
2. select valid top-`k` neighbours
3. aggregate their evidence
4. rank candidate items by `r^_uj`

# Slide 56: Practicality: Making Neighbourhoods Work
**Offline: prepare the neighbourhoods**
* Compute user-user or item-item similarities.
* Store selected peer groups rather than every possible comparison when appropriate.
* Refresh them as ratings, users, and items change.

**Online: serve a recommendation**
* Combine the target’s valid neighbours to make one prediction.
* Score many candidate items to produce a ranked top-`k` list.
* Balance latency, memory, freshness, and coverage—not accuracy alone.

> **Note**
> User-based similarity scales with user pairs; item-based similarity scales with item pairs.

> **To think about**
> Which work can be done before a request arrives, and which work must wait for the target user and item?

# Slide 57: Section 05 Summary: Neighbourhood CF
1. **Represent**: Start with sparse user item evidence and make missingness visible.
2. **Compare**: Choose an axis, similarity, centring rule, and confidence treatment.
3. **Predict & rank**: Select valid neighbours, aggregate evidence, then rank candidates.

Neighbourhood CF is transparent and local—but its outputs inherit the data’s sparsity, exposure patterns, and modelling choices.

# Slide 58: Summary
A recommender is never only an algorithm: it is a choice about evidence, objectives, and consequences.

**Three ideas to carry forward**
* **Frame:** define users, items, evidence, and usefulness.
* **Compare:** neighbourhood CF turns sparse overlap into a local estimate.
* **Question:** ask who is missing and whose outcome is served.

# Slide 59: NEXT WEEK PRE-LECTURE EXERCISE
**Look for Hidden Structure**
Canvas discussion board · post before Mon, 23:59 SGT
Post a short, concrete response to this question; then reply constructively to one classmate.

*Name one latent dimension that could explain several ratings—but is not an item metadata label.*

Next week: turn this intuition into latent-factor representations of the user-item matrix.

# Slide 60: NEXT WEEK PREVIEW — WEEK 02
**Latent Factor Models**
What if useful similarities are not directly visible in a sparse matrix?
* **Move beyond local overlap** — infer compact user and item representations from many ratings together.
* **Interpret sparse evidence** — ask which hidden preferences may explain observed patterns.
* **Keep the same questions** — what evidence is represented, what does the model optimise, and who might it leave unseen?
