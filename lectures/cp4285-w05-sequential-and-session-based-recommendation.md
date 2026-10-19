# Slide 1: Sequential and Session-Based Recommendation
**CP4285 Modern Recommendation Systems — Week 05**
CP4285 Instruction Team
08 Sep 2026

Scan for the slides: `soc-n.us/cp4285-t2610-w05`

# Slide 2: 01 Announcements and Context
**CHANGELOG**
* 6 Sep 2026 — Added the Weeks 01–05 quiz-question contribution and Week 07 quiz details.
* 5 Sep 2026 — Added the Week 04 recap and Week 06 preview.

# Slide 3: WEEK 04 RECAP

# Slide 4: Quizzes - From Week 01 (10%)
* Quizzes are **jointly created by peers and instructors**
* Peers contribute candidate questions, answers, and worked solutions where needed
  * Min assigns each student a week, lecture section, difficulty, and question format.
  * Min curates, verifies, and finalises questions so that assessments remain fair, accurate, and aligned with the course learning outcomes.
* Combined worth **10%** of total marks
* **AI tools must not be used during in-class quizzes.**

> **Note**
> Co-creation makes quizzes part of the pilot course: students help build better learning materials for this cohort and future cohorts.

# Slide 5: Peer Quiz Question-Bank Contribution
Covers **Weeks 01–05**. Your identifier determines one lecture section and one difficulty: **easy, medium, or hard**.

**Your assigned contribution**
* **Two MCQ/MRQ questions**, **two fill-in-the-blank questions**, or **one calculation question** with full working
* Include the answer for every question.

Submit on Canvas by **Mon, 21 Sep 2026, 23:59 SGT**. You have two weeks to prepare.
**5 points (1% of total marks) for completion only.**

**Week 07 (Open Book) Quiz · Tue, 29 Sep 2026**
~30 minutes at the beginning of class. A mix of instructor and peer questions, completed on paper and proctored.

# Slide 6: 02 Lecture Overview

# Slide 7: LAST WEEK PRE-LECTURE EXERCISE
**Where Does One Session End?**
Canvas discussion board · post before Mon, 7 Sep 2026 · 23:59 SGT

Post one **100–140 word** decision memo. No classmate reply is required.

> A person asks Ask Maps for a 2pm lunch route, explores nearby cultural sites, then later asks for a halal eatery open after 10pm. Does the late-evening query begin a new session? Answer yes or no and cite two trace details.

Next week: sequences make order, recency, and local context part of the recommendation signal.
*(Optional reading: The Straits Times, 11 Aug 2026)*

# Slide 8: LAST WEEK PRE-LECTURE EXERCISE · CANVAS VOICES
**NO · Retain the session**
* “I assume all three queries … are issued sequentially in one sitting, with no idle gap.”
* “The word ‘then’ implies that the query is on the same itinerary.”
* “I would keep the same session while placing more emphasis on the most recent request.”
* “A good recommender should avoid recommending it twice for lunch and again for supper.”

**YES · Begin a new session**
* “Session data should only contain information … useful to the current context, whilst more general preferences … can be persisted across sessions.”
* “Adding previous search context into an explicit ask is un-expected and might not be good UX.”
* “Starting a new session does not mean discarding all earlier information.”
* “That is roughly eight hours of inactivity, well beyond a typical session timeout window of thirty to sixty minutes.”

# Slide 9: Which interaction history is available?
* **Session-based:** Use the ordered events in the current session when no history is available.
* **Session-aware:** Combine the current session with earlier sessions from an identified user.
* **Sequence-aware:** Model a longer ordered history when earlier interactions may still matter.

*(Source: Ravanmehr, R., & Mohamadrezaei, R. (2024). Session-based recommender systems using deep learning)*

# Slide 10: Week 05: What changes when interactions have an order?
**Week 04**
Attention selected useful features for a user–item score.

**Week 05**
A sequence model selects useful **earlier actions** to predict the next item.

**By the end of today, you should be able to:**
* model temporal preferences;
* implement next-item prediction;
* compare static and dynamic representations; and
* critically assess engagement-driven objectives.

# Slide 11: 03 Session-Based Models

# Slide 12: The immediate-intent problem
**Observed now**
“late-night halal food” → nearby restaurant pages → delivery times

**Do not infer too much**
A short trace can reveal a task without revealing a permanent identity or preference.

**Session-based recommendation:** predict a next item from the ordered interactions in the current, often anonymous, session.

> **Note**
> **“Often anonymous” includes logged-in use:** a system can keep the current session separate from the long-term profile, letting a temporary task guide the next recommendation without becoming permanent.

# Slide 13: Bandit Time: What should this system use?
An anonymous visitor opens three restaurant pages in sequence during a late-night visit. Before recommending the next page, which should be the system’s primary representation of the visitor’s immediate intent?

* **A.** A permanent “late-night diner” profile inferred from IP location and device locale
* **B.** The ordered prefix of pages in the current session
* **C.** An unordered bag of the three pages, ranked by citywide click through rate
* **D.** The highest-rated restaurant page the visitor has not opened

▶ Start Timer (00:30)

# Slide 14: Sessionisation is a modelling decision
**1. Temporal context**
* Timestamp and inactivity gap
* Visit duration and event pace

**2. Content and task context**
* Query terms and item categories
* Action types and stated constraints

**3. Interaction and situational context**
* Navigation path and device or browser
* Location and account continuity

> **Note**
> The boundary is an operational rule, not direct proof of intent.

> **To think about**
> A device, location, or account signal may improve next-item accuracy. What user expectation or privacy cost should limit its use?

# Slide 15: Deciding what starts a new session?
Your group owns one session-boundary method. Apply it to the Ask Maps trace, then compare it with the shared 30-minute inactivity baseline.

**Shared simplified implementation**
* **Fixed inactivity:** begin a new session after 30 minutes with no event. It is clear and cheap, but one threshold cannot fit every task.

1. **Adaptive timeout:** Change the gap using event pace, device, time of day, or product context.
2. **Task similarity:** Split when queries, items, or actions indicate a different task.
3. **Boundary classifier:** Learn a same-task or new-task decision from labelled traces.
4. **Intent change point:** Infer when the likely hidden intent state changes.

▶ Start Timer (03:00)

# Slide 16: From an event log to next-item examples
`view ramen` → `open halal filter` → `inspect delivery time` → `predict: late-night restaurant`

| Observed prefix | Training target |
| :--- | :--- |
| `[ramen]` | halal filter |
| `[ramen, halal filter]` | delivery time |
| `[ramen, halal filter, delivery time]` | late-night restaurant |

# Slide 17: GRU4Rec
Paper: Hidasi, B., Karatzoglou, A., Baltrunas, L., & Tikk, D. (2016). Session-based Recommendations with Recurrent Neural Networks. ICLR.

# Slide 18: Video: GRU intuition

# Slide 19: GRU4Rec: carry a changing session state
*(Diagram of Gated Recurrent Unit)*

* **Update gate `zt`**: Call out the final blend: how much of the previous hidden state should persist versus be replaced by the candidate state?
* **Reset gate `rt`**: Call out the path into the candidate state: how much earlier context should shape the next proposal?
* **Read the diagram**: Both gates receive the current input and previous state. Their outputs shape the candidate and the final blend.

# Slide 20: GRU4Rec architecture: one event in, scores out
*(Diagram showing Input -> Embedding layer -> GRU layer(s) -> Feedforward layers -> Output: scores on items)*

# Slide 21: GRU4Rec implementation and evaluation lens
* **Batch**: One next-item step per active row.
* **Reset on replacement**: End → next session; state = zero.
* **Evaluate forward in time**: Later sessions: HR, MRR, NDCG@K.

# Slide 22: GRU4Rec evaluation: BPR and TOP1
The comparison evaluated two ranking objectives: BPR and TOP1. For the observed next item `i` and a sampled negative `j`, both losses want `r_i > r_j`.

* **BPR · push the observed item above a negative**
  `L_BPR = -log σ(r_i - r_j)`
  Only the relative score gap matters.
* **TOP1 · approximate rank, then stabilise scores**
  `L_TOP1 = σ(r_j - r_i) + σ((r_j)^2)`
  The first term penalises a negative that outranks the observed item; the second term keeps negative scores near zero.

> **Note**
> Orange: observed next item · Teal: sampled negative · Navy: TOP1 regulariser

# Slide 23: Key Point: GRU4Rec turns a session into a next-item ranking
Use the **current session** to update a state, then rank what could plausibly come next.

* **1 · Observe order:** Read the sequence of actions within one session.
* **2 · Update state:** A GRU decides what earlier context to carry forward.
* **3 · Rank next items:** Score candidate actions for the next step—not a permanent profile.

**Predict a next action with humility: a session state is useful, but it is not a person.**

# Slide 24: Break — 5 minutes
WHILE YOU PAUSE: ☕ BREAK
*(Evolution of recommendation algorithms timeline diagram)*
Nicoomanesh, A. (2024). Evolution of recommendation algorithms, Part I...

▶ Start Timer (05:00)

# Slide 25: 04 Sequential Recommendation

# Slide 26: From the current session to a user sequence
* **Session state:** What seems useful for the task happening now?
* **Sequential user state:** Which earlier actions still matter after preferences evolve?

# Slide 27: Bandit Time: Compare GRU4Rec and a new sequence model
A new causal self-attention model is proposed as an alternative to GRU4Rec for next-item ranking. Which comparison makes a fair deployment claim?

* **A.** Use the same time cutoff, but report each model’s strongest metric from its original paper.
* **B.** Let self-attention read the full test sequence, while GRU4Rec receives only the prefix.
* **C.** Give GRU4Rec an anonymous session but give the new model each user’s complete earlier history.
* **D.** Use one chronological cutoff; give both models only each target’s preceding prefix, then use the same candidates and ranking metric.

▶ Start Timer (00:45)

# Slide 28: Why self-attention?
* **GRU4Rec — previous focus:** A recurrent encoder compresses the prefix into a changing hidden state.
* **SASRec — current focus:** Self-attention lets each prediction selectively use different earlier actions. 

# Slide 29: SASRec: causal self-attention for next-item prediction
`item embeddings + position embeddings` → `masked self-attention` → `contextual sequence representation` → `score next items`

* **Causal mask:** At position `t`, the model cannot inspect future actions.
* **Position matters:** The same items in a different order can imply a different next item.

# Slide 30: SASRec walkthrough: one next-item recommendation at a time
**Causal self-attention over the available session prefix**
`museum ticket (earlier)` → `vegetarian café (earlier)` → `late-night halal food (current query)`

* **Step 1 · attention scores:** 0.08, 0.22, 0.70
* **Step 2 · causal softmax weights:** 8%, 22%, 70%
* **Step 3 · contextual state:** The model gives the late-night halal action the largest weight, while retaining the earlier context.

**Ranked next-item recommendations**
* **1. delivery-time check:** highest-ranked next item (based on the full prefix)
* **2. halal restaurant details:** re-rank after the new observed action
* **3. order tracking:** re-rank again with the longer prefix

Only past actions are visible.

# Slide 31: Preserve chronology in the learning pipeline
`earlier history` → `train sequence model` → `later held-out interaction` → `rank candidates`

* **A random split leaks the future:** It lets the model learn from actions that had not happened at recommendation time.
* **Causal masking applies the same rule within a sequence:** Before softmax, set scores for future positions to −∞. Each prediction can use its prefix, never a later action.

# Slide 32: Transformer recommenders: related tasks, different masks
| Model framing | Training visibility | Natural use |
| :--- | :--- | :--- |
| **SASRec** | Causal: past only | Next-item prediction |
| **BERT4Rec-style** | Masked positions with bidirectional context | Masked-item training; adapted to next-item ranking at inference |

# Slide 33: Worked trace: what should the model attend to?
**Trace:** `vegetarian café` → `museum ticket` → `“late-night halal food”` → `delivery-time check` → `?`

Name one action likely to be highly relevant, one action that may be weakly relevant, and one uncertainty that a log alone cannot resolve.

# Slide 34: Long-term profile or sharp current intent?
For the worked trace, decide what should influence the next recommendation, what must not overwrite the profile, and one non-click outcome that would show helpfulness.

Write a two-sentence answer. Be ready to name one temporal signal, one uncertainty, and one risk of optimizing only engagement.

▶ Start Timer (03:00)

# Slide 35: Choose the smallest sufficient temporal model
| Evidence and need | Reasonable starting point |
| :--- | :--- |
| One immediately preceding action is enough | Markov / co-occurrence baseline |
| Short, changing anonymous intent | GRU4Rec or session-aware method |
| Long ordered history with selective dependencies | SASRec-style sequential Transformer |

# Slide 36: 05 Summary

# Slide 37: Time changes what a preference signal means.
* **Session**: Local intent can be useful without defining a person.
* **Sequence**: Order and recency can change which history matters.
* **Objective**: A next click is not the whole definition of help.

# Slide 38: NEXT WEEK PRE-LECTURE EXERCISE
**When should the current session outweigh history?**
A user’s long-term history is vegetarian cafés. In their current session, they search “late-night halal food”, open nearby restaurant pages, and check delivery times. What should the app consider now—and what should it not assume?

Canvas: 120–140 words by Mon, 14 Sep 2026, 23:59 SGT. Name two session-relevant item types, one risk of overwriting durable preference, and one non-click helpfulness signal.

Next week: retrieve plausible options before ranking and re-ranking them.

# Slide 39: NEXT WEEK PREVIEW — WEEK 06
**Retrieval and Ranking Architectures**
A production recommender cannot compare every catalogue item in one step. It first retrieves plausible candidates, then ranks and re-ranks the short list for the user and situation.

**In Week 06:**
* trace the candidate-generation → ranking → re-ranking pipeline;
* compare retrieval and ranking objectives; and
* examine two-tower architectures and visibility trade-offs.
