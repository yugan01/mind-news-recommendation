# Project objective and work plan

The project builds a personalized news recommender using a user's previously clicked articles. For every recommendation event, the system receives the user's click history, the set of candidate articles shown, and each article's title and abstract; it must rank the candidates so clicked articles appear near the top.

The revised direction, core question, model list, and evaluation plan are anchored in `project-direction.md`. This document tracks completed work, remaining work, and responsibilities.

## Work completed so far

- Public GitHub repository created.
- MIND-small dataset downloaded and parsed.
- Popularity model implemented.
- TF-IDF model implemented (now a bonus reference row).
- NDCG@5 and MRR evaluation implemented.
- User-level bootstrap confidence intervals implemented.
- Prediction exports implemented.
- Automated tests added.
- A fixed 5,000-user preliminary experiment completed on the supplied validation split (13 Oct, seed 4285): TF-IDF 0.2929 vs popularity 0.2301 NDCG@5, bootstrap CI entirely positive.

The implementation and results are in the [GitHub repository](https://github.com/yugan01/mind-news-recommendation).

## Work that still needs to be done

### 1. Establish the development and final-evaluation split

The smoke run used the supplied validation data, so its results are preliminary. From this point onward:

- Create a chronological development/tuning split from the training data.
- Develop and debug all remaining models only on that split.
- Freeze model and preprocessing decisions.
- Run the supplied validation set once as the final reported comparison, disclosing the earlier pipeline check.

### 2. Frozen semantic embeddings

Encode every article's title + abstract once with `all-MiniLM-L6-v2` (frozen), cache the vectors, average the latest 20 history vectors per user, rank impression candidates by cosine similarity, and evaluate on exactly the same users and impressions as the other models.

### 3. BPR-MF baseline and shared training loop

Implement matrix factorization trained with BPR on impression click labels. This is the classical collaborative baseline and also establishes the shared BPR training loop (embedding tables, pair sampling, loss) that LightGCN reuses.

### 4. LightGCN

Add normalized neighbor aggregation over the user-article click graph on top of the BPR-MF training loop. BPR-MF versus LightGCN is the controlled ablation of what graph propagation contributes.

### 5. Two-tower extension (last)

Only after the core comparison is stable: a user tower over reading history and an article tower over article text, trained on impression click labels, compared separately against the frozen and graph methods.

### 6. Beyond-accuracy metrics

On the prediction exports: intra-list diversity, novelty (inverse popularity of ranked articles), catalogue coverage across users, and exposure concentration. These run on existing and future prediction CSVs without model changes.

### 7. LLM-as-judge panel

Sample a user subset; prompt an LLM with each user's history and each model's top-5; rate alignment, novelty, diversity, and article quality against a fixed rubric published with the project; manually spot-check calibration before the full run.

### 8. Statistical comparison and subgroup analysis

For each model pair: NDCG@5, MRR, mean paired difference, 95% user-level bootstrap confidence interval. Subgroups: history length (10-19 vs 20+), interest narrowness (category entropy), user connectivity in the click graph. Also history-length sensitivity (latest 5/10/20 on users with at least 20).

### 9. Failure analysis

Inspect examples where keyword matching succeeds but embeddings fail and vice versa, where graph structure fails on cold-start or niche users, and where popularity unexpectedly performs best. Feed representative cases into the presentation.

### 10. Ethical and limitation analysis

Clicks do not necessarily represent satisfaction; users can only click articles they were shown; reading histories are privacy-sensitive; recommendations may create narrow topic bubbles; popularity and graph methods may over-promote already-popular content; the experiment ranks displayed candidates and cannot evaluate unseen articles. Ground these claims in the exposure and diversity metrics where possible.

### 11. Final deliverables

Comparison charts, confidence-interval chart, subgroup and history-length analyses, representative success and failure examples, ethical discussion, defensible conclusion, poster or slide deck, and a clear contribution statement for each team member.

Open item (undecided): whether to redo the project critique deck for feedback on the new direction (Prof Kan offered; old deck still counts for grades) and whether to present at the optional Week 10-12 critique session. No owner assigned until the team decides.

## Responsibilities

Percentages stay 30/30/20/20. Implementation deadline 26 Oct; analysis and deliverables ready by 9 Nov for the 11 Nov presentation. K = Kevin, M = Marcus, Y = Yugan, S = Sood.

- Kevin, 30% — infrastructure and evaluation tooling: dev/final split discipline; BPR-MF and the shared BPR training loop; frozen semantic embeddings and an initial run testing them; beyond-accuracy metric code; LLM-as-judge pipeline.
- Marcus, 30% — neural graph and trained models: LightGCN on the click graph (reusing the shared training loop); two-tower extension after the core comparison is stable.
- Yugan, 20% — evaluation and testing: paired statistical comparisons and bootstrap CIs across all model pairs; subgroup analyses; history-length sensitivity; failure-case collection; test coverage for the new components. His earlier pipeline and baseline implementation counts toward this share.
- Sood, 20% — deliverables and reporting: poster/presentation design and assembly; final report editing for consistent methods, figures, and claims; ethical analysis write-up; contribution statement; submission coordination and rehearsals.

Handoffs: Kevin supplies Marcus with the shared training loop and the evaluation interface early; by 26 Oct Kevin and Marcus supply Yugan and Sood with working code, completed experiments, per-example predictions, and method notes so the analysis and deliverables proceed without further implementation help.
