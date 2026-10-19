# Project Direction

Team anchor for the revised project scope. This supersedes the original TF-IDF-versus-embeddings framing, which Prof Kan judged aligned but "not very substantial" (see `instructor-feedback.md`).

## Core question

Among embedding-based recommenders — generic semantic meaning (frozen text encoder), collaborative structure (LightGCN on the click graph), and task-learned representations (two-tower trained on impression labels) — which produces the best personalized news rankings, for which users, and at what cost in diversity, novelty, and exposure?

All three embed users and articles in a shared vector space and score by similarity, differing only in where the embeddings come from. BPR-MF matrix factorization and training-click popularity serve as classical baselines, and BPR-MF versus LightGCN doubles as a controlled ablation of what graph propagation contributes. The already-implemented TF-IDF is kept as a bonus reference row at no extra cost.

## Models

1. Popularity baseline — completed. Non-personalized floor.
2. TF-IDF — completed. Bonus reference row; no longer a project pillar.
3. BPR-MF — classical collaborative baseline; also builds the shared BPR training loop that LightGCN reuses.
4. Frozen semantic embeddings (`all-MiniLM-L6-v2`) — generic text meaning; encode articles once, cache vectors, average latest-20 history vectors, cosine scoring.
5. LightGCN — embeddings learned from the user-article click graph through normalized neighbor aggregation, trained with BPR.
6. Two-tower — extension only, after the core comparison is stable; user tower from history, article tower from text, trained on impression click labels.

All models share identical inputs: title + abstract text, latest 20 history articles (eligibility: history >= 10; impressions with at least one clicked and one non-clicked candidate), the same users and impressions.

## Evaluation

- Accuracy: NDCG@5 primary, MRR secondary, user-level paired bootstrap confidence intervals for every model difference. Final numbers come from the supplied validation split run once, after all development moves to a chronological dev split carved from the training data (the validation split was already used for the 5,000-user smoke check and this will be disclosed).
- For whom: subgroup analysis by history length, interest narrowness (category entropy of history), and user connectivity in the click graph. A per-user answer is the substantive finding; the aggregate ranking is secondary.
- Beyond accuracy: intra-list diversity, novelty (inverse popularity of ranked articles), catalogue coverage, and exposure concentration across users.
- LLM-as-judge: on a sampled user subset, an LLM sees the user's history and each model's top-5 and rates alignment, novelty, diversity, and article quality against a fixed published rubric, with manual spot-check calibration.

## Requirements mapping

- Classical baseline: BPR-MF (collaborative filtering / matrix factorization), plus popularity. 
- Neural or advanced model: frozen embeddings, LightGCN, two-tower.
- Rigorous offline evaluation: NDCG@5, MRR, bootstrap CIs.
- Error analysis: failure-case comparison across models (where each representation fails).
- Ethical analysis: clicks are not satisfaction; exposure bias; privacy of reading histories; topic-bubble concentration; popularity amplification — grounded in the exposure and diversity metrics above.
