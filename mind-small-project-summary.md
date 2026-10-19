# News Recommendation with Text Embeddings

CP4285 group project proposal — 5 October 2026. Revised 19 October 2026 after instructor feedback (see `docs/instructor-feedback.md`); the current direction is anchored in `docs/project-direction.md` and the live work plan is `docs/project-stuff-to-do.md`.

## 1. Core idea and thesis

Recommend news articles using a user's previous article clicks and the text of candidate articles. The central question is which signal best personalizes news ranking, for which users, and at what cost beyond accuracy.

**Core question:** Among embedding-based recommenders — generic semantic meaning (a frozen pretrained text encoder), collaborative structure (LightGCN on the click graph), and task-learned representations (a two-tower model trained on impression labels) — which produces the best personalized news rankings, for which users, and at what cost in diversity, novelty, and exposure?

All three embed users and articles in a shared vector space and score by similarity, differing only in where the embeddings come from. BPR-MF matrix factorization and training-click popularity serve as classical baselines, and BPR-MF versus LightGCN doubles as a controlled ablation of what graph propagation contributes. The already-implemented TF-IDF is kept as a bonus reference row.

This revision responds to Prof Kan's feedback that a bare TF-IDF-versus-embeddings accuracy comparison is "not very substantial": the comparison now spans multiple embedding sources, per-user subgroups, beyond-accuracy metrics, and an LLM-as-judge panel.

## 2. Dataset and feasibility

Use [MIND-small](https://msnews.github.io/), a news recommendation dataset containing article text, previous clicks, and impression logs. An impression is a set of articles displayed to a user at one recorded time.

The relevant data are:

- Article records: article ID, category, title, and abstract.
- Behavior records: impression ID, user ID, time, ordered history of previously clicked article IDs, and displayed article IDs with click labels.

The [official documentation](https://github.com/msnews/msnews.github.io/blob/master/assets/doc/introduction.md) defines these fields. Labels indicate clicked or not clicked; they do not establish whether the user read or liked the full article.

Our feasibility audit measured history lengths by counting each user once per split. It combined the user and impression IDs in the [RecZoo copy](https://huggingface.co/datasets/reczoo/MIND_small_x1) with uncapped title histories in the [MTEB copy](https://huggingface.co/datasets/mteb/mind_small), checking the corresponding history lengths.

- Training: 50,000 users; mean history of 18.5 articles and median of 11; 236,344 clicked candidate entries across 156,965 impressions.
- Validation: 50,000 users; mean history of 25.5 articles and median of 15; 111,383 clicked candidate entries across 73,152 impressions.
- Validation includes 32,109 users with at least 10 history articles, contributing 50,844 impressions with both clicked and non-clicked candidates.
- Validation impressions contain an average of 37.5 candidate articles, with a median of 23.

This provides enough examples for a controlled comparison. The initial experiment will use users with at least 10 history articles. The existing history and later impression labels provide the input and outcome; we do not need to mask additional history articles.

## 3. Experiment design

### Data setup

Use the supplied training data for development and model training; carve a chronological development/tuning split from it and develop every remaining model only on that split. The supplied validation data is the final evaluation set: it was used once for a fixed 5,000-user pipeline smoke check on 13 October (popularity 0.2301 vs TF-IDF 0.2929 NDCG@5), so final results will disclose that prior use and the validation set will otherwise not be touched until model and preprocessing decisions are frozen.

Start with a fixed random sample of 5,000 eligible users from the development split to check new models. For the final comparison, evaluate all eligible users in the held-out validation set. Retain impressions containing at least one clicked and one non-clicked article.

For each impression, use up to the most recent 20 articles in its supplied history. Rank only the articles displayed in that impression. Use title plus abstract as article text for every text-based method. Evaluation click labels remain hidden until the ranking has been produced.

### Models

1. Popularity baseline (completed): rank candidates by their click counts in the training data.
2. TF-IDF (completed, bonus reference row): fit the word representation on training article text. Average the history article vectors to form a user-interest vector, then rank candidates by cosine similarity to it.
3. BPR-MF: matrix factorization trained with BPR on impression click labels — the classical collaborative baseline, and the shared BPR training loop (embedding tables, pair sampling, loss) reused by LightGCN.
4. Frozen semantic embeddings: encode article text with a pretrained model such as [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2). Average the history article vectors, rank candidates by cosine similarity. Do not fine-tune the encoder.
5. LightGCN: user and article embeddings learned through normalized neighbor aggregation over the user–article click graph, trained with BPR. BPR-MF versus LightGCN isolates what graph propagation contributes.
6. Small two-tower model (extension, last): one network converts the history into a user vector; another converts article features into an article vector. Train on impression click labels so compatible user–article pairs receive higher scores.

The embedding-source comparison (frozen encoder vs LightGCN vs two-tower) is the core; the baselines anchor that text and personalization help at all.

### Example

A user has previously clicked articles about electric vehicles. A later impression shows 23 articles, including stories about batteries, football, and cooking. Each model ranks the same 23 articles without seeing their click labels. We then check whether the articles actually clicked appear near the top.

Precompute article vectors so the same article is encoded once and reused across users. Neither simple text method requires training a separate network for every user.

## 4. Evaluation and interpretation

Use NDCG@5 as the primary metric: it rewards placing clicked articles near the top five positions. Also report MRR, which rewards placing the first clicked article near the top.

Average scores across each user's impressions, then across users. Compare methods on identical examples. Estimate a 95% confidence interval for the paired score difference by resampling users.

Because no single aggregate number answers the core question, add three further evaluation strands:

- Subgroup analysis: users with 10–19 history articles versus 20 or more; interest narrowness (category entropy of the history); user connectivity in the click graph. Also history-length sensitivity using the latest 5, 10, and 20 articles on users with at least 20 available. The substantive finding is which method works for whom, not only the aggregate ranking.
- Beyond-accuracy metrics: intra-list diversity, novelty (inverse popularity of ranked articles), catalogue coverage across users, and exposure concentration — whether a model funnels everyone into the same popular articles.
- LLM-as-judge: on a sampled user subset, an LLM sees the user's history and each model's top-5 and rates alignment, novelty, diversity, and article quality against a fixed published rubric, with manual spot-checks for calibration. This measures recommendation quality where click labels provide no ground truth.

Inspect examples where one representation succeeds and another fails — including graph methods failing on cold-start or niche users — and where popularity unexpectedly performs best.

A method is "better" only relative to a question: better on average, better for a user group, or better beyond accuracy. The project's contribution is the map of these trade-offs across embedding sources, not a single leaderboard number.

Results establish prediction of recorded clicks within displayed candidate lists. They do not establish user satisfaction or discovery of articles that were never shown. Discuss exposure bias, the privacy implications of reading histories, and whether recommendations concentrate on a narrow set of topics. Include representative failure cases and these limits in the final presentation.

## 5. Suggested work breakdown and timeline

### Total workload

The complete project consists of six work packages:

- Finalize the thesis, data rules, model comparisons, and evaluation split.
- Prepare the dataset, classical baselines, and shared evaluation pipeline.
- Implement embedding models, run experiments, and export predictions.
- Compare results, estimate uncertainty, and analyze failures.
- Examine ethical implications and state defensible conclusions.
- Prepare the final poster/presentation, rehearse, and submit.

Revised 19 October after the direction change. Workload shares stay K 30%, M 30%, Y 20%, and S 20% (K = Kevin, M = Marcus, Y = Yugan, S = Sood). K and M complete the experiment pipeline by 26 October. Y and S complete the analysis and delivery by 9 November for the 11 November presentation. The live version of this breakdown is `docs/project-stuff-to-do.md`.

### K — infrastructure, BPR-MF, and evaluation tooling — 30%

- Establish the chronological development split and the final-evaluation protocol with disclosure of the earlier validation smoke check.
- Implement BPR-MF and the shared BPR training loop that LightGCN reuses.
- Implement frozen semantic embeddings (single cached article encoding, averaged history profile, cosine scoring) and run an initial test with them.
- Implement the beyond-accuracy metrics on the prediction exports.
- Build the LLM-as-judge pipeline with a fixed rubric and calibration spot-checks.

Finish by 26 October. Supply M with the shared training loop and evaluation interface immediately.

### M — LightGCN and the two-tower extension — 30%

- Implement LightGCN on the user–article click graph, reusing the shared BPR training loop; normalized neighbor aggregation is the delta over BPR-MF.
- Implement the small two-tower extension after the core comparison is stable.
- Export results, predictions, configurations, and model-method notes.

Finish by 26 October. Prioritize LightGCN before the two-tower extension.

### Y — evaluation, testing, and statistical comparison — 20%

- Compare models and calculate confidence intervals from their predictions.
- Analyze history-length groups and other subgroups, and collect representative successes and failures.
- Own test coverage for the new components.
- Produce final charts and explain what the evidence supports.

The data pipeline, popularity and TF-IDF baselines, and evaluation exports Y already built count toward this share. Complete the analysis during weeks 3–5, by 9 November.

### S — final deliverable, ethics, and presenting — 20%

- Write the motivation, thesis, and related-work summary.
- Write the ethical analysis and limitations, grounded in the exposure and diversity metrics.
- Assemble the poster/presentation using K/M's method notes and Y's analysis.
- Edit the deliverable for consistent methods, figures, and claims.
- Coordinate submission and rehearsals, and present the overall findings.

Work throughout weeks 1–5. Have the deliverable and rehearsals ready by 9 November for the 11 November presentation.

Undecided: whether to redo the project critique deck for feedback on the new direction and whether to present at the optional Week 10–12 critique session. No owner assigned yet.

### Handoffs

Keep coordination to two main handoffs. Early in week 1, K supplies M with prepared data and the evaluation interface. By 26 October, K and M supply Y and S with working code, completed experiments, per-example predictions, and method notes. Y and S must be able to complete the analysis and presentation without further help from K or M.

## 6. Expected deliverables

- Reproducible data preparation, model code, and run instructions.
- Model predictions, metric summaries, confidence intervals, and comparison charts.
- Subgroup analyses, beyond-accuracy metrics, and LLM-as-judge results.
- Error analysis, ethical analysis, and conclusions tied to the tested core question.
- Final poster/presentation and a clear statement of each member's contribution.
