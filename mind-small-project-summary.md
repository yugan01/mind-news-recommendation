# News Recommendation with Text Embeddings

CP4285 group project proposal — 5 October 2026

## 1. Core idea and thesis

Recommend news articles using a user's previous article clicks and the text of candidate articles. The central question is whether semantic text embeddings produce better personalized rankings than a classical keyword representation.

**Thesis to test:** Semantic text embeddings rank clicked news articles more accurately than TF-IDF when both methods use the same reading history and recommendation procedure.

TF-IDF represents articles through their words. A pretrained neural text encoder represents their meaning as vectors. We will compare these representations directly, then test whether a small trained two-tower model improves on the simple embedding approach.

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

Use the supplied training data for development and model training. Reserve its final chronological portion for tuning. Keep the supplied validation data untouched for the final evaluation; it serves as the project's held-out evaluation set.

Start with a fixed random sample of 5,000 eligible users from the tuning data to check the pipeline. For the final comparison, evaluate all eligible users in the held-out set. Retain impressions containing at least one clicked and one non-clicked article.

For each impression, use up to the most recent 20 articles in its supplied history. Rank only the articles displayed in that impression. Use title plus abstract as article text for every text-based method. Evaluation click labels remain hidden until the ranking has been produced.

### Models

1. Popularity baseline: rank candidates by their click counts in the training data.
2. TF-IDF: fit the word representation on training article text. Average the history article vectors to form a user-interest vector, then rank candidates by cosine similarity to it.
3. Frozen semantic embeddings: encode article text with a pretrained model such as [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2). Use the same history averaging and cosine scoring as TF-IDF. Do not fine-tune the encoder in this comparison.
4. Small two-tower model: one network converts the history into a user vector; another converts article features into an article vector. Train on impression click labels so compatible user–article pairs receive higher scores. Keep this as the extension after the core comparison works.

The TF-IDF versus frozen-embedding comparison isolates the text representation. The two-tower comparison tests the additional benefit of learning from recommendation feedback.

### Example

A user has previously clicked articles about electric vehicles. A later impression shows 23 articles, including stories about batteries, football, and cooking. Each model ranks the same 23 articles without seeing their click labels. We then check whether the articles actually clicked appear near the top.

Precompute article vectors so the same article is encoded once and reused across users. Neither simple text method requires training a separate network for every user.

## 4. Evaluation and interpretation

Use NDCG@5 as the primary metric: it rewards placing clicked articles near the top five positions. Also report MRR, which rewards placing the first clicked article near the top.

Average scores across each user's impressions, then across users. Compare methods on identical examples. Estimate a 95% confidence interval for the paired score difference by resampling users.

Analyze users with 10–19 history articles separately from users with 20 or more. To test sensitivity to history length, compare the latest 5, 10, and 20 articles on the same users who have at least 20 available. Inspect examples where keyword matching succeeds and semantic embeddings fail, and vice versa.

The thesis receives support if semantic embeddings improve held-out NDCG@5 consistently, with the uncertainty interval supporting a positive difference. A tie or worse result is also a useful finding. The two-tower model is assessed separately because it changes how recommendations are learned.

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

Suggested workload shares are K 30%, M 30%, Y 20%, and S 20%. K and M complete the experiment pipeline and results within three weeks. Y and S own the analysis and final delivery, including presenting.

### K — data, baselines, and evaluation — 30%

- Prepare the data and establish training, tuning, and evaluation splits.
- Implement popularity and TF-IDF recommenders.
- Build the shared metric code and prediction-output format.
- Run baselines and export their results and predictions.
- Provide data-method notes and instructions for rerunning the pipeline.

Finish by 26 October. Supply M with the data format and evaluation interface early in week 1.

### M — embedding models and experiments — 30%

- Implement frozen article embeddings and averaged user profiles.
- Implement the small two-tower extension after the core comparison works.
- Run models through K's evaluation pipeline.
- Run the history-length comparisons.
- Export results, predictions, configurations, and model-method notes.

Finish by 26 October. Prioritize the completed TF-IDF versus frozen-embedding experiment before extending the models.

### Y — analysis, conclusions, and presenting — 20%

- Compare models and calculate confidence intervals from their predictions.
- Analyze history-length groups and representative successes and failures.
- Produce final charts and explain what the evidence supports.
- Write the ethical analysis and limitations.
- Present results and discussion with S.

Complete the analysis and presentation content during weeks 3–5, by 9 November.

### S — final deliverable and presenting — 20%

- Write the motivation, thesis, and related-work summary.
- Assemble the poster/presentation using K/M's method notes and Y's analysis.
- Edit the deliverable for consistent methods, figures, and claims.
- Coordinate submission and rehearsals.
- Present the problem, methods, and overall findings with Y.

Work throughout weeks 1–5. Have the deliverable and rehearsals ready by 9 November for the 11 November presentation.

### Handoffs

Keep coordination to two main handoffs. Early in week 1, K supplies M with prepared data and the evaluation interface. By 26 October, K and M supply Y and S with working code, completed experiments, per-example predictions, and method notes. Y and S must be able to complete the analysis and presentation without further help from K or M.

## 6. Expected deliverables

- Reproducible data preparation, model code, and run instructions.
- Model predictions, metric summaries, confidence intervals, and comparison charts.
- Error analysis, ethical analysis, and conclusions tied to the tested thesis.
- Final poster/presentation and a clear statement of each member's contribution.
