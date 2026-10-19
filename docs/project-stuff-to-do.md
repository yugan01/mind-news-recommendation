## Project objective

The project aims to build a personalized news recommender using a user’s previously clicked articles.

For every recommendation event, the system receives:

1. The user’s article-click history.
2. A set of candidate articles shown to that user.
3. The title and abstract of each article.

It must rank the candidate articles so that the articles the user eventually clicked appear near the top.

The broader research objective is to determine whether understanding an article’s **meaning** produces better recommendations than matching its **keywords**.

## Central hypothesis

The hypothesis is:

> Semantic text embeddings will rank clicked news articles more accurately than TF-IDF when both methods use the same user history, candidate articles, and recommendation procedure.

This comparison must be controlled:

| Component | TF-IDF | Semantic embeddings |
|---|---|---|
| Article text | Title + abstract | Title + abstract |
| User history | Latest 20 articles | Latest 20 articles |
| User profile | Average article vectors | Average article vectors |
| Candidate score | Cosine similarity | Cosine similarity |
| Difference | Keyword representation | Semantic representation |

Keeping everything else identical means that any improvement should mainly come from the text representation.

The popularity baseline is only a basic reference point. The current result—TF-IDF beating popularity—does **not** test the central hypothesis yet because semantic embeddings have not been evaluated.

## Models in the project

There are four planned models:

1. **Popularity baseline** — completed  
   Ranks articles according to their training-data click counts. It is not personalized.

2. **TF-IDF baseline** — completed  
   Represents articles through important words and ranks candidates according to keyword similarity with the user’s history.

3. **Frozen semantic embeddings** — not completed  
   Uses a pretrained model such as `all-MiniLM-L6-v2` to represent article meaning. The pretrained encoder should initially remain frozen.

4. **Two-tower model** — not completed  
   Learns separate user and article vectors from recommendation feedback. This is an extension and should only be attempted after the main TF-IDF-versus-embeddings comparison works.

## Work completed so far

- Public GitHub repository created.
- MIND-small dataset downloaded and parsed.
- Popularity model implemented.
- TF-IDF model implemented.
- NDCG@5 and MRR evaluation implemented.
- User-level bootstrap confidence intervals implemented.
- Prediction exports implemented.
- Automated tests added.
- A fixed 5,000-user preliminary experiment completed.
- TF-IDF significantly outperformed popularity on that sample.

The implementation and results are in the [GitHub repository](https://github.com/yugan01/mind-news-recommendation).

## Work that still needs to be done

### 1. Establish the development and final-evaluation split

The current smoke experiment used the supplied validation data. Therefore, its results should be treated as **preliminary**, not as the untouched final evaluation.

From this point onward, we should:

- Create a chronological development/tuning split from the training data.
- Develop and debug the remaining models only on that split.
- Freeze model and preprocessing decisions.
- Run the supplied validation set once more as the final reported comparison.
- Clearly disclose that it was previously used for an initial pipeline check.

### 2. Implement frozen semantic embeddings

This is the most important next step because it tests the central hypothesis.

The implementation should:

- Combine each article’s title and abstract.
- Encode every article once using `all-MiniLM-L6-v2`.
- Cache the article vectors.
- Average the latest 20 history vectors to produce a user vector.
- Rank candidates using cosine similarity.
- Evaluate exactly the same users and impressions as TF-IDF.

### 3. Run the controlled comparison

Compare:

- Popularity
- TF-IDF
- Frozen semantic embeddings

For each model, report:

- NDCG@5
- MRR
- Mean paired difference against TF-IDF
- 95% user-level bootstrap confidence interval

The hypothesis is supported if the semantic model’s NDCG@5 improvement over TF-IDF is positive and its confidence interval supports that conclusion.

### 4. Test sensitivity to history length

For users with at least 20 historical clicks, rerun both text models using:

- Latest 5 articles
- Latest 10 articles
- Latest 20 articles

This tests whether longer histories improve recommendations or introduce unrelated older interests.

### 5. Analyze user groups

Compare users with:

- 10–19 history articles
- 20 or more history articles

This helps determine whether one representation works better for users with richer histories.

### 6. Perform failure analysis

Inspect examples where:

- TF-IDF succeeds but embeddings fail.
- Embeddings succeed but TF-IDF fails.
- Both fail.
- Popularity unexpectedly performs best.

Possible explanations include keyword overlap, semantic similarity, ambiguous interests, changing interests, and very popular articles.

### 7. Implement the two-tower extension

Only after the central comparison is complete:

- Build a user tower from reading history.
- Build an article tower from article features.
- Train them using impression click labels.
- Compare the trained model separately against the frozen methods.

Because it changes both the representation and the learning procedure, it should not be treated as part of the clean TF-IDF-versus-embeddings test.

### 8. Complete the ethical and limitation analysis

The report should discuss:

- Clicks do not necessarily represent satisfaction.
- Users can only click articles they were shown.
- Reading histories are privacy-sensitive.
- Recommendations may create narrow topic bubbles.
- Popularity methods may over-promote already popular content.
- The experiment ranks displayed candidates and cannot evaluate unseen articles.

### 9. Prepare the final deliverables

The remaining presentation work includes:

- Comparison charts
- Confidence-interval chart
- History-length analysis
- Representative success and failure examples
- Ethical discussion
- Defensible conclusion
- Poster or slide deck
- Clear contribution statement for each team member

## Responsibilities

Work breakdown from the proposal (`mind-small-project-summary.md`, section 5): K 30%, M 30%, Y 20%, S 20%. K = Kevin, M = Marcus, Y = Yugan, S = Sood.

- Kevin (K) — data, baselines, and evaluation; finish by 26 Oct. Largely preempted: Yugan already implemented the data pipeline, popularity/TF-IDF baselines, and evaluation, so the remaining K work is the dev/final split discipline and pipeline notes.
- Marcus (M) — embedding models and experiments (frozen embeddings, then two-tower), finish by 26 Oct. In chat Yugan suggested Kevin take the frozen-embedding step; confirm who owns it.
- Yugan (Y) — analysis: model comparison, confidence intervals, history-length groups, failure cases, ethics, conclusions, presenting; weeks 3–5, by 9 Nov.
- Sood (S) — final deliverable: motivation/thesis/related work, poster assembly, editing, submission coordination, presenting; ready by 9 Nov for the 11 Nov presentation.

Handoffs: K→M early week 1 (data format + evaluation interface); K and M → Y and S by 26 Oct (working code, experiments, per-example predictions, method notes).

Everyone uses `docs/` in this repo to stay on the same page about project direction.