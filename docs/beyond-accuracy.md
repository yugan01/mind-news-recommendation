# Beyond-accuracy metrics

The implementation in `src/beyond.py` reads existing predictions and cached semantic vectors. It does not fit models, load an encoder, or generate new rankings. All enabled models must match the current evaluation impressions, users, candidate order, and labels exactly. Missing or outdated candidate embeddings are rejected rather than silently re-encoded.

Run after the full development experiment and its semantic cache exist:

```sh
python -m src.beyond
```

After Marcus runs the final experiment, post-process its predictions separately:

```sh
python -m src.beyond_final
```

The latter reads final-validation examples to check alignment, so it is reserved for the final evaluation stage. Neither command should be run now to generate full results. Outputs are `beyond_accuracy.json` in the corresponding development/final output directory. `scripts.smoke_mind` exercises this path on its isolated tiny sample.

## Definitions

Each impression uses its top five scored candidates, or all candidates if fewer than five exist. Ties use the shared stable candidate-order rule.

- Intra-list diversity: average `1 - cosine(a, b)` over all unordered pairs of distinct top-list articles. Vectors are the same EmbeddingGemma 2 representations for every recommender, normalized before the calculation. Singleton lists receive zero diversity. This is semantic distance, not category diversity; its values can range from 0 to 2.
- Novelty: mean self-information `-log2 p(article)` over the top list, in bits, following the popularity-based formulation discussed by Vargas & Castells (2011), *Rank and Relevance in Novelty and Diversity Metrics for Recommender Systems*. Here `p = (training_click_count + 1) / (total_training_clicks + number_of_articles_in_shared_metadata)`. Add-one smoothing gives finite novelty to articles without training clicks. Counts come only from clicked candidates in fitting impressions, not history occurrences or held-out labels.
- Catalogue coverage: number of distinct articles appearing in any top list divided by the distinct union of candidates in the evaluation impressions. This opportunity catalogue excludes metadata-only articles that no model was allowed to recommend. It is identical for every aligned model.

Diversity and novelty are averaged across impressions within each user, then across users, matching the accuracy aggregation. Coverage describes the union of recommendations across all evaluated impressions. These are descriptive summaries, not confidence intervals or satisfaction measurements. Embedding-based diversity reflects the chosen encoder's geometry and is not independent ground truth.

All three metrics are exercised by `tests/test_beyond.py`, discovered by `python -m unittest discover -s tests`, and by `python -m scripts.smoke_mind` on its isolated real-data sample. Development and final processing are separate commands, not automatic additions to the accuracy runner.

These definitions make the otherwise ambiguous smoothing and catalogue choices explicit for team review. The master project document remains the authority on overall scope.
