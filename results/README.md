# Baseline results

## Fixed 5,000-user smoke sample

Run date: 13 October 2026  
Dataset: RecZoo `MIND_small_x1` mirror  
Eligibility: history length >= 10 and impressions with both positive and negative candidates  
History used: latest 20 articles  
Random seed: 4285  
Evaluated users: 5,000  
Evaluated impressions: 7,946

| Model | NDCG@5 | MRR |
|---|---:|---:|
| Training-click popularity | 0.2301 | 0.2575 |
| TF-IDF title + abstract | 0.2929 | 0.3227 |

TF-IDF improved the user-averaged NDCG@5 by 0.0627 (about 27.3% relative)
and MRR by 0.0652 (about 25.3% relative) over popularity.

Paired user-level bootstrap results with 10,000 resamples:

| Metric | Mean paired difference | 95% CI |
|---|---:|---:|
| NDCG@5 | +0.0627 | [+0.0535, +0.0721] |
| MRR | +0.0652 | [+0.0559, +0.0745] |

Both intervals are entirely positive on this sample. This supports TF-IDF as a
meaningful baseline, but it does **not** yet address the project's core
question: the embedding-based models (frozen semantic embeddings, LightGCN,
two-tower) must be evaluated on exactly the same users and impressions. This
run also used the supplied validation split as a pipeline smoke check, so it is
preliminary; final results come from a single validation run after development
moves to a chronological split of the training data, with the prior use
disclosed.

Generated prediction files and raw data are excluded from Git because of their
size. Reproduce these figures with the commands in the project README.

