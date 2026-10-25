# Marcus model handoff

## Protocol

- Development only; supplied validation remained unopened.
- BPR pairs: each fitting click versus one displayed unclicked candidate from the same impression per epoch.
- LightGCN graph: distinct fitting history/click edges, symmetric degree normalization, two propagation layers, and a uniform mean over layer 0 through layer 2 in the fixed development configuration.
- Shared settings: dimension 64, 20 epochs, Adam 0.001, sampled layer-0 L2 0.0001, seed 4285.
- Scoring: inner product with explicit trained-evidence masking and history fallback for untrained users.

## Smoke evidence

- Command: `/usr/bin/time -l .venv/bin/python -m scripts.smoke_mind`
- Status: passed on the tiny real-MIND cohort.
- Device: CPU.
- BPR pair counts: 2 usable fitting impressions and 2 sampled pairs per epoch; 4 sampled optimization pairs across the 2-epoch smoke.
- LightGCN graph edges: 43 unique undirected user-article edges.
- Peak resident memory: 2,951,413,760 bytes (`/usr/bin/time -l` maximum resident set size).
- Elapsed time: 316.73 seconds.
- Smoke metrics are functionality checks and are not reportable project results.

## Remaining gate

The full development run and artifact audit passed at revision `352b0c9ed835345f8f2993bb28090c90f284e351`.

Two-tower gate: open. Core unit, lint, type, real-MIND smoke, full development, paired-output, beyond-accuracy, and byte-determinism checks all passed; validation remained unopened.

## Development run

- Revision: `352b0c9ed835345f8f2993bb28090c90f284e351`.
- Cohort: 125,572 training impressions; 5,000 evaluation users; 8,332 evaluation impressions.
- Runtime and peak resident memory: 103.96 seconds and 2,098,757,632 bytes with the frozen semantic cache already populated.
- BPR-MF: NDCG@5 0.19826178154576643; MRR 0.2291822606533819; final loss 0.6733455061912537; 45,848 trained users; 14,010 trained articles.
- LightGCN: NDCG@5 0.19745020413461783; MRR 0.22678512540867188; final loss 0.6861361265182495; 720,811 unique graph edges; 45,848 trained users; 38,086 trained articles.
- Pairwise differences and user-bootstrap confidence intervals: `outputs/dev/comparisons.json`.
- Determinism: byte-identical BPR-MF and LightGCN prediction exports on a second fit at the same revision.

## Interpretation boundaries

- These are development results, not the held-out final evaluation.
- Click labels measure prediction of recorded clicks among displayed candidates, not satisfaction or causal preference.
- Untrained collaborative candidates receive zero and must be included in failure and subgroup analysis.
- BPR-MF versus LightGCN isolates graph propagation only under the shared fixed optimizer and sampler.

## AI-use record for Sood

- Codex inspected the shared repository, proposed and repeatedly audited the design and implementation plans, and drafted model code, tests, and verification fixes during execution.
- Marcus approved the architecture and is responsible for reviewing the implementation and deciding which results and claims enter the assessed deliverable.
- Before submission, replace this operational record with a declaration that states the review, edits, and experiment execution Marcus actually completed; do not claim work that did not occur.

## Downstream files

- `outputs/dev/<model>/predictions.csv`: paired per-candidate scores and labels.
- `outputs/dev/<model>/impression_metrics.csv`: per-impression NDCG@5/MRR and history lengths.
- `outputs/dev/metrics.json`: aggregate metrics, fixed settings, and training metadata.
- `outputs/dev/comparisons.json`: all paired user-bootstrap comparisons.
- `outputs/dev/beyond_accuracy.json`: diversity, novelty, and candidate-catalogue coverage.
