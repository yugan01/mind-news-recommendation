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

The full development run and artifact audit must pass before two-tower work starts.
