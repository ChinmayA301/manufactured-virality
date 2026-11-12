# manufactured-virality

Research code for three questions. What does organic virality look like on GitHub and Twitter? How much of it is controllable? What does it cost a manipulator to get past a detector?

See [DESIGN.md](DESIGN.md) for the questions, phases, label-circularity protocol and release policy.

- Forked from the virality-BO study ([blog-artifacts/virality-bayesian-optimization](https://github.com/ChinmayA301/blog-artifacts/tree/main/virality-bayesian-optimization)).
- Builds on SignalGraph's scoring package ([`signalgraph/packages/scoring`](https://github.com/ChinmayA301/Signal-Graph)).

## Setup

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python numpy scipy pandas pyarrow pymongo remotezip scikit-learn matplotlib networkx igraph
uv pip install --python .venv/bin/python -e ../signalgraph/packages/scoring
.venv/bin/python data/extract_starscout.py    # streams StarScout's Mongo dump into Parquet (no local copy)
```

Data lives outside the repo, by default in Google Drive under `manufactured-virality-data/`. Set `MV_DATA` to override.

## Data notice

StarScout labels mark *suspected* fake stars (He et al., ICSE 2026). Individual repositories and accounts may be false positives. This project uses them for statistics only and never names anyone.
