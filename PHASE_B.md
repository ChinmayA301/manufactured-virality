# Phase B results so far: a timing-only detector on GitHub (12 Nov 2026)

Numbers come from `results/b_github.json`, produced by `phaseB/github_detector.py`.
The Twitter side of Phase B (synthetic injections into the damped cascade model) and the
caveated profile-feature analysis have not been run yet.

## Protocol

- **Features:** timing only, computed from each repo's own star times.
  - Account-profile features are left out because StarScout fetched profiles mostly for
    suspects (72% of low-activity-flagged stargazers have one, against 25% of normal-repo
    stargazers).
  - Co-adoption features against the label tables' repo pool are left out because suspected
    repos are over-represented in that pool.
- **Labels:**
  - **Low-activity strict (headline):** ≥ 20 stars flagged low-activity, and ≥ 50% of the repo's stars.
  - **Low-activity loose:** ≥ 20 flagged and ≥ 10%.
  - **Lockstep:** partly circular for timing features, since lockstep means many accounts
    starring within a short window.
- **Negatives:** StarScout's normal sample without the sub-day dumps.
- **Split:** by the date of the repo's busiest day; before 2024 = train, 2024 = test. The
  lockstep repos with full histories all fall in 2024, so they use 5-fold cross-validation.
- **Baseline:** SignalGraph's burst score, 0.45 × best-1-day share + 0.35 × best-7-day
  share + 0.20 × best-30-day share. SignalGraph's other manipulation components need data
  this table doesn't have (forks, contributors, profiles).

## Results

| Test | Test pos / neg | SignalGraph burst: AUC / recall at 1% FPR | Timing model: AUC / recall at 1% FPR |
|---|---|---|---|
| Low-activity strict vs normal | 284 / 1,111 | 0.950 / 0.73 | 0.958 / 0.74 |
| Low-activity loose vs normal | 556 / 1,111 | 0.809 / 0.49 | **0.930 / 0.55** |
| Low-activity strict, **spread-out campaigns only** (best day < 90% of stars), 5-fold | 173 / 8,935 | 0.891 / 0.28 | **0.945 / 0.45** |
| Low-activity strict vs normal, **dumps counted as negatives** | 284 / 1,525 | 0.779 / 0.02 | 0.782 / 0.04 |
| Lockstep vs normal (partly circular), 5-fold | 8,000 / 8,935 | 0.995 / 0.91 | 0.999 / 0.96 |
| Low-activity strict vs examined-but-clean repos | — | Not possible: no size-matched clean repos exist. Unmatched, repo size alone gives AUC 1.0. | |

Medians by class:

| | Normal | "Normal" dumps | Low-activity strict | Lockstep |
|---|---|---|---|---|
| Best 1-day share of stars | 0.04 | 1.00 | 1.00 | 1.00 |
| Share of gaps < 60 s | 0.00 | 0.99 | 0.84 | 0.99 |
| Stars in the 7 days after the burst, per burst star | 0.50 | 0.00 | 0.00 | 0.00 |
| Repos | 8,935 | 1,025 | 493 | 8,000 |

## What it means

1. **Today's flagged campaigns are crude.** The typical one is a single-day dump: every star
   within a day, most of them seconds apart, and nothing afterwards. Timing alone catches them
   (AUC 0.95), and SignalGraph's simple burst score already does nearly as well. That high AUC
   describes how naive current campaigns are, not how robust the detector is.
2. **The 1,025 "normal" dumps are almost certainly campaigns StarScout missed.** Their timing
   profile is identical to flagged campaigns. The detector flags 100% of them at a 1%
   false-positive rate on clean normals. Counting them as negatives collapses recall at 1% FPR
   to 2–4%. Any evaluation that treats StarScout's normal sample as clean understates
   detection, or misreports it.
3. **The timing model earns its keep where the burst score weakens.** On the looser label
   (AUC 0.93 vs 0.81) and on spread-out campaigns (0.945 vs 0.891), it gains through:
   - 30-day concentration;
   - irregular gaps;
   - sub-minute arrivals.
4. **Burst aftermath doesn't separate spread-out campaigns** (feature AUC 0.53). The Phase A
   idea that organic shocks have follow-on stars and campaigns don't holds for dumps but not
   for spread-out campaigns.
5. **The obvious evasion is to drip stars over weeks, with organic-looking gaps, using
   accounts that look active.** That defeats every feature above except profiles, which this
   data can't evaluate fairly. That is Phase C's question: what does such a campaign cost, and
   which features raise that cost?

## Still to do in Phase B

- **Twitter:** inject synthetic campaigns into the damped cascade model, and test structural
  features (follow-explained share, structural virality, depth) alongside timing. Only
  synthetic labels exist there.
- **Profiles:** analyse profile features restricted to stargazers whose profile was fetched,
  reported with the coverage asymmetry. Deleted accounts (18% of fetched profiles) form their
  own signal.
- **A fair co-adoption (lockstep) feature:** this needs complete star histories from GH
  Archive (BigQuery sandbox, column-pruned queries).
- **Feeding back into SignalGraph:** propose the timing features (30-day concentration, gap
  dispersion, sub-minute share) as a PR to `packages/scoring`, keeping its "suspicious, not
  proof" language.
