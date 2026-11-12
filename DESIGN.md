# Manufactured virality: an organic-diffusion model, a shared detector, and an attacker–defender loop

Status: design, 12 Nov 2026. Forked from the virality-BO study
(`blog-artifacts/virality-bayesian-optimization`), whose stage 6 mapped when sample-efficient
methods win at seeding. This project turns to the question the BO study was circling. What
does real, organic virality look like? How much of it can anyone control? And how much budget
does a manipulator need before a detector stops catching a manufactured cascade?

## Questions

1. **What is the right organic model?** Find a generative diffusion model, on each domain, that
   reproduces the statistics of real organic adoption: size distribution, depth and
   structural virality (broadcast vs viral), timing, and the profile of adopting accounts.
   Every later number depends on this, because a detector trained against a bad simulator
   learns the simulator's quirks.
2. **How much of virality is controllable?** Split outcome variance into what the controllable
   inputs explain (seeds, timing, budget) and what is left as luck. That gives a ceiling on
   any "engineered virality" claim.
3. **What does evasion cost?** An attacker manufactures adoptions under a budget, and a defender
   (a shared detector built on SignalGraph's features) tries to catch them. Measure how much
   budget is needed to keep detection below a target rate, before and after the defender
   adapts. The goal is to make that cost high.

## Two domains, one event schema

Both domains reduce to an **adoption event stream**: `(item, actor, t, actor features, parent?)`.

| | GitHub stars | Twitter retweets (Higgs) |
|---|---|---|
| Item | a repository | the Higgs announcement topic, split into per-source cascades |
| Adoption | a star (WatchEvent) | a retweet |
| Parent edge | none observed (stars carry no attribution) | retweeted account, plus the follow graph for exposure |
| Actor features | account age at the star, public repos, followers, total stars | followers and followees (crawl-truncated) |
| Organic data | StarScout's matched sample of normal repos (`sample_stars`) | the Higgs activity log, already calibrated in the BO study |
| Real manipulation labels | **yes**: StarScout's suspected fake stars, with each star labelled by each heuristic separately | **no**. The Twitter information-operations archive may have been discontinued (not confirmed). Only synthetic injections. |

## Data

- **StarScout replication package** (He et al., ICSE 2026; Zenodo 17009694; CC-BY-4.0).
  `data/extract_starscout.py` streams the 5.6 GB Mongo dump over HTTP range requests into
  Parquet on Google Drive (`manufactured-virality-data/starscout/`). Nothing is stored locally.
  The tables are `low_activity_stars`, `clustered_stars`, `sample_stars` and `actors`.
  - **Disclaimer carried over:** the labels mark *suspected* fakes. Use them for statistics,
    never to name individual repositories or accounts.
- **Higgs** (SNAP): the follower graph and activity log, as used in the BO study's `validation/`.
- **GH Archive** in BigQuery (free sandbox) is available if the StarScout sample proves too
  narrow. Query only the needed columns; a full scan costs terabytes.

## The label-circularity problem (design constraint)

StarScout labels come from two heuristics: **low-activity** (accounts with almost no other
activity) and **lockstep** (CopyCatch-style groups starring the same repos within a short
window). SignalGraph's `low_activity_stargazer_share` is nearly the low-activity rule.
Evaluating it against low-activity labels would grade a heuristic against itself. So:

- **Report every detector feature against each label source separately.**
- **Headline evaluation uses cross-heuristic labels.** Account-profile features are scored
  against lockstep labels, and timing/co-adoption features against low-activity labels.
- **Repos flagged by both heuristics** form the higher-confidence positive set. Repos flagged
  by neither, from the matched sample, are the negatives. Both are still noisy.

## Phases

### Phase A — organic virality model ("the correct setup")

**A1. Twitter.** Extend the BO study's calibrated weighted cascade (follower graph, floor 20,
per-account heterogeneity) with a **delay model** fitted to retweet inter-arrival times. Then
check it against per-source real cascades:
- cascade size distribution;
- depth, and **structural virality** (mean pairwise distance in the cascade tree; Goel et al. 2016);
- the broadcast share of reach;
- the time profile: time to half of final size, and burstiness.

Report predicted vs observed distributions, not single numbers.

**A2. GitHub.** No parent edges are observed, so model star arrival per repo as a
self-exciting point process: a Hawkes process with exogenous bursts (trending, posts).
Fit it on the normal-repo sample, then check:
- the distribution of 1/7/30-day burst concentration (the same quantities SignalGraph scores);
- inter-arrival dispersion;
- the stargazer account-profile mix;
- the growth-curve shape.

The organic model only has to make the detector's false-positive rate realistic, so it is
judged on exactly the features a detector looks at.

**A3. Controllability.**
- **Inside the simulator:** fix the controllable inputs and vary only the randomness. The
  run-to-run variance is luck. The variance across strategies is control.
- **On real data:** predict final cascade or star count from information available at launch
  (account size, history, timing) versus information from early growth. This reproduces the
  launch-vs-early-signal gap (Cheng et al. 2014) on both domains, and its ceiling bounds what
  pre-launch engineering can claim.

**Gate A:** the organic models reproduce the checked statistics within stated tolerances,
or the failures are documented before Phase B uses them.

### Phase B — shared detector

- **Features on the adoption stream:**
  - SignalGraph's: burst concentration, low-activity share, recent-account share.
  - Timing regularity: inter-arrival coefficient of variation, and periodicity.
  - Co-adoption lockstep: a CopyCatch-lite score over actor–item–time.
  - Profile-mix divergence from the item's organic cohort.
  - Twitter only: the follow-explained share, and structural virality.
- **Models:** SignalGraph's current weighted heuristic as the baseline, plus a calibrated
  gradient-boosted model.
- **Evaluation:**
  - GitHub: real labels under the cross-heuristic protocol above, split by time (train
    on 2019–2023, test on 2024).
  - Twitter: synthetic injections into the organic model only, reported as such.
- **Output:** a PR back to SignalGraph (`packages/scoring`) with the features that survive.
  It keeps SignalGraph's posture: "suspicious" and "elevated risk", never accusations.

### Phase C — attacker–defender loop (simulation only)

- **Attacker action space:**
  - number of fake accounts;
  - account profile mix (fresh, aged, or purchased with history);
  - timing schedule (burst, drip, or mimicking organic Hawkes parameters);
  - mixing with organic adoption;
  - Twitter only: which real accounts to target.
- **Cost model:** a per-account price by profile, plus per-action cost. Assumptions are
  stated, not measured.
- **Attacker realism check:** before the attacker optimises anything, its *unoptimised*
  campaigns must resemble the real StarScout campaigns' statistics. This is the same
  calibration discipline as Phase A.
- **Attacker optimiser:** BO. This is where it belongs: an expensive, noisy black box with
  no exploitable structure once a detector is in the loop.
- **Defender:** retrains or re-thresholds on the attacker's latest campaigns. Alternate
  the two sides and record the equilibrium path.
- **Headline result:** an **evasion-cost curve**, the budget needed to get k undetected
  adoptions at detection rate ≤ x, for the static and the adaptive defender. Also report
  which detector features push the attacker's cost up most.

## Ethics and release policy

- **The attacker exists only in simulation.** No real accounts or campaigns, and no
  interaction with GitHub or Twitter beyond reading public data.
- **Publish the detector, the evasion-cost curves and the feature-importance findings.**
  **Don't publish the attacker's optimised parameter sets or a ready-to-run campaign
  generator.** The attacker code is released only at the level needed to reproduce the
  cost curves, with best-found configurations aggregated.
- **Follow StarScout's disclaimer.** No repository or account is named as fake in any
  output; examples are aggregated or anonymised.

## Prior work to read before claiming novelty

These are checked where marked. Verify the rest before citing them.
- **Coordinated inauthentic behaviour and bot detection:** Botometer, and CopyCatch
  (Beutel et al., WWW 2013). Unchecked.
- **Fake GitHub stars:** He et al., ICSE 2026. Checked.
- **Unpredictability of social outcomes:**
  - MusicLab (Salganik, Dodds & Watts, Science 2006). Unchecked.
  - Cheng et al., WWW 2014. Unchecked.
  - Martin et al., WWW 2016. Unchecked.
- **Structural virality:** Goel et al., Management Science 2016. Unchecked.
- **Adversarially robust detection, and the cost of evasion for an optimising attacker:** the
  narrower and more open angle this project takes. Needs a literature pass.
