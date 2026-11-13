# Phase B results: the shared detector (12 Nov 2026)

Numbers come from `results/b_github.json`, `b_github_coadopt.json`, `b_github_profiles.json` and
`b_twitter.json`. They were produced by the scripts in `phaseB/`.

## B1. GitHub, timing only

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

## B2. GitHub: fair co-adoption from GH Archive

**The feature.** For every star, count how many *other* repositories the same account starred
that UTC day, across all of GitHub. The source is GH Archive, queried in the BigQuery free
sandbox (`phaseB/coadopt.sql`; 418 GB scanned, within the free monthly 1 TB). 99.7% of the
6.9M stars matched an archive event. The feature uses only public events available when the
stars happened, and it doesn't depend on StarScout's repo pool.

| Test | Timing: AUC / recall at 1% FPR | Co-adoption | **Timing + co-adoption** |
|---|---|---|---|
| Low-activity strict vs normal (2024 hold-out) | 0.959 / 0.74 | 0.963 / 0.71 | **0.993 / 0.89** |
| Low-activity loose vs normal | 0.936 / 0.55 | 0.914 / 0.59 | **0.977 / 0.75** |
| Spread-out campaigns only (5-fold) | 0.953 / 0.53 | 0.960 / 0.69 | **0.989 / 0.86** |
| Lockstep vs normal (circular, descriptive) | 0.999 / 0.97 | 0.994 / 0.88 | 0.999 / 0.98 |

| Median per repo | Normal | "Normal" dumps | Low-activity strict | Lockstep |
|---|---|---|---|---|
| Stargazers who starred ≥ 20 other repos the same day | 5% | 100% | 0% | 96% |

The co-adoption signal points both ways, which shows **two campaign styles**:
- **Throwaway accounts.** Low-activity campaign accounts star *nothing else* that day.
- **Farm accounts.** Lockstep accounts star dozens of repos that day, every one of them.

Normal stargazers sit in between. The "normal" dumps look like lockstep farms. Combined with
timing, recall on spread-out campaigns at 1% FPR rises from 53% to 86%.

## B3. GitHub: account profiles, with the coverage problem

Profiles exist only where StarScout fetched them. Median share of a repo's stargazers with a
fetched profile:
- low-activity strict: 86%;
- normal: 21%;
- lockstep: 1.5%;
- dumps: 1.4%.

So profile features describe a selected subset. They are scored cross-heuristically, over
repos with ≥ 10 fetched stargazers.

| Median, over fetched stargazers | Normal | Dumps | Low-activity strict | Lockstep |
|---|---|---|---|---|
| Account deleted since | 0% | 69% | **100%** | 14% |
| Low-activity profile (SignalGraph's rule) | 3% | 32% | — (all deleted) | 42% |
| Account created ≤ 14 days before the star | 0% | 0% | 0% | 0% |

- **Deletion is the strongest profile signal** (single-feature AUC 0.95 vs low-activity labels).
  But it is **retrospective**: accounts are mostly deleted later by GitHub's moderation. A detector
  running when the stars arrive can't see it, so it measures enforcement after the fact, not
  live detectability.
- **SignalGraph's low-activity rule separates lockstep repos from normal** (AUC 0.90, a
  cross-heuristic test), on the 1,693 lockstep repos that have enough fetched profiles.
- **SignalGraph's "recent account" rule (created ≤ 14 days before starring) is inert here.**
  Campaign accounts in this data are older than two weeks when they star (single-feature AUC
  0.59–0.60).

## B4. Twitter: synthetic campaigns injected into real cascades

There are no real labels, so the test is built as follows:
- **Positives** are real Higgs cascades (≥ 3 retweeters; 11,348 roots) with fake retweeters
  injected. **Negatives** are the same cascades untouched.
- **Split:** by root account, so no cascade appears in both train and test.
- **Size excluded:** repo size would leak the label in this paired design, so it isn't a feature.
- **A built-in limit:** fake accounts are new nodes that nobody real follows, so they can never
  trigger organic downstream spread.

| Feature family | AUC | Recall at 1% FPR |
|---|---|---|
| Timing | 0.949 | 0.55 |
| Structure (follow share, broadcast share, depth, structural virality, childless first-hop share) | 0.978 | 0.63 |
| Accounts (followers, zero-follower and zero-followee shares) | 0.946 | 0.49 |
| All | **0.994** | **0.89** |

| Fakes as a share of all adopters | ≤ 20% | 20–50% | 50–80% | > 80% |
|---|---|---|---|---|
| Recall at 1% FPR (all features) | **0.29** | 0.53 | 0.88 | 1.00 |

- **Unoptimised campaigns are easy to catch whenever they dominate a cascade**, and most Higgs
  cascades are tiny, so they usually do. The structural tell: fakes never have children.
  99.5% of first-hop adopters are childless with fakes present, against 86% organically.
- **A small campaign hidden inside a large organic cascade is the hard case** (29% recall below
  20% fake share).
- **Detectors trained only on crude campaigns don't transfer to mimicking ones.** Trained on
  dump/drip and tested on timing-mimic campaigns: AUC 0.94, but **recall at 1% FPR only 7%**.
  This is the gap an optimising attacker will exploit. It is Phase C's starting point.

## B5. Fed back into SignalGraph

Merged to SignalGraph main on 13 Nov 2026 (Signal-Graph PR #1):
- **`arrival_regularity_score`** (share of sub-minute gaps), weight 0.15.
- **`co_adoption_farm_score`** (share of stargazers starring 20+ other repos that day), weight
  0.10. It reads a new optional `StarEventInput.same_day_other_stars` field and is inactive
  until that field is ingested.
- **New "reasons" text,** in the same "confidence estimate, not proof" register.
- **Tests:** the scoring tests (5) and the API smoke tests (2) pass.
- **Score shift:** `_weighted_sum` renormalises, so existing manipulation scores shift slightly.

### B6. Measuring co-adoption live, through GitHub GraphQL (13 Nov 2026)

SignalGraph now fills `same_day_other_stars` itself
(`apps/api/app/services/coadoption_ingest.py`). For a sample of stargazers, it reads each
account's star list newest-first and counts the other repos starred on the same UTC day. A
per-analysis GH Archive scan would not fit the BigQuery free tier.

**Validation** (`phaseB/validate_graphql_coadopt.py`; aggregates in
`results/b_graphql_validation*.json`). The SignalGraph code was run on the same (repo, account,
day) stars that the GH Archive measure used: 15 normal and 15 lockstep (CL) repos, 30 stars
each, with production paging limits.

| | Normal repos | Lockstep repos that still exist | Lockstep repos, any |
|---|---|---|---|
| Repos still on GitHub | 15 / 15 | 15 / 15 (16 of 260 checked survive) | 1 / 15 |
| Stars resolved | 52–59% | 52% | 51% |
| Signal active (≥ 8 resolved) | 15 / 15 | 10 / 15 | 10 / 15 |
| Rank correlation with GH Archive (active repos) | 0.74 (pooled) | | 0.20 (pooled) |
| Share > 0.3 by GH Archive / by GraphQL | 0% / 0% | 27% / 30% | 40% / 0% |

- **Where the evidence survives, GraphQL agrees with GH Archive.** The three farm-heavy
  surviving repos read 0.83, 0.85 and 0.86 in GH Archive and 0.90, 0.95 and 0.85 via GraphQL.
  Normal repos stay at or below 0.10 on both.
- **Once GitHub cleans up, the evidence is gone.** 94% of the lockstep repos checked have been
  deleted. On those, GraphQL reads 0.00 where GH Archive read 0.54–1.00: the other
  repos the farm accounts starred have been removed too. The signal can only catch campaigns
  before cleanup.
- **Cost:** about 2 s of GitHub time per account for years-old stars. Three queries in flight
  triggered GitHub's secondary rate limit (403), so production sends one query at a time, with
  a 30 s budget.

**A bigger constraint found along the way.** Since 30 June 2026, GitHub shows a repository's
stargazer list only to its admins and collaborators: the GraphQL `stargazers` connection returns
0 and REST `/stargazers` returns 404 for everyone else (checked 13 Nov 2026 with two tokens).
GH Archive's WatchEvent capture also collapsed: about 200k per day in June 2023 and 120k in June
2025, against 0.7k–8k per day since July 2026. So per-stargazer signals (arrival timing,
co-adoption, account age) are now only available for repos you administer, or from the
historical archive. For other repos SignalGraph's live mode falls back to synthetic data and says so on the report
(`ingestion_source: mock_no_stargazer_access` plus a warning banner).

## Phase B summary

| | Finding |
|---|---|
| Current campaigns | Crude: single-day dumps from throwaway accounts or from farms. Timing plus co-adoption catches 89% of the headline set at 1% FPR. |
| Hardest real cases | Spread-out GitHub campaigns: 86% recall with both feature families. Small Twitter campaigns diluted in organic cascades: 29%. |
| Biggest data caveat | About 10% of StarScout's "normal" sample is probably missed campaigns; treating it as clean collapses recall at 1% FPR to 2–4%. |
| Retrospective vs live | Deletion is a strong but after-the-fact signal; leave it out of live detection claims. |
| The opening for Phase C | Detectors trained on crude attacks miss mimicking ones at usable thresholds (7% recall). |
