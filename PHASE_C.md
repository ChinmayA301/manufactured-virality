# Phase C results: what does evading the detector cost? (12 Nov 2026)

Part 1 covers GitHub, Part 2 covers Twitter. Both use a winner's-curse guard: before a
winner is chosen, the five cheapest candidates that looked feasible on 12 targets are
re-checked on 40 fresh targets. Every reported winner is then re-evaluated on 120–150 fresh
targets.

# Part 1: GitHub

**Simulation only.** No real accounts, repositories or campaigns were touched. This file and
`results/c_github.json` hold aggregated outcomes only. The attacker's optimised settings live
only in the local data directory (`$MV_DATA/phaseC_private/`) and are not published, per the
release policy in DESIGN.md. Script: `phaseC/github_game.py`; figure: `results/c_github.png`.

## Set-up

- **Targets.** 538 real normal repos from 2024 that the defender never trained on. Another 538
  set the defender's threshold.
- **Two scenarios:**
  - **Spam repo:** the campaign is the repo's whole star history. This is what StarScout's
    flagged repos mostly are.
  - **Boost:** the campaign is injected into a real repo's organic history. This is the case an
    investor using SignalGraph cares about.
- **Defender.** Phase B's timing plus co-adoption features. Co-adoption is now finer: the share
  of stargazers who starred 0, 1–4, 5+ and 20+ other repos that day, from a second GH Archive
  query (`phaseB/coadopt_v2.sql`). On normal repos, 44% of stargazers star nothing else that
  day, against 99% for throwaway campaigns and 0% for farm campaigns.
- **Attacker.** Four settings, each in [0, 1]:
  - duration, from 15 minutes to 90 days;
  - how much of the timing copies organic patterns;
  - share of aged accounts;
  - farm vs throwaway among the rest.
- **Prices** (relative, an assumption): throwaway 1, farm 0.6, aged 8. The cheapest possible
  campaign costs 0.6 per star.
- **Optimiser.** GP-EI BO: 45 evaluations, each over 12 random targets. The winner is
  re-evaluated on 120 fresh targets. Random search with the same budget is the control.
- **Goal.** Keep detection ≤ 10% (or ≤ 50%) at the lowest cost.

**Realism gate (passed).** The unoptimised default (a 15-minute dump, half farm, half
throwaway) matches real flagged campaigns on the detector's features:

| | Default campaign | Real campaigns |
|---|---|---|
| Best 1-day share | 1.00 | 1.00 |
| Share of gaps < 60 s | 1.00 | 0.99 |
| Stars in the 7 days after | 0.00 | 0.00 |

On spam repos it is detected **100%** of the time. Untouched targets are flagged 1.5% of the
time, against a 1% nominal threshold.

## Results

**1. The naive dump that real campaigns use is caught on spam repos, but not when it boosts a
real repo.** Injected into real repos, the same 200-star dump is detected 8–13% of the time.
Real flagged campaigns mostly sit on throwaway repos, so the detector learned what a fake *repo*
looks like, not what fake *stars on a real repo* look like.

**2. Against the static supervised detector, evasion is essentially free.** BO finds campaigns
detected ≤ 10% of the time at **0.60–0.90 per star**, against a cheapest-possible price of 0.60,
for every size (50, 200, 1,000 stars) in both scenarios. The winning campaigns use mostly farm
accounts, with timing spread over days to weeks. The supervised model only ever saw farm
accounts *together with* a burst, so it learned the conjunction, and a spread-out farm campaign
falls outside it. Constraining the model to be monotone doesn't help (0.60–0.69): a monotone
model can still be flat in farm share once timing looks organic.

**3. BO is the right tool for the attacker.** It found evasions at 0.60–0.90 per star where
random search with the same budget needed 0.75–1.80, and random search sometimes failed the
10% detection target on re-evaluation. This is the role the BO study predicted for sample-efficient
optimisation: an expensive, noisy black box with no exploitable structure.

**4. Only one defender design raises the price, and only by accepting more false flags.**

| Defender (boost scenario, K = 200) | Untouched real repos flagged | Naive dump caught | Cheapest evasion per star (3 BO runs; floor 0.60) |
|---|---|---|---|
| Supervised | 1.5% | 8% | 0.60, 0.60, 0.64 |
| Supervised, monotone | 1.5% | 16% | 0.60, 0.61, 0.64 |
| Anomaly, 1% target | 2.2% | 8% | 0.66, 0.69, 0.82 |
| Anomaly, 2% target | 4.3% | 26% | 0.60, 0.62, 0.70 |
| Anomaly, 5% target | 8.2% | 36% | 0.69, 0.76, 0.77 |
| **Anomaly, trimmed reference** | 7.8% | **68%** | **0.94, 1.10, 6.42** |

Spam scenario, trimmed reference: 1.22, 1.27, 4.67 per star; every other design 0.62–0.82.

- **The anomaly detector is unsupervised.** It flags repos whose co-adoption mix or minute-scale
  arrival pattern is extreme in either direction, relative to normal repos.
- **Raising its sensitivity alone barely moves the price.** StarScout's "normal" repos have a fat
  farm-like tail; about 1% have most stargazers starring 20+ others the same day. So any threshold
  set on them sits inside the campaign region.
- **Trimming the reference is what forces cost up.** The trimmed variant drops "suspect normals"
  (farm share > 50% or sub-minute share > 20%, a rule fixed in advance) before setting the
  threshold. That forces the attacker towards aged accounts, raising the median price about 2× and up to
  11× in one run, and catches 68% of naive boosts.
- **The price of trimming:** 7.8% of untouched real repos get flagged, against 1.5%. The spread
  across runs is wide; some BO runs still find a cheap gap.

**5. Retraining on the attacker's campaigns helps for one round at a time.** Over three rounds
(K = 200, detection ≤ 10%), each retrained defender catches 63–70% of the previous round's
campaign. The attacker re-optimises and evades again at 0.65, 2.39 and 0.82 per star. Three rounds
show no stable escalation in either direction.

## What it means

- **A detector trained on today's labels measures today's attackers.** StarScout-labelled
  campaigns are crude, so supervised detectors score very well (Phase B) and are evaded almost
  for free once an attacker optimises against them.
- **Making manipulation expensive means forcing the attacker to buy accounts that behave like
  real ones.** In this simulation, only a defender anchored to what normal looks like, rather than
  to past attacks, did that. It also needs a cleaned definition of normal, and it costs about 8%
  false flags. For a diligence tool like SignalGraph, an 8% "review this" rate may be acceptable
  where an accusation wouldn't be.
- **The signals that resist cheap evasion are account-level**: real activity, co-adoption that
  looks organic, account history. Timing alone is cheap to fake.

## Caveats

- **Prices are assumptions.** Cost per star is linear in them. The aged-account premium (8×)
  drives most of the trimmed defender's effect.
- **Simulation simplifications:**
  - Per-star co-adoption for real target repos is synthesised from their GH Archive aggregates.
    Reference, threshold and targets all go through the same synthesis.
  - Aged accounts copy organic co-adoption exactly; real bought accounts may be easier to spot.
- **What the defender doesn't see:** account history beyond that day's co-adoption, the social
  graph, profiles (which leak the label in this data), and GitHub's own after-the-fact
  deletions.
- **Small study:** a 4-dimensional attacker, 3 BO runs per design, 3 adaptive rounds. Ranges
  are wide.

# Part 2: Twitter

Script: `phaseC/twitter_game.py`; results: `results/c_twitter.json`. The cascade code is shared
with Phase B4 (`phaseB/twitter_common.py`).

## Set-up

- **Targets.** Real Higgs cascades, split by root account three ways:
  - defender training (3,782);
  - threshold calibration (3,782);
  - targets (3,784): 3,159 small cascades (3–19 organic retweeters) and 625 large (20+).
- **Three scenarios:**
  - **Boost small:** inject into a small real cascade.
  - **Boost large:** inject into a large real cascade.
  - **Spam:** the post has no organic retweeters; the campaign is the whole cascade.
- **No realism gate is possible.** There are no real Twitter manipulation labels. The default
  campaign is Phase B4's crude "dump that follows the target" (fresh accounts, one-hour burst).
- **Attacker** (five settings in [0, 1], K = 10 or 50 fake retweets):
  - timing mimicry;
  - burst window;
  - share of fakes following the target;
  - share of **aged** accounts (follower and followee counts drawn from real adopters);
  - share of fakes **chained** under other fakes (sockpuppets following each other, which gives
    fakes children and depth).
- **Prices** (an assumption): fresh 1, aged 8.
- **Defenders,** at 1% and 5% false-flag thresholds on calibration cascades:
  - supervised, trained on B4's archetypes injected into training cascades;
  - unsupervised anomaly detector.

## Results

Cost per fake for detection ≤ 10%. "≈" marks runs whose fresh re-evaluation came out at
11–20%; "✗" marks runs where no candidate passed the re-check.

| Defender | Scenario | Naive dump caught | K = 10 | K = 50 |
|---|---|---|---|---|
| Supervised, 1% | Boost small | 100% | 8.00 | 4.36 |
| | Boost large | 76% | 1.70 ≈ | 6.88 |
| | Spam | 100% | **1.00** | **1.28** |
| Supervised, 5% | Boost small | 100% | 8.00 | 6.74 ≈ |
| | Boost large | 91% | 8.00 ✗ | 8.00 ✗ |
| | Spam | 100% | 7.30 | 7.02 |
| Anomaly, 1% | Boost small | 84% | **1.00** | 4.64 ≈ |
| | Boost large | 6% | 3.10 ≈ | 6.46 ≈ |
| | Spam | 100% | 1.70 ≈ | 6.18 |
| Anomaly, 5% | Boost small | 100% | 8.00 | 7.44 |
| | Boost large | 81% | 8.00 ≈ | 7.86 ✗ |
| | Spam | 100% | 8.00 | 6.74 ✗ (85% detected) |

1. **On Twitter, evasion usually means buying aged accounts.** Against most defender and
   scenario pairs, the cheapest campaign that keeps detection near 10% is mostly or entirely
   aged accounts: 4–8× the cheapest price. At 5% false flags it is often impossible within
   this attacker's options. On GitHub the same goal cost about the cheapest price.
2. **The difference is what the detector can see.** The Twitter detector sees the follow
   graph (who is follow-explained, who has children, how deep cascades go) and account
   follower counts. That is account-level and structural evidence. The GitHub detector sees
   only timing and same-day co-adoption aggregates, which cheap accounts can match.
3. **Two cheap holes remain:**
   - **Fake-only cascades against the supervised detector** (1.00–1.28 per fake). It was
     trained only on injections into real cascades, so an all-fake cascade falls outside what
     it learned. This is the same failure as on GitHub, mirrored.
   - **Small injections into small cascades against the anomaly detector** (10 fakes at the
     cheapest price). They stay inside the organic envelope.
4. **BO again beats random search** in most cells: for example 4.36 vs 7.58 per fake, 1.00 vs
   6.60, and 1.00 vs 3.80.
5. **Adaptive rounds** (supervised, boost large, K = 50): the attacker pays 7.3–7.9 per fake
   each round, close to all aged accounts. Each retrained defender catches 35–82% of the
   previous round's campaign.

## The caveat that matters most

**Aged fakes here copy real adopters' follower counts, but nobody real follows them,** so they
can never trigger real retweets. A real aged or bought account with real followers could, and
the "fakes have no organic children" tell would weaken. **This simulation likely overstates
how hard Twitter evasion is.** A fair test needs a model of what bought accounts' real
audiences do.

# What Phase C shows across both domains

- **Detectors trained on observed campaigns are evaded almost for free** by an optimising
  attacker whenever their features can be matched with cheap accounts: timing on both
  domains, co-adoption aggregates on GitHub. They are also evaded by scenarios outside their
  training data: boosting real repos on GitHub, fake-only cascades on Twitter.
- **Cost rises only when the defender's evidence is account-level or structural:** the social
  graph on Twitter, and on GitHub co-adoption measured against a cleaned definition of
  normal. Then the attacker must buy accounts that behave like real ones (4–8× here).
- **The price is false flags** (about 8% on GitHub with a trimmed reference; the 5% thresholds
  on Twitter), plus the modelling caveats above. For a diligence tool like SignalGraph, that
  supports "review this" signals backed by account-level evidence over automated verdicts.
