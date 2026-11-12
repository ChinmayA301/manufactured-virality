# Phase C results: what does evading the detector cost? (12 Nov 2026)

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

Spam scenario, trimmed reference: 1.22, 2.72, 4.67 per star; every other design 0.62–0.82.

- **The anomaly detector is unsupervised.** It flags repos whose co-adoption mix or minute-scale
  arrival pattern is extreme in either direction, relative to normal repos.
- **Raising its sensitivity alone barely moves the price.** StarScout's "normal" repos have a fat
  farm-like tail; about 1% have most stargazers starring 20+ others the same day. So any threshold
  set on them sits inside the campaign region.
- **Trimming the reference is what forces cost up.** The trimmed variant drops "suspect normals"
  (farm share > 50% or sub-minute share > 20%, a rule fixed in advance) before setting the
  threshold. That forces the attacker towards aged accounts, raising the median price 2–5× and
  up to 11× in one run, and catches 68% of naive boosts.
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
- **GitHub only.** The Twitter game (attacker choosing fake count, follow behaviour and timing
  against the Phase B4 detector) has not been run.
