"""Phase C (GitHub) - what does it cost a star-campaign attacker to evade the detector?

SIMULATION ONLY. No real accounts, repositories or campaigns are touched. Results are
reported as aggregated cost curves; the attacker's optimised settings are written only to
the local data directory ($MV_DATA/phaseC_private/), never to the repo or results/.

Scenarios.  'spam': the campaign is the repo's whole star history (what StarScout's flagged
repos mostly are; the realism gate applies here). 'boost': the campaign is injected into a
real organic repo's history (the case an investor using SignalGraph cares about).

Targets.  Real normal repositories whose busiest day falls in 2024 (never used to train the
defender), split in half: CALIBRATION repos set the defender's 1%-false-positive threshold,
TARGET repos receive campaigns. Each target keeps its real star times; per-star co-adoption
values (other repos starred that day) are synthesised from the repo's real GH Archive
aggregates (shares 0, 1-4, >= 5, >= 20, mean), and calibration repos are treated the same way, so
the threshold and the targets see the same synthesis.

Defender.  Gradient boosting on timing + co-adoption features (Phase B), trained on real
StarScout positives (low-activity strict/loose, lockstep) vs pre-2024 normal repos.
A campaign is detected if the target repo's score exceeds the threshold.

Attacker (theta in [0,1]^4) delivering K fake stars:
  duration     10^(-2 + 3.95 theta0) days, about 15 minutes to 90 days
  mimic        share of stars timed by an organic pattern (inter-arrivals copied from a real
               normal repo and rescaled to the duration); the rest arrive in a tight burst
  aged         share of aged accounts with organic-looking co-adoption
  farm         among the non-aged: share of farm accounts (star 20-60 other repos that day);
               the rest are throwaways (star nothing else)
Cost per fake star (relative units, an ASSUMPTION; sensitivity reported): throwaway 1,
farm 0.6, aged 8.

Realism gate. The unoptimised default (15-minute dump, half farm, half throwaway) must look like
real flagged campaigns on the detector's features before optimisation results count.

Outputs: results/c_github.json (aggregates), results/c_github.png
"""
import os, json, time
import numpy as np, pandas as pd
from scipy.stats import norm
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
import warnings; warnings.filterwarnings('ignore')

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get('MV_DATA', os.path.expanduser('~/projects/_data/manufactured-virality-data')) + '/starscout'
PRIV = os.path.join(os.path.dirname(DATA), 'phaseC_private'); os.makedirs(PRIV, exist_ok=True)
RES = os.path.join(HERE, '..', 'results')
SPLIT = pd.Timestamp('2024-01-01').timestamp() / 86400
COST = {'throwaway': 1.0, 'farm': 0.6, 'aged': 8.0}
N_EVAL_T, N_FINAL_T, N_EVALS, N_INIT = 12, 120, 45, 8
KS = [50, 200, 1000]; XS = [0.5, 0.1]; ROUNDS = 3
QUICK = os.environ.get('QUICK') == '1'
if QUICK:
    N_EVAL_T, N_FINAL_T, N_EVALS, N_INIT, KS, ROUNDS = 6, 30, 12, 4, [200], 2
rng = np.random.default_rng(5)
t0 = time.time()

# ------------------------------------------------------------------ data
F = pd.read_parquet(os.path.join(DATA, 'b_github_features.parquet')).drop_duplicates('repo')
C = pd.read_csv(os.path.join(DATA, 'coadopt_by_repo_v2.csv'))   # phaseB/coadopt_v2.sql
C['log_mean_other'] = np.log1p(C.mean_other_same_day)
D = F.merge(C, on='repo', how='inner')
stars = pd.read_parquet(os.path.join(DATA, 'sample_stars.parquet')).dropna().drop_duplicates(['repo', 'actor'])
times_of = {r: np.sort(g.t.values.astype(float)) for r, g in stars.groupby('repo')}
normal = D[(D.label == 'normal') & D.repo.isin(times_of)]
held = normal[normal.burst_day >= SPLIT].sample(frac=1, random_state=1)
calib, targets = held.iloc[: len(held) // 2], held.iloc[len(held) // 2:]
print('normal repos held out (2024): %d calibration, %d targets [%.0fs]' % (len(calib), len(targets), time.time() - t0))


def timing_features(t):
    t = np.sort(t / 86400.0); n = len(t); f = {'log_n': np.log(n)}
    for d, k in ((1, 'top1d'), (7, 'top7d'), (30, 'top30d')):
        j = np.searchsorted(t, t + d, side='right'); f[k] = (j - np.arange(n)).max() / n
    j = np.searchsorted(t, t + 1, side='right'); b = int(np.argmax(j - np.arange(n))); b_end = t[b] + 1
    day = t[b:j[b]]
    jh = np.searchsorted(day, day + 1 / 24, side='right'); f['burst_hour_share'] = (jh - np.arange(len(day))).max() / len(day)
    jm = np.searchsorted(day, day + 10 / 1440, side='right'); f['burst_min_share'] = (jm - np.arange(len(day))).max() / len(day)
    ia_d = np.diff(day); f['burst_cv'] = ia_d.std() / ia_d.mean() if len(ia_d) > 1 and ia_d.mean() > 0 else 0.0
    f['after7_ratio'] = ((t >= b_end) & (t < b_end + 7)).sum() / len(day)
    f['before7_ratio'] = ((t < t[b]) & (t >= t[b] - 7)).sum() / len(day)
    ia = np.diff(t); f['cv'] = ia.std() / ia.mean() if ia.mean() > 0 else 0.0
    f['frac_lt60s'] = (ia < 60 / 86400).mean()
    span = max(t[-1] - t[0], 1.0); f['active_days_share'] = len(np.unique(np.floor(t))) / np.ceil(span)
    return f


def coadopt_features(c):
    return dict(log_mean_other=np.log1p(c.mean()), share_5plus_other_same_day=(c >= 5).mean(),
                share_20plus_other_same_day=(c >= 20).mean(), share_zero_other_same_day=(c == 0).mean(),
                share_1to4_other_same_day=((c >= 1) & (c <= 4)).mean())


TIMING = ['top1d', 'top7d', 'top30d', 'burst_hour_share', 'burst_min_share', 'burst_cv', 'after7_ratio',
          'before7_ratio', 'cv', 'frac_lt60s', 'log_n', 'active_days_share']
COAD = ['log_mean_other', 'share_5plus_other_same_day', 'share_20plus_other_same_day', 'share_zero_other_same_day',
        'share_1to4_other_same_day']


def synth_c(row, n, r):
    """Per-star 'other repos starred that day' from a repo's aggregates (4-bin mixture: 0, 1-4, 5-19, 20+)."""
    s5, s20, mean = row.share_5plus_other_same_day, row.share_20plus_other_same_day, row.mean_other_same_day
    s0 = row.share_zero_other_same_day
    u = r.random(n); c = np.empty(n)
    zero = u >= 1 - s0; lo = (u >= s5) & ~zero; mid = (u < s5) & (u >= s20); hi = u < s20
    c[zero] = 0; c[lo] = r.integers(1, 5, lo.sum()); c[mid] = r.integers(5, 20, mid.sum())
    rest = max(mean * n - c[lo].sum() - c[mid].sum(), 20.0 * hi.sum())
    if hi.sum():
        c[hi] = np.maximum(20, r.lognormal(np.log(rest / hi.sum()) - 0.5, 1.0, hi.sum()))
    return c


ORGANIC_C = np.concatenate([synth_c(row, 50, rng) for row in normal.sample(400, random_state=2).itertuples()])
ORGANIC_IA = [np.diff(times_of[r]) for r in normal.repo.values[:2000] if len(times_of[r]) > 30]


def campaign(theta, K, target_times, r):
    dur = 10 ** (-2 + 3.95 * theta[0]) * 86400
    if len(target_times) == 0:          # spam scenario: the campaign is the repo's whole history
        start = 1.7e9
    else:
        span = target_times[-1] - target_times[0]
        start = target_times[0] + r.uniform(0.1, 0.9) * span
    n_mimic = int(round(theta[1] * K))
    ia = ORGANIC_IA[r.integers(len(ORGANIC_IA))]
    ia = np.resize(ia, max(n_mimic, 1))[:n_mimic]
    mt = start + np.cumsum(ia) / max(ia.sum(), 1e-9) * dur if n_mimic else np.array([])
    bt = start + r.uniform(0, min(dur, 900), K - n_mimic)          # burst: within 15 minutes
    tm = np.concatenate([mt, bt])
    n_aged = int(round(theta[2] * K)); n_farm = int(round(theta[3] * (K - n_aged))); n_throw = K - n_aged - n_farm
    c = np.concatenate([r.choice(ORGANIC_C, n_aged), r.uniform(20, 60, n_farm), np.zeros(n_throw)])
    cost = n_aged * COST['aged'] + n_farm * COST['farm'] + n_throw * COST['throwaway']
    return tm, c, cost


def repo_features(t, c):
    f = timing_features(t) | coadopt_features(c)
    return {k: f[k] for k in TIMING + COAD}      # fixed column order


# ------------------------------------------------------------------ defender
def featurise_real(df):
    return df[TIMING + COAD].values
pos = D[D.label.isin(['LA_strict', 'LA_loose', 'CL'])]
neg_tr = normal[normal.burst_day < SPLIT]
X0 = np.vstack([featurise_real(pos), featurise_real(neg_tr)]); y0 = np.r_[np.ones(len(pos)), np.zeros(len(neg_tr))]
CAL_F = np.array([[v for v in repo_features(times_of[row.repo], synth_c(row, len(times_of[row.repo]), rng)).values()]
                  for row in calib.itertuples()])
TGT = [(times_of[row.repo], synth_c(row, len(times_of[row.repo]), rng)) for row in targets.itertuples()]
FEAT_ORDER = list(repo_features(TGT[0][0], TGT[0][1]).keys())
assert FEAT_ORDER == TIMING + COAD


class Defender:
    def __init__(self, cols, extra_X=None, extra_y=None):
        self.idx = [FEAT_ORDER.index(c) for c in cols]
        X, y = X0, y0
        if extra_X is not None:
            X, y = np.vstack([X0, extra_X]), np.r_[y0, extra_y]
        self.clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=0).fit(X[:, self.idx], y)
        self.thr = np.quantile(self.score(CAL_F), 0.99)              # 1% FPR on held-out real normals
    def score(self, X):
        return self.clf.predict_proba(np.atleast_2d(X)[:, self.idx])[:, 1]


EMPTY = (np.array([]), np.array([]))
SCEN = {'boost': None, 'spam': None}   # set below


def evaluate(theta, K, defender, n_targets, r, keep=None, scenario='boost'):
    det, cost = 0, 0.0
    for j in r.choice(len(TGT), n_targets, replace=False):
        tt, cc = TGT[j] if scenario == 'boost' else EMPTY
        ft, fc, cst = campaign(theta, K, tt, r)
        x = np.array(list(repo_features(np.concatenate([tt, ft]), np.concatenate([cc, fc])).values()))
        det += defender.score(x)[0] > defender.thr; cost = cst
        if keep is not None: keep.append(x)
    return det / n_targets, cost


def objective(p_det, cost, K, x_target):
    # minimise cost per star, heavy penalty above the allowed detection rate
    return -(cost / K + 50.0 * max(0.0, p_det - x_target))


def ei(Xc, gp, best):
    mu, sd = gp.predict(Xc, return_std=True); sd = np.maximum(sd, 1e-9); z = (mu - best) / sd
    return (mu - best) * norm.cdf(z) + sd * norm.pdf(z)


def optimise(method, K, x_target, defender, seed, scenario='boost'):
    r = np.random.default_rng(seed); X, Y, P, Cc = [], [], [], []
    for i in range(N_EVALS):
        if method == 'random' or i < N_INIT:
            th = r.random(4)
        else:
            gp = GaussianProcessRegressor(ConstantKernel() * Matern(nu=2.5) + WhiteKernel(), normalize_y=True,
                                          random_state=seed).fit(np.array(X), np.array(Y))
            cand = r.random((2000, 4)); th = cand[np.argmax(ei(cand, gp, max(Y)))]
        p, c = evaluate(th, K, defender, N_EVAL_T, r, scenario=scenario)
        X.append(th); Y.append(objective(p, c, K, x_target)); P.append(p); Cc.append(c)
    ok = [i for i in range(len(X)) if P[i] <= x_target]
    best = min(ok, key=lambda i: Cc[i]) if ok else int(np.argmin(P))
    pf, cf = evaluate(X[best], K, defender, N_FINAL_T, np.random.default_rng(seed + 999), scenario=scenario)   # fresh re-evaluation
    return dict(theta=X[best].tolist(), p_noisy=P[best], p_final=pf, cost_per_star=cf / K, feasible_noisy=bool(ok),
                curve=np.minimum.accumulate([Cc[i] / K if P[i] <= x_target else np.inf for i in range(len(X))]).tolist())


DEFAULT = np.array([0.0, 0.0, 0.0, 0.5])      # 15-minute dump, no mimicry, no aged accounts, half farm

# ------------------------------------------------------------------ realism gate
static = Defender(TIMING + COAD)
keep = []
p0, _ = evaluate(DEFAULT, 200, static, N_FINAL_T, np.random.default_rng(0), keep, scenario='spam')
syn = pd.DataFrame(keep, columns=FEAT_ORDER)
p0_boost, _ = evaluate(DEFAULT, 200, static, N_FINAL_T, np.random.default_rng(0), scenario='boost')
real_pos = D[D.label.isin(['LA_strict', 'CL'])]
gate = {f: dict(default_campaign=float(syn[f].median()), real_campaigns=float(real_pos[f].median()), real_normal=float(normal[f].median()))
        for f in ['top1d', 'frac_lt60s', 'burst_min_share', 'after7_ratio', 'share_20plus_other_same_day', 'log_mean_other']}
fpr_targets = float(np.mean([static.score(np.array(list(repo_features(tt, cc).values())))[0] > static.thr for tt, cc in TGT]))
print('realism gate (spam scenario): default dump K=200 detected %.2f | same dump injected into real repos (boost): %.2f | untouched targets flagged %.3f'
      % (p0, p0_boost, fpr_targets))
print(pd.DataFrame(gate).T.round(3))
out = dict(settings=dict(N_EVAL_T=N_EVAL_T, N_FINAL_T=N_FINAL_T, N_EVALS=N_EVALS, KS=KS, XS=XS, ROUNDS=ROUNDS, COST=COST, QUICK=QUICK,
                         n_calibration=len(calib), n_targets=len(targets)),
           realism_gate=dict(default_detect_rate_K200_spam=p0, default_detect_rate_K200_boost=p0_boost, untouched_target_flag_rate=fpr_targets, medians_spam_default=gate),
           static={}, adaptive={}, ablation={})
private = {}

# ------------------------------------------------------------------ static defender: BO vs random, cost curve
for sc in ('spam', 'boost'):
  for K in KS:
    for xt in XS:
        bo = optimise('bo', K, xt, static, seed=K + int(100 * xt), scenario=sc); rs = optimise('random', K, xt, static, seed=K + int(100 * xt) + 7, scenario=sc)
        out['static']['%s_K%d_x%.2f' % (sc, K, xt)] = {m: {k: v for k, v in d.items() if k != 'theta'} for m, d in (('bo', bo), ('random', rs))}
        private['static_%s_K%d_x%.2f' % (sc, K, xt)] = dict(bo=bo['theta'], random=rs['theta'])
        print('static %-5s K=%4d x<=%.2f | BO cost/star %.2f (P_det fresh %.2f) | random %.2f (%.2f) [%.0fs]'
              % (sc, K, xt, bo['cost_per_star'], bo['p_final'], rs['cost_per_star'], rs['p_final'], time.time() - t0))

# ------------------------------------------------------------------ adaptive defender rounds (K=200, x=0.1)
K, xt = 200, 0.1
defender = static; extra_X, extra_y = None, None; rounds = []
for rd in range(ROUNDS):
    att = optimise('bo', K, xt, defender, seed=700 + rd)
    keepx = []; evaluate(att['theta'], K, defender, 200, np.random.default_rng(800 + rd), keepx)
    # attacker's campaigns become positives; untouched synthesised targets are added as negatives so the
    # defender cannot key on synthesis artifacts
    neg_syn = np.array([list(repo_features(tt, cc).values()) for tt, cc in TGT[:200]])
    ex = np.vstack(keepx + [neg_syn]); ey = np.r_[np.ones(len(keepx)), np.zeros(len(neg_syn))]
    extra_X = ex if extra_X is None else np.vstack([extra_X, ex]); extra_y = ey if extra_y is None else np.r_[extra_y, ey]
    new_def = Defender(TIMING + COAD, extra_X, extra_y)
    p_after, _ = evaluate(att['theta'], K, new_def, N_FINAL_T, np.random.default_rng(900 + rd))
    rounds.append(dict(round=rd, attacker_cost_per_star=att['cost_per_star'], attacker_p_det_vs_current=att['p_final'],
                       same_campaign_p_det_after_retrain=p_after))
    private['adaptive_round%d' % rd] = att['theta']
    print('adaptive round %d: attacker cost/star %.2f at P_det %.2f; after defender retrains, same campaign detected %.2f [%.0fs]'
          % (rd, att['cost_per_star'], att['p_final'], p_after, time.time() - t0))
    defender = new_def
out['adaptive'] = dict(K=K, x=xt, rounds=rounds)

# ------------------------------------------------------------------ feature-family ablation (K=200, x=0.1)
for sc in ('spam', 'boost'):
    for fam, cols in (('timing_only', TIMING), ('coadoption_only', COAD), ('timing+coadoption', TIMING + COAD)):
        dfn = Defender(cols); rr = [optimise('bo', 200, 0.1, dfn, seed=1234 + s, scenario=sc) for s in range(1 if QUICK else 3)]
        out['ablation']['%s_%s' % (sc, fam)] = dict(cost_per_star=[x['cost_per_star'] for x in rr], p_final=[x['p_final'] for x in rr],
                                                    default_dump_detect=evaluate(DEFAULT, 200, dfn, N_FINAL_T, np.random.default_rng(3), scenario=sc)[0])
        print('ablation %-5s %-18s evasion cost/star %s (P_det %s); default dump detected %.2f'
              % (sc, fam, [round(x['cost_per_star'], 2) for x in rr], [round(x['p_final'], 2) for x in rr], out['ablation']['%s_%s' % (sc, fam)]['default_dump_detect']))

# ------------------------------------------------------------------ defender designs (K=200, x=0.1)
# The supervised model only saw farm accounts together with bursts, so the attacker evades it by
# spreading farm stars out. Two alternatives that do not depend on which conjunctions the labels
# happened to contain:
#   monotone   same GBM, constrained so more burst / sub-minute arrival / farm share can only
#              raise suspicion
#   anomaly    unsupervised: sum of the three most extreme two-sided empirical tails (-log p)
#              against real normal repos; flags "too many" and "too few" alike (farms, throwaways)
UP = ['top1d', 'top7d', 'top30d', 'burst_hour_share', 'burst_min_share', 'frac_lt60s', 'share_5plus_other_same_day',
      'share_20plus_other_same_day']   # share_zero is two-sided (organic ~0.44), so unconstrained


class MonotoneDefender(Defender):
    def __init__(self):
        self.idx = list(range(len(FEAT_ORDER)))
        cst = [1 if f in UP else 0 for f in FEAT_ORDER]
        self.clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=0,
                                                  monotonic_cst=cst).fit(X0, y0)
        self.thr = np.quantile(self.score(CAL_F), 0.99)


# features that do not depend on how much of the repo's history is observed (2024 repos are
# truncated at the data end, which makes size / active-day / 7-30-day / aftermath features look
# extreme for every recent repo and inflates any anomaly threshold)
ANOM = COAD + ['frac_lt60s', 'burst_min_share', 'burst_hour_share', 'burst_cv']
ANOM_IDX = [FEAT_ORDER.index(f) for f in ANOM]


class AnomalyDefender:
    """Unsupervised. Reference = pre-2024 normal repos built through the same per-star synthesis;
    score = sum of the three most extreme two-sided feature tails (history-length-robust features
    only). Threshold = (1 - fpr) quantile of the held-out 2024 calibration normals.
    trim=True drops 'suspect normals' (farm share > 0.5 or sub-minute share > 0.2, a rule fixed in
    advance) from both the reference and the calibration set; its realised flag rate on ALL
    untouched targets is then reported, since it will exceed the nominal fpr."""
    def __init__(self, fpr=0.01, trim=False):
        sub = neg_tr[neg_tr.repo.isin(times_of)].sample(min(3000, len(neg_tr)), random_state=4)
        r = np.random.default_rng(44)
        ref = np.array([list(repo_features(times_of[row.repo], synth_c(row, len(times_of[row.repo]), r)).values())
                        for row in sub.itertuples()])
        cal = CAL_F
        if trim:
            ref, cal = ref[~self.suspect(ref)], cal[~self.suspect(cal)]
        self.ref = np.sort(ref[:, ANOM_IDX], axis=0)
        self.thr = np.quantile(self.score(cal), 1 - fpr)
    @staticmethod
    def suspect(X):
        return (X[:, FEAT_ORDER.index('share_20plus_other_same_day')] > 0.5) | (X[:, FEAT_ORDER.index('frac_lt60s')] > 0.2)
    def score(self, X):
        X = np.atleast_2d(X)[:, ANOM_IDX]; n = len(self.ref)
        lo = np.array([np.searchsorted(self.ref[:, j], X[:, j], side='right') for j in range(X.shape[1])]).T / n
        hi = 1 - np.array([np.searchsorted(self.ref[:, j], X[:, j], side='left') for j in range(X.shape[1])]).T / n
        p = np.clip(2 * np.minimum(lo, hi), 1 / (2 * n), 1)
        return np.sort(-np.log(p), axis=1)[:, -3:].sum(axis=1)


designs = {'supervised': static, 'monotone': MonotoneDefender(), 'anomaly_fpr1': AnomalyDefender(0.01),
           'anomaly_fpr2': AnomalyDefender(0.02), 'anomaly_fpr5': AnomalyDefender(0.05), 'anomaly_trimmed': AnomalyDefender(0.01, trim=True)}
out['designs'] = {}
for name, dfn in designs.items():
    fpr_t = float(np.mean([dfn.score(np.array(list(repo_features(tt, cc).values())))[0] > dfn.thr for tt, cc in TGT]))
    for sc in ('spam', 'boost'):
        rr = [optimise('bo', 200, 0.1, dfn, seed=4321 + s, scenario=sc) for s in range(1 if QUICK else 3)]
        dd = evaluate(DEFAULT, 200, dfn, N_FINAL_T, np.random.default_rng(3), scenario=sc)[0]
        out['designs']['%s_%s' % (name, sc)] = dict(untouched_target_flag_rate=fpr_t, default_dump_detect=dd,
                                                    evasion_cost_per_star=[x['cost_per_star'] for x in rr],
                                                    p_final=[x['p_final'] for x in rr], feasible=[x['feasible_noisy'] for x in rr])
        private['design_%s_%s' % (name, sc)] = [x['theta'] for x in rr]
        print('design %-10s %-5s untouched flagged %.3f | default dump detected %.2f | evasion cost/star %s (P_det %s, feasible %s) [%.0fs]'
              % (name, sc, fpr_t, dd, [round(x['cost_per_star'], 2) for x in rr], [round(x['p_final'], 2) for x in rr],
                 [x['feasible_noisy'] for x in rr], time.time() - t0))

# cost-model sensitivity: re-price the attacker's chosen mixes with aged accounts at 4x and 16x
out['cost_sensitivity_note'] = 'cost per star is linear in prices; with aged at 4/16 instead of 8, costs scale by the aged share of each mix'
json.dump(out, open(os.path.join(RES, 'c_github%s.json' % ('_quick' if QUICK else '')), 'w'), indent=1)
json.dump(private, open(os.path.join(PRIV, 'c_github_thetas%s.json' % ('_quick' if QUICK else '')), 'w'), indent=1)
print('total %.0fs' % (time.time() - t0))
