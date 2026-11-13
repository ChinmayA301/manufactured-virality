"""Phase C (Twitter) - what does it cost to inject fake retweets the detector won't see?

SIMULATION ONLY, on the real Higgs cascades. Aggregated results go to results/c_twitter.json;
the attacker's optimised settings stay in $MV_DATA/phaseC_private/.

No real manipulation labels exist for Twitter, so unlike GitHub there is no realism gate
against real campaigns. The default campaign is Phase B4's crude 'dump that follows the
target' (fresh accounts, one-hour burst).

Scenarios
  boost_small   inject into a real cascade with 3-19 organic retweeters
  boost_large   inject into a real cascade with >= 20 organic retweeters
  spam          the target post has no organic retweeters; the campaign is the whole cascade
Roots are split three ways: defender training, threshold calibration, targets.

Attacker (theta in [0,1]^5), K fake retweets:
  mimic    share of fakes timed by the target cascade's own organic delays (else burst)
  window   burst window 10^(1.8 + 3.6 theta1) seconds (about 1 minute to 3 days)
  follow   share of fakes that follow the target (their retweets look follow-explained)
  aged     share of aged accounts, with follower/followee counts drawn from real adopters
           (else fresh: 0 followers)
  chain    share of fakes placed under an earlier fake (sockpuppets following each other),
           which gives fakes children and depth
Cost per fake (relative, an ASSUMPTION): fresh 1, aged 8.

Defenders (1% / 5% false-flag thresholds on untouched calibration cascades)
  supervised   gradient boosting on Phase B4's features, trained on B4's crude archetypes
               (dump, dump_follow, drip, mimic) injected into training cascades
  anomaly      unsupervised: sum of the three most extreme two-sided feature tails against
               untouched training cascades (no attack labels)
  adaptive     supervised, retrained each round on the attacker's last campaigns
"""
import os, sys, json, time
import numpy as np, pandas as pd
from scipy.stats import norm
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
import warnings; warnings.filterwarnings('ignore')

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'phaseB'))
from twitter_common import (roots, followers, followees, real_fol, real_fee, all_off, cascade, feats, FEATS, EXTERNAL, fake_id)

DATA = os.environ.get('MV_DATA', os.path.expanduser('~/projects/_data/manufactured-virality-data'))
PRIV = os.path.join(DATA, 'phaseC_private'); os.makedirs(PRIV, exist_ok=True)
RES = os.path.join(HERE, '..', 'results')
QUICK = os.environ.get('QUICK') == '1'
N_EVAL_T, N_FINAL_T, N_EVALS, N_INIT = (6, 40, 12, 4) if QUICK else (12, 150, 45, 8)
KS = [50] if QUICK else [10, 50]
SCEN = ['boost_small', 'boost_large', 'spam']
ROUNDS = 2 if QUICK else 3
COST = {'fresh': 1.0, 'aged': 8.0}
rng = np.random.default_rng(17)
t0 = time.time()

# ------------------------------------------------------------------ cascades and splits
C = {r: cascade(r, s, e) for r, s, e in roots}
rs = np.array([r for r, _, _ in roots]); rng.shuffle(rs)
third = len(rs) // 3
train_r, calib_r, target_r = rs[:third], rs[third:2 * third], rs[2 * third:]
size = {r: len(C[r][0]) for r in rs}
TGT = {'boost_small': [r for r in target_r if size[r] < 20], 'boost_large': [r for r in target_r if size[r] >= 20],
       'spam': list(target_r)}
print('cascades: train %d, calibration %d, targets %d (small %d, large %d) [%.0fs]'
      % (len(train_r), len(calib_r), len(target_r), len(TGT['boost_small']), len(TGT['boost_large']), time.time() - t0))
EMPTY = (np.array([], np.int64), np.array([]), np.array([], np.int64))


def fvec(d):
    return np.array([d[f] for f in FEATS])


def organic(r):
    return fvec(feats(r, *C[r]))


# ------------------------------------------------------------------ attacker
def campaign(theta, K, r, real_times, rr):
    t_first = real_times.min() if len(real_times) else 1.34e9
    n_mim = int(round(theta[0] * K))
    pool = (real_times - t_first) if len(real_times) >= 20 else all_off
    win = 10 ** (1.8 + 3.6 * theta[1])
    tm = np.concatenate([t_first + rr.choice(pool, n_mim) + rr.uniform(0, 600, n_mim),
                         t_first + rr.uniform(0, 3600) + rr.uniform(0, win, K - n_mim)])
    tm = np.sort(tm)
    n_aged = int(round(theta[3] * K)); aged = np.zeros(K, bool); aged[rr.choice(K, n_aged, replace=False)] = True
    fol = np.where(aged, rr.choice(real_fol, K), 0); fee = np.where(aged, rr.choice(real_fee, K), 0)
    par = np.full(K, EXTERNAL, np.int64)
    follow = rr.random(K) < theta[2]; par[follow] = r
    chain = (rr.random(K) < theta[4]) & (np.arange(K) > 0)
    for i in np.where(chain)[0]:                     # parent = a random earlier fake (sockpuppets follow each other)
        par[i] = fake_id(rr.integers(0, i)); fee[i] = max(fee[i], 1)
    fee[follow] = np.maximum(fee[follow], 1)
    cost = n_aged * COST['aged'] + (K - n_aged) * COST['fresh']
    return tm, par, fol, fee, cost


def injected(theta, K, r, scen, rr):
    nodes, times, par = C[r] if scen != 'spam' else EMPTY
    tm, fp, fol, fee, cost = campaign(theta, K, r, times if len(times) else C[r][1], rr)
    return fvec(feats(r, nodes, times, par, tm, fp, fol, fee)), cost


# B4's crude archetypes as settings of the same attacker (for training the supervised defender)
ARCH = {'dump': [0, 0.36, 0, 0, 0], 'dump_follow': [0, 0.36, 1, 0, 0], 'drip': [0, 1.0, 1, 0.5, 0], 'mimic': [1, 0.5, 1, 1, 0]}
DEFAULT = np.array(ARCH['dump_follow'])

# ------------------------------------------------------------------ defenders
Xo_tr = np.array([organic(r) for r in train_r]); Xo_cal = np.array([organic(r) for r in calib_r])
Xa_tr = []
for j, r in enumerate(train_r):
    a = list(ARCH.values())[j % 4]; K = [10, 50, 200][(j // 4) % 3]
    Xa_tr.append(injected(np.array(a, float), K, r, 'boost', rng)[0])
Xa_tr = np.array(Xa_tr)
print('defender training: %d organic, %d archetype campaigns [%.0fs]' % (len(Xo_tr), len(Xa_tr), time.time() - t0))


class Supervised:
    def __init__(self, fpr=0.01, extra_X=None):
        X = np.vstack([Xo_tr, Xa_tr] + ([extra_X] if extra_X is not None else []))
        y = np.r_[np.zeros(len(Xo_tr)), np.ones(len(Xa_tr) + (len(extra_X) if extra_X is not None else 0))]
        self.clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=0).fit(X, y)
        self.thr = np.quantile(self.score(Xo_cal), 1 - fpr)
    def score(self, X):
        return self.clf.predict_proba(np.atleast_2d(X))[:, 1]


class Anomaly:
    def __init__(self, fpr=0.01):
        self.ref = np.sort(Xo_tr, axis=0); self.thr = np.quantile(self.score(Xo_cal), 1 - fpr)
    def score(self, X):
        X = np.atleast_2d(X); n = len(self.ref)
        lo = np.array([np.searchsorted(self.ref[:, j], X[:, j], side='right') for j in range(X.shape[1])]).T / n
        hi = 1 - np.array([np.searchsorted(self.ref[:, j], X[:, j], side='left') for j in range(X.shape[1])]).T / n
        p = np.clip(2 * np.minimum(lo, hi), 1 / (2 * n), 1)
        return np.sort(-np.log(p), axis=1)[:, -3:].sum(axis=1)


def evaluate(theta, K, dfn, scen, n_t, rr, keep=None):
    det = 0; cost = 0.0
    for r in rr.choice(TGT[scen], n_t, replace=False):
        x, cost = injected(theta, K, r, scen, rr)
        det += dfn.score(x)[0] > dfn.thr
        if keep is not None: keep.append(x)
    return det / n_t, cost


def ei(Xc, gp, best):
    mu, sd = gp.predict(Xc, return_std=True); sd = np.maximum(sd, 1e-9); z = (mu - best) / sd
    return (mu - best) * norm.cdf(z) + sd * norm.pdf(z)


def optimise(method, K, x_t, dfn, scen, seed):
    rr = np.random.default_rng(seed); X, Y, P, Cc = [], [], [], []
    for i in range(N_EVALS):
        if method == 'random' or i < N_INIT:
            th = rr.random(5)
        else:
            gp = GaussianProcessRegressor(ConstantKernel() * Matern(nu=2.5) + WhiteKernel(), normalize_y=True, random_state=seed).fit(np.array(X), np.array(Y))
            cand = rr.random((2000, 5)); th = cand[np.argmax(ei(cand, gp, max(Y)))]
        p, c = evaluate(th, K, dfn, scen, N_EVAL_T, rr)
        X.append(th); Y.append(-(c / K + 50.0 * max(0.0, p - x_t))); P.append(p); Cc.append(c)
    # winner's-curse guard: re-check the 5 cheapest noisy-feasible candidates on 40 fresh targets
    ok = [i for i in range(len(X)) if P[i] <= x_t + 0.1]
    rc = np.random.default_rng(seed + 555); best = None; passed = False
    for i in sorted(ok, key=lambda i: Cc[i])[:5]:
        p40, _ = evaluate(np.array(X[i]), K, dfn, scen, 40, rc)
        if p40 <= x_t:
            best = i; passed = True; break
    if best is None:
        best = int(np.argmin(P))
    pf, cf = evaluate(X[best], K, dfn, scen, N_FINAL_T, np.random.default_rng(seed + 999))
    return dict(theta=X[best].tolist(), p_final=pf, cost_per_fake=cf / K, rechecked_feasible=passed)


out = dict(settings=dict(N_EVAL_T=N_EVAL_T, N_FINAL_T=N_FINAL_T, N_EVALS=N_EVALS, KS=KS, COST=COST, QUICK=QUICK,
                         n_train=len(train_r), n_calib=len(calib_r), targets={k: len(v) for k, v in TGT.items()}),
           default={}, designs={}, adaptive={})
private = {}
designs = {'supervised_fpr1': Supervised(0.01), 'supervised_fpr5': Supervised(0.05), 'anomaly_fpr1': Anomaly(0.01), 'anomaly_fpr5': Anomaly(0.05)}
for name, dfn in designs.items():
    flag = float(np.mean(dfn.score(Xo_cal) > dfn.thr))
    for sc in SCEN:
        dd = evaluate(DEFAULT, 50, dfn, sc, N_FINAL_T, np.random.default_rng(3))[0]
        out['default']['%s_%s' % (name, sc)] = dict(default_detect_K50=dd, calib_flag_rate=flag)
        for K in KS:
            bo = optimise('bo', K, 0.1, dfn, sc, seed=11 + K)
            rnd = optimise('random', K, 0.1, dfn, sc, seed=12 + K) if name in ('supervised_fpr1', 'anomaly_fpr1') else None
            key = '%s_%s_K%d' % (name, sc, K)
            out['designs'][key] = dict(bo={k: v for k, v in bo.items() if k != 'theta'},
                                       random=({k: v for k, v in rnd.items() if k != 'theta'} if rnd else None))
            private[key] = bo['theta']
            print('%-16s %-11s K=%2d | default dump detected %.2f | BO evasion cost/fake %.2f (P_det %.2f)%s [%.0fs]'
                  % (name, sc, K, dd, bo['cost_per_fake'], bo['p_final'],
                     ' | random %.2f (%.2f)' % (rnd['cost_per_fake'], rnd['p_final']) if rnd else '', time.time() - t0))

# adaptive supervised defender, boost_large, K=50 (the dilution case where crude detection was weakest)
extra = None; dfn = designs['supervised_fpr1']; rounds = []
for rd in range(ROUNDS):
    att = optimise('bo', 50, 0.1, dfn, 'boost_large', seed=500 + rd)
    keepx = []; evaluate(np.array(att['theta']), 50, dfn, 'boost_large', 150, np.random.default_rng(600 + rd), keepx)
    extra = np.array(keepx) if extra is None else np.vstack([extra, keepx])
    new = Supervised(0.01, extra)
    p_after = evaluate(np.array(att['theta']), 50, new, 'boost_large', N_FINAL_T, np.random.default_rng(700 + rd))[0]
    rounds.append(dict(round=rd, attacker_cost_per_fake=att['cost_per_fake'], p_det_vs_current=att['p_final'],
                       same_campaign_p_det_after_retrain=p_after))
    private['adaptive_round%d' % rd] = att['theta']
    print('adaptive round %d: attacker cost/fake %.2f at P_det %.2f; retrained defender catches it %.2f [%.0fs]'
          % (rd, att['cost_per_fake'], att['p_final'], p_after, time.time() - t0))
    dfn = new
out['adaptive'] = dict(scenario='boost_large', K=50, rounds=rounds)
json.dump(out, open(os.path.join(RES, 'c_twitter%s.json' % ('_quick' if QUICK else '')), 'w'), indent=1)
json.dump(private, open(os.path.join(PRIV, 'c_twitter_thetas%s.json' % ('_quick' if QUICK else '')), 'w'), indent=1)
print('total %.0fs' % (time.time() - t0))
