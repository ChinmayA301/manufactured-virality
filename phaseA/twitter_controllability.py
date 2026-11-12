"""Phase A3 (Twitter) - how much of virality is controllable?

Part 1, inside the calibrated simulator (law of total variance).
  For a single post: Var(size) = E[Var(size | account)]  (luck: same account, same model,
  different random draws) + Var(E[size | account])  (who posts). Estimated on a random
  sample of posting accounts with R draws each, under the homogeneous and the per-account
  weighted cascade. The "who posts" share is the most any account choice can control.
  For a campaign (k=20 seeds): luck = sd of reach across draws for a fixed seed set;
  control = the spread of expected reach across seeding strategies.

Part 2, on the real data (no simulator).
  Predict each account's real direct follow-explained retweets from what is known at launch
  (followers, followees, launch time) versus launch plus early signal (the first 5
  adopters' timing and audience). 5-fold cross-validated gradient boosting; Spearman and
  R^2 on log(1 + size). The launch-only ceiling bounds what pre-launch "engineering" can
  claim; the gap to early-signal prediction is the monitoring advantage (Cheng et al. 2014).

Outputs: results/a3_twitter.json
"""
import os, gzip, json, time
import numpy as np
from scipy.special import gammaln
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import cross_val_predict, KFold

HERE = os.path.dirname(os.path.abspath(__file__))
HIGGS = os.environ.get('HIGGS_DIR', os.path.expanduser('~/projects/blog-artifacts/virality-bayesian-optimization/pipeline'))
CAL = os.path.join(HIGGS, '..', 'validation', 'calibration.json')
RES = os.path.join(HERE, '..', 'results'); os.makedirs(RES, exist_ok=True)
N_ACC, R_DRAWS, K = int(os.environ.get('N_ACC', 3000)), int(os.environ.get('R_DRAWS', 30)), 20
rng = np.random.default_rng(7)
t0 = time.time()

cal = json.load(open(CAL)); F = cal['chosen']['floor']; C = cal['full']['wc_c_by_floor'][str(F)]
z = np.load(os.path.join(HIGGS, 'follower_graph.npz'))
indptr, indices, ids = z['indptr'].astype(np.int64), z['indices'].astype(np.int64), z['ids']
n, m = len(ids), len(indices)
src = np.repeat(np.arange(n), np.diff(indptr))
followers = np.diff(indptr); followees = np.bincount(indices, minlength=n)
w_e = 1.0 / np.maximum(followees, F)[indices]

pos = np.full(int(ids.max()) + 1, -1, np.int64); pos[ids] = np.arange(n)
raw = np.loadtxt(gzip.open(os.path.join(HIGGS, 'higgs-activity_time.txt.gz'), 'rt'), dtype=str)
a_, b_, ts, kind = raw[:, 0].astype(np.int64), raw[:, 1].astype(np.int64), raw[:, 2].astype(np.int64), raw[:, 3]
ok = (a_ <= ids.max()) & (b_ <= ids.max()); a_, b_, ts, kind = pos[a_[ok]], pos[b_[ok]], ts[ok], kind[ok]
ok = (a_ >= 0) & (b_ >= 0); a_, b_, ts, kind = a_[ok], b_[ok], ts[ok], kind[ok]
rt = kind == 'RT'
posted = np.zeros(n, bool); posted[a_] = True; posted[b_[rt | (kind == 'RE')]] = True
roots = np.where(posted)[0]

# follow-explained retweets (retweeter follows source), first per pair
key_sorted = np.sort(src * n + indices)
R_a, R_b, R_t = a_[rt], b_[rt], ts[rt]
kq = R_b * n + R_a; i = np.searchsorted(key_sorted, kq); i[i >= len(key_sorted)] = 0
fol = (key_sorted[i] == kq) & (R_a != R_b)
R_a, R_b, R_t = R_a[fol], R_b[fol], R_t[fol]
o = np.lexsort((R_t, R_a, R_b)); R_a, R_b, R_t = R_a[o], R_b[o], R_t[o]
first = np.ones(len(R_a), bool); first[1:] = (R_a[1:] != R_a[:-1]) | (R_b[1:] != R_b[:-1])
R_a, R_b, R_t = R_a[first], R_b[first], R_t[first]
A = np.bincount(R_b, minlength=n).astype(float)

mass = np.bincount(src, weights=w_e, minlength=n)
act = np.where(posted & (mass > 0))[0]
def nb_ll(al):
    y, mu = A[act], C * mass[act]
    return (gammaln(y + al) - gammaln(al) + al * np.log(al / (al + mu)) + y * np.log(mu / (al + mu))).sum()
grid = np.exp(np.linspace(np.log(0.01), np.log(100), 81)); ALPHA = float(grid[np.argmax([nb_ll(x) for x in grid])])
c_het = np.full(n, C); c_het[act] = (A[act] + ALPHA) / (mass[act] + ALPHA / C)
MODELS = {'wc': np.minimum(1, C * w_e), 'wc_hetero': np.minimum(1, c_het[src] * w_e)}


def size(ep, seeds):
    on = np.zeros(n, bool); seeds = np.unique(np.atleast_1d(seeds)); on[seeds] = True
    frontier = seeds; tot_n = 0
    while frontier.size:
        st, en = indptr[frontier], indptr[frontier + 1]; ln = en - st; tot = int(ln.sum())
        if tot == 0: break
        offs = np.repeat(st - (np.cumsum(ln) - ln), ln) + np.arange(tot)
        tg = indices[offs[rng.random(tot) < ep[offs]]]
        tg = np.unique(tg); tg = tg[~on[tg]]; on[tg] = True; tot_n += tg.size; frontier = tg
    return tot_n


out = dict(settings=dict(N_ACC=N_ACC, R_DRAWS=R_DRAWS, K=K, F=F, c=C, alpha=ALPHA), part1={}, part2={})

# ---------------- Part 1a: single posts, luck vs account
sample = rng.choice(roots, min(N_ACC, len(roots)), replace=False)
for name, ep in MODELS.items():
    t1 = time.time()
    S = np.array([[size(ep, r) for _ in range(R_DRAWS)] for r in sample], float)
    res = {}
    for lab, Y in (('size', S), ('log1p_size', np.log1p(S))):
        total = Y.var(); within = Y.var(axis=1, ddof=1).mean(); between = Y.mean(axis=1).var()
        res[lab] = dict(total_var=float(total), luck_share=float(within / (within + between)),
                        account_share=float(between / (within + between)))
    res['p_zero_mean'] = float((S == 0).mean())
    out['part1']['single_post_' + name] = res
    print('part1 single posts %s: account share of var(size) %.3f, of var(log1p size) %.3f [%.0fs]'
          % (name, res['size']['account_share'], res['log1p_size']['account_share'], time.time() - t1))

# ---------------- Part 1b: campaigns, luck vs strategy (k=20)
pool = np.argsort(-followers, kind='stable')[:3000]
for name, ep in MODELS.items():
    strategies = {'top_degree': pool[:K]}
    for j in range(5):
        strategies['random_top3000_%d' % j] = rng.choice(pool, K, replace=False)
    for j in range(3):
        strategies['random_top200_%d' % j] = rng.choice(pool[:200], K, replace=False)
    draws = {k: np.array([size(ep, s) for _ in range(R_DRAWS)], float) for k, s in strategies.items()}
    means = {k: float(v.mean()) for k, v in draws.items()}
    luck_sd = float(np.mean([v.std(ddof=1) for v in draws.values()]))
    out['part1']['campaign_' + name] = dict(means=means, luck_sd_mean=luck_sd,
                                            luck_sd_pct_of_top_degree=100 * luck_sd / means['top_degree'],
                                            strategy_range_pct_of_top_degree=100 * (max(means.values()) - min(means.values())) / means['top_degree'])
    print('part1 campaign %s: luck sd %.0f (%.1f%% of top-degree); strategy means %s'
          % (name, luck_sd, 100 * luck_sd / means['top_degree'], {k: round(v) for k, v in means.items()}))

# ---------------- Part 2: real-data predictability
y_all = A[roots]
starts = np.searchsorted(R_b, roots); ends = np.searchsorted(R_b, roots, 'right')
first_seen = np.full(n, np.inf); np.minimum.at(first_seen, a_, ts); np.minimum.at(first_seen, b_, ts)
t_launch = np.where(ends > starts, np.minimum(first_seen[roots], R_t[np.minimum(starts, len(R_t) - 1)]), first_seen[roots])
# Launch information only: audience, followees, and when the account first appears.
# (No activity counts: for accounts never retweeted they would include the whole week.)
T0 = ts.min()
X_launch = np.column_stack([np.log1p(followers[roots]), np.log1p(followees[roots]),
                            (t_launch - T0) / 3600.0, ((t_launch // 3600) % 24)])
cv = KFold(5, shuffle=True, random_state=0)
def evaluate(X, y, label):
    yl = np.log1p(y)
    pred = cross_val_predict(HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05), X, yl, cv=cv)
    r2 = 1 - ((yl - pred) ** 2).sum() / ((yl - yl.mean()) ** 2).sum()
    rho = spearmanr(yl, pred).statistic
    out['part2'][label] = dict(n=int(len(y)), r2_log=float(r2), spearman=float(rho))
    print('part2 %-40s n=%6d  R2(log)=%.3f  Spearman=%.3f' % (label, len(y), r2, rho))
evaluate(X_launch, y_all, 'all posting accounts, launch features')
evaluate(np.log1p(followers[roots])[:, None], y_all, 'all posting accounts, followers only')
# cascades that reached 5 adopters: launch vs launch + first-5 signal
big = np.where(ends - starts >= 5)[0]
early = []
for j in big:
    tt = R_t[starts[j]:ends[j]]; aa = R_a[starts[j]:ends[j]]
    o5 = np.argsort(tt)[:5]; t5 = tt[o5]
    early.append([(t5[-1] - t5[0]) / 3600.0, np.log1p(followers[aa[o5]]).mean(), np.log1p(followers[aa[o5]]).max(),
                  (t5[-1] - t_launch[j]) / 3600.0])
early = np.array(early)
evaluate(X_launch[big], y_all[big], '>=5 adopters, launch features')
evaluate(np.column_stack([X_launch[big], early]), y_all[big], '>=5 adopters, launch + first-5 signal')
evaluate(early, y_all[big], '>=5 adopters, first-5 signal only')

json.dump(out, open(os.path.join(RES, 'a3_twitter.json'), 'w'), indent=1)
print('total %.0fs' % (time.time() - t0))
