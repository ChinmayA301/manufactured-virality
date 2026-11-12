"""Phase A1 - does a calibrated cascade model reproduce real Higgs diffusion?

Real cascades. For every account that posted about Higgs (authored any action, or was
retweeted/replied to), its cascade is the set of distinct accounts that retweeted it.
Twitter attributes a retweet to the original author, so trees are inferred from the
follow graph (Goel et al. 2016): an adopter's parent is the most recent account it
follows that adopted earlier (the root counts as adopting first). Adopters with no
followee among earlier adopters are "external" (search, trends, ...); they and their
descendants are left out of the tree, and the external share is reported.

Simulated cascades. Each posting account seeds one independent cascade on the follower
graph; edge delays are resampled from the observed parent->child delays. The SAME parent
inference is applied to simulated cascades, so real and simulated trees are measured
through the same lens.

Models (p_uv for v following u):
  uniform      p = 0.0149                         (whole-week calibration)
  wc           c / max(followees(v), 20), c=0.726 (whole-week calibration, floor 20)
  wc_hetero    c_u / max(followees(v), 20), c_u per account from its direct follow-explained
               retweets (gamma-Poisson shrinkage). IN-SAMPLE for cascade size: c_u is fitted
               to the first layer of these very cascades. Judge it on depth, structural
               virality and timing, which it was not fitted to.

Statistics compared (per cascade, distributions):
  size (follow-explained adopters), P(size = 0), tail P(size >= 10, 100)
  depth, broadcast share (adopters at depth 1), structural virality (mean pairwise
  tree distance; cascades with >= 5 adopters), time to half of final size (>= 5 adopters)

Outputs: results/a1_twitter.json, results/a1_twitter.png
"""
import os, gzip, json, time
import numpy as np
import scipy.sparse as sp
from scipy.special import gammaln
from scipy.stats import ks_2samp

HERE = os.path.dirname(os.path.abspath(__file__))
HIGGS = os.environ.get('HIGGS_DIR', os.path.expanduser('~/projects/blog-artifacts/virality-bayesian-optimization/pipeline'))
CAL = os.path.join(HIGGS, '..', 'validation', 'calibration.json')
RES = os.path.join(HERE, '..', 'results'); os.makedirs(RES, exist_ok=True)
rng = np.random.default_rng(2026)
t0 = time.time()

cal = json.load(open(CAL)); F = cal['chosen']['floor']
C = cal['full']['wc_c_by_floor'][str(F)]; P_UNI = cal['full']['uniform_p']
z = np.load(os.path.join(HIGGS, 'follower_graph.npz'))
indptr, indices, ids = z['indptr'].astype(np.int64), z['indices'].astype(np.int64), z['ids']
n, m = len(ids), len(indices)
src = np.repeat(np.arange(n), np.diff(indptr))
followees_n = np.bincount(indices, minlength=n)
w_e = 1.0 / np.maximum(followees_n, F)[indices]
# reverse CSR: followees of v (for parent inference)
order = np.argsort(indices, kind='stable')
r_ptr = np.zeros(n + 1, np.int64); np.cumsum(followees_n, out=r_ptr[1:])
r_idx = src[order]

# ---------------------------------------------------------------- real cascades
pos = np.full(int(ids.max()) + 1, -1, np.int64); pos[ids] = np.arange(n)
raw = np.loadtxt(gzip.open(os.path.join(HIGGS, 'higgs-activity_time.txt.gz'), 'rt'), dtype=str)
a_, b_, ts, kind = raw[:, 0].astype(np.int64), raw[:, 1].astype(np.int64), raw[:, 2].astype(np.int64), raw[:, 3]
ok = (a_ <= ids.max()) & (b_ <= ids.max()); a_, b_, ts, kind = pos[a_[ok]], pos[b_[ok]], ts[ok], kind[ok]
ok = (a_ >= 0) & (b_ >= 0); a_, b_, ts, kind = a_[ok], b_[ok], ts[ok], kind[ok]
rt = kind == 'RT'
posted = np.zeros(n, bool); posted[a_] = True; posted[b_[rt | (kind == 'RE')]] = True
roots_all = np.where(posted)[0]
# first retweet time of each (retweeter, source) pair
R_a, R_b, R_t = a_[rt], b_[rt], ts[rt]
o = np.lexsort((R_t, R_a, R_b)); R_a, R_b, R_t = R_a[o], R_b[o], R_t[o]
first = np.ones(len(R_a), bool); first[1:] = (R_a[1:] != R_a[:-1]) | (R_b[1:] != R_b[:-1])
R_a, R_b, R_t = R_a[first], R_b[first], R_t[first]
starts = np.searchsorted(R_b, np.arange(n)); ends = np.searchsorted(R_b, np.arange(n), 'right')
print('posting accounts %d; retweeted sources %d; unique retweet pairs %d [%.0fs]'
      % (len(roots_all), int((ends > starts).sum()), len(R_a), time.time() - t0))

tadopt = np.full(n, np.inf)


def infer_tree(root, nodes, times):
    """Parent = most recent followee adopted before (root adopted first). Returns parents
    (-1 = external) aligned with nodes, which must exclude the root."""
    tadopt[root] = -np.inf; tadopt[nodes] = times
    par = np.full(len(nodes), -1, np.int64)
    for i, (v, tv) in enumerate(zip(nodes, times)):
        f = r_idx[r_ptr[v]:r_ptr[v + 1]]
        if f.size == 0:
            continue
        tf = tadopt[f]; ok = tf < tv
        if ok.any():
            fo, to = f[ok], tf[ok]; par[i] = fo[np.argmax(to)]   # choose among valid candidates only
    tadopt[root] = np.inf; tadopt[nodes] = np.inf
    return par


def tree_stats(root, nodes, times, par):
    """Stats over the follow-explained tree rooted at the source."""
    # keep adopters whose inferred chain reaches the root (drop externals and their descendants)
    o0 = np.argsort(times, kind='stable'); kept = {root}; keep = np.zeros(len(nodes), bool)
    for i in o0:
        if par[i] >= 0 and par[i] in kept:
            keep[i] = True; kept.add(nodes[i])
    nodes, times, par = nodes[keep], times[keep], par[keep]
    k = len(nodes)
    if k == 0:
        return dict(size=0, depth=0, d1=0, sv=np.nan, t_half=np.nan)
    idx = {root: 0}; idx.update({v: i + 1 for i, v in enumerate(nodes)})
    o = np.argsort(times, kind='stable')           # parents adopt before children
    depth = np.zeros(k + 1, np.int64); parent_i = np.zeros(k + 1, np.int64)
    for i in o:
        pi = idx[par[i]]; parent_i[i + 1] = pi; depth[i + 1] = depth[pi] + 1
    sub = np.ones(k + 1)                            # subtree sizes, children first
    for i in o[::-1]:
        sub[parent_i[i + 1]] += sub[i + 1]
    N = k + 1
    sv = (2.0 * (sub[1:] * (N - sub[1:])).sum() / (N * (N - 1))) if k >= 5 else np.nan
    tt = np.sort(times)
    t_half = (tt[(k - 1) // 2] - tt[0]) / 3600 if k >= 5 else np.nan
    return dict(size=k, depth=int(depth.max()), d1=int((depth == 1).sum()), sv=sv, t_half=t_half)


real, delays, n_external, n_adopt = [], [], 0, 0
for r in roots_all:
    s, e = starts[r], ends[r]
    if e == s:
        real.append(dict(size=0, depth=0, d1=0, sv=np.nan, t_half=np.nan)); continue
    nodes, times = R_a[s:e], R_t[s:e].astype(float)
    keep = nodes != r; nodes, times = nodes[keep], times[keep]
    par = infer_tree(r, nodes, times)
    n_external += int((par < 0).sum()); n_adopt += len(par)
    # parent->child delays between two observed adopters (root post time is unobserved)
    tmap = dict(zip(nodes.tolist(), times.tolist()))
    for v, tv, pv in zip(nodes, times, par):
        if pv >= 0 and pv != r:
            delays.append(tv - tmap[pv])
    real.append(tree_stats(r, nodes, times, par))
delays = np.array(delays)
print('real: %d adopters, %.1f%% external; %d parent->child delays, median %.1f h [%.0fs]'
      % (n_adopt, 100 * n_external / max(n_adopt, 1), len(delays), np.median(delays) / 3600, time.time() - t0))

# ---------------------------------------------------------------- models
# follow-explained direct adoptions per source (retweeter follows source), as in calibration
key_sorted = np.sort(src * n + indices)
kq = R_b * n + R_a; i = np.searchsorted(key_sorted, kq); i[i >= m] = 0
fol = key_sorted[i] == kq
A = np.bincount(R_b[fol], minlength=n).astype(float)
mass = np.bincount(src, weights=w_e, minlength=n)
act = np.where(posted & (mass > 0))[0]
def nb_ll(al):
    y, mu = A[act], C * mass[act]
    return (gammaln(y + al) - gammaln(al) + al * np.log(al / (al + mu)) + y * np.log(mu / (al + mu))).sum()
grid = np.exp(np.linspace(np.log(0.01), np.log(100), 81)); ALPHA = float(grid[np.argmax([nb_ll(x) for x in grid])])
c_het = np.full(n, C); c_het[act] = (A[act] + ALPHA) / (mass[act] + ALPHA / C)
MODELS = {'uniform': np.full(m, P_UNI), 'wc': np.minimum(1, C * w_e),
          'wc_hetero': np.minimum(1, c_het[src] * w_e)}


def simulate(ep, root):
    on_nodes = [root]; on = {root}
    frontier = np.array([root]); ftime = np.array([0.0])
    nodes, times = [], []
    while frontier.size:
        st, en = indptr[frontier], indptr[frontier + 1]; ln = en - st; tot = int(ln.sum())
        if tot == 0:
            break
        offs = np.repeat(st - (np.cumsum(ln) - ln), ln) + np.arange(tot)
        hit = rng.random(tot) < ep[offs]
        if not hit.any():
            break
        tg = indices[offs[hit]]; tpar = np.repeat(ftime, ln)[hit]
        perm = rng.permutation(len(tg)); tg, tpar = tg[perm], tpar[perm]   # random winner on ties
        tg_u, first_i = np.unique(tg, return_index=True)
        fresh = np.array([v not in on for v in tg_u.tolist()], bool)
        tg_u, tp = tg_u[fresh], tpar[first_i][fresh]
        tnew = tp + delays[rng.integers(0, len(delays), len(tg_u))]
        on.update(tg_u.tolist()); nodes.append(tg_u); times.append(tnew)
        frontier, ftime = tg_u, tnew
    if not nodes:
        return np.array([], np.int64), np.array([])
    return np.concatenate(nodes), np.concatenate(times)


sims = {}
for name, ep in MODELS.items():
    t1 = time.time(); out = []
    for r in roots_all:
        nodes, times = simulate(ep, r)
        if nodes.size == 0:
            out.append(dict(size=0, depth=0, d1=0, sv=np.nan, t_half=np.nan)); continue
        par = infer_tree(r, nodes, times)
        out.append(tree_stats(r, nodes, times, par))
    sims[name] = out
    print('simulated %s [%.0fs]' % (name, time.time() - t1))

# ---------------------------------------------------------------- compare
def arr(rows, k): return np.array([x[k] for x in rows], float)
def summary(rows):
    size, dep, d1, sv, th = (arr(rows, k) for k in ('size', 'depth', 'd1', 'sv', 't_half'))
    pos_ = size > 0
    return dict(n=len(rows), p_zero=float((size == 0).mean()), mean_size=float(size.mean()),
                p_ge10=float((size >= 10).mean()), p_ge100=float((size >= 100).mean()), max_size=int(size.max()),
                share_depth_ge2=float((dep[pos_] >= 2).mean()), broadcast_share=float(d1.sum() / max(size.sum(), 1)),
                sv_median=float(np.nanmedian(sv)), sv_p90=float(np.nanpercentile(sv, 90)),
                t_half_median_h=float(np.nanmedian(th)))
out = dict(models=dict(F=F, c=C, uniform_p=P_UNI, alpha=ALPHA), real=summary(real),
           real_external_share=n_external / max(n_adopt, 1), delays_median_h=float(np.median(delays) / 3600),
           sim={k: summary(v) for k, v in sims.items()}, ks={})
rs = arr(real, 'size'); rsv = arr(real, 'sv'); rth = arr(real, 't_half')
for k, v in sims.items():
    ss, ssv, sth = arr(v, 'size'), arr(v, 'sv'), arr(v, 't_half')
    out['ks'][k] = dict(size_pos=float(ks_2samp(rs[rs > 0], ss[ss > 0]).statistic),
                        sv=float(ks_2samp(rsv[~np.isnan(rsv)], ssv[~np.isnan(ssv)]).statistic),
                        t_half=float(ks_2samp(rth[~np.isnan(rth)], sth[~np.isnan(sth)]).statistic))
json.dump(out, open(os.path.join(RES, 'a1_twitter.json'), 'w'), indent=1)

import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
COL = {'real': '#222222', 'uniform': '#DD8452', 'wc': '#4C72B0', 'wc_hetero': '#C44E52'}
LAB = {'real': 'Real (Higgs)', 'uniform': 'Uniform IC', 'wc': 'Weighted cascade', 'wc_hetero': 'WC, per-account rates (size in-sample)'}
series = {'real': real, **sims}
fig, ax = plt.subplots(1, 4, figsize=(18, 4.6))
for k, rows in series.items():
    s = arr(rows, 'size'); s = s[s > 0]; xs = np.sort(s)
    ax[0].loglog(xs, 1 - np.arange(len(xs)) / len(xs), color=COL[k], lw=2, label=LAB[k])
    d = arr(rows, 'depth'); d = d[arr(rows, 'size') > 0]
    ax[1].plot(range(1, 8), [(d >= j).mean() for j in range(1, 8)], 'o-', color=COL[k], lw=2)
    v = arr(rows, 'sv'); v = np.sort(v[~np.isnan(v)])
    ax[2].plot(v, np.arange(len(v)) / len(v), color=COL[k], lw=2)
    t = arr(rows, 't_half'); t = np.sort(t[~np.isnan(t)])
    ax[3].semilogx(np.maximum(t, 1e-3), np.arange(len(t)) / len(t), color=COL[k], lw=2)
ax[0].set(xlabel='Cascade size (follow-explained adopters)', ylabel='P(size >= x | size > 0)', title='Size')
ax[1].set(xlabel='Depth j', ylabel='P(depth >= j | size > 0)', yscale='log', title='Depth')
ax[2].set(xlabel='Structural virality (>= 5 adopters)', ylabel='CDF', title='Broadcast (low) vs viral (high)')
ax[3].set(xlabel='Hours from first to median adoption (>= 5 adopters)', ylabel='CDF', title='Timing')
ax[0].legend(fontsize=8)
for a in ax: a.grid(alpha=.3)
fig.suptitle('Phase A1: calibrated cascade models vs real Higgs cascades (%d posting accounts, one simulated cascade each)' % len(roots_all),
             x=0.01, ha='left')
plt.tight_layout(); plt.savefig(os.path.join(RES, 'a1_twitter.png'), dpi=130, bbox_inches='tight')
print(json.dumps(out, indent=1)); print('total %.0fs' % (time.time() - t0))
