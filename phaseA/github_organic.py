"""Phase A2 - an organic model of GitHub star arrival, judged on what a detector looks at.

Data: StarScout's matched sample of normal repositories (sample_stars: 10,000 repos,
3.9M stars, Jul 2019 - Jan 2025). "Normal" means sampled for comparison and not flagged
by StarScout. That is not proof every star is organic; the share of stars in these repos
that StarScout's heuristics flag is reported as contamination.

No parent edges are observed for stars, so each repo's star times are modelled as a
point process on its own window [first star, data end]:
  poisson   constant rate (the null)
  hawkes    lambda(t) = mu + alpha * beta * sum exp(-beta (t - t_i)): self-exciting, where each
            star raises the chance of more (visibility, trending); alpha = branching ratio
  hawkes+shock  as hawkes, plus one exogenous burst of S immigrants at the repo's busiest day,
            decaying at rate gamma (an external post or trending listing; Crane & Sornette 2008)

Checks
  1. Per repo, in-sample: time-rescaling test. If the model is right, compensator
     increments are Exp(1); report the share of repos passing a KS test at 5%.
  2. Out of sample (the one that matters for a detector): fit on half the repos, draw new
     synthetic repos from the fitted parameter population, and compare the distributions of
     SignalGraph's burst features (best 1/7/30-day share of all stars) and inter-arrival
     dispersion against the held-out half.

Repos with fewer than 20 stars are skipped (too few to fit); the largest are capped to their
first 20,000 stars for fitting speed. Times in days.

Outputs: results/a2_github.json, results/a2_github.png
"""
import os, json, time
import numpy as np, pandas as pd
from numba import njit
from scipy.optimize import minimize
from scipy.stats import kstest, ks_2samp

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get('MV_DATA', os.path.expanduser('~/projects/_data/manufactured-virality-data')) + '/starscout'
RES = os.path.join(HERE, '..', 'results'); os.makedirs(RES, exist_ok=True)
MIN_STARS, CAP = 20, 20000
# EXCLUDE_DUMPS=1 drops repos whose stars mostly (>90%) arrive within one day. ~10% of
# StarScout's 'normal' sample got 50-964 stars in a median ~15 minutes, with stargazers
# far more often lockstep-flagged elsewhere: likely missed campaigns or artifacts.
EXCLUDE_DUMPS = os.environ.get('EXCLUDE_DUMPS') == '1'
SUF = '_nodumps' if EXCLUDE_DUMPS else ''
rng = np.random.default_rng(11)
t0 = time.time()

s = pd.read_parquet(os.path.join(DATA, 'sample_stars.parquet')).dropna(subset=['t'])
s = s.drop_duplicates(['repo', 'actor'])
T_END = s.t.max() / 86400.0 + 1.0
repos = []
for repo, g in s.groupby('repo', sort=False):
    t = np.sort(g.t.values.astype(float) / 86400.0)
    if len(t) >= MIN_STARS:
        if EXCLUDE_DUMPS:
            j = np.searchsorted(t, t + 1.0, side='right')
            if (j - np.arange(len(t))).max() / len(t) > 0.9:
                continue
        repos.append((repo, t))
print('%d repos with >= %d stars (of %d) [%.0fs]' % (len(repos), MIN_STARS, s.repo.nunique(), time.time() - t0))


@njit(cache=True)
def hawkes_nll(params, t, T0, T1):
    mu, alpha, beta = np.exp(params[0]), 1.0 / (1.0 + np.exp(-params[1])), np.exp(params[2])
    ll = 0.0; A = 0.0; prev = t[0]
    for i in range(t.size):
        if i > 0:
            A = np.exp(-beta * (t[i] - prev)) * (1.0 + A)
        prev = t[i]
        ll += np.log(mu + alpha * beta * A)
    comp = mu * (T1 - T0)
    for i in range(t.size):
        comp += alpha * (1.0 - np.exp(-beta * (T1 - t[i])))
    return -(ll - comp)


@njit(cache=True)
def shock_nll(params, t, T0, T1, tau):
    """Hawkes + one exogenous shock: immigrants mu + S*gamma*exp(-gamma (t - tau)) for t >= tau."""
    mu, alpha, beta = np.exp(params[0]), 1.0 / (1.0 + np.exp(-params[1])), np.exp(params[2])
    S, gam = np.exp(params[3]), np.exp(params[4])
    ll = 0.0; A = 0.0; prev = t[0]
    for i in range(t.size):
        if i > 0:
            A = np.exp(-beta * (t[i] - prev)) * (1.0 + A)
        prev = t[i]
        ex = S * gam * np.exp(-gam * (t[i] - tau)) if t[i] >= tau else 0.0
        ll += np.log(mu + ex + alpha * beta * A)
    comp = mu * (T1 - T0) + S * (1.0 - np.exp(-gam * (T1 - tau)))
    for i in range(t.size):
        comp += alpha * (1.0 - np.exp(-beta * (T1 - t[i])))
    return -(ll - comp)


@njit(cache=True)
def shock_compensator_increments(mu, alpha, beta, S, gam, t, T0, tau):
    out = np.empty(t.size); A = 0.0; prevL = 0.0
    for i in range(t.size):
        if i > 0:
            A = np.exp(-beta * (t[i] - t[i - 1])) * (1.0 + A)
        ex = S * (1.0 - np.exp(-gam * (t[i] - tau))) if t[i] >= tau else 0.0
        L = mu * (t[i] - T0) + ex + alpha * (i - A)
        out[i] = L - prevL; prevL = L
    return out


@njit(cache=True)
def hawkes_compensator_increments(mu, alpha, beta, t, T0):
    # Lambda(t_i) - Lambda(t_{i-1}); Lambda(t) = mu (t - T0) + alpha sum_{t_j < t} (1 - exp(-beta (t - t_j)))
    out = np.empty(t.size); A = 0.0; prevL = 0.0; S = 0.0
    for i in range(t.size):
        # sum_{j<i} exp(-beta (t_i - t_j)) via recursion
        if i > 0:
            A = np.exp(-beta * (t[i] - t[i - 1])) * (1.0 + A)
        L = mu * (t[i] - T0) + alpha * (i - A)
        out[i] = L - prevL; prevL = L
    return out


def fit(t):
    t = t[:CAP]; T0 = t[0] - 1e-6; T1 = T_END if len(t) < CAP else t[-1] + 1e-6
    t = t + rng.uniform(0, 1e-5, len(t)); t.sort()          # break exact ties
    n = len(t); base = np.log(n / (T1 - T0))
    best = None
    for a0, b0 in ((-1.0, 0.0), (1.0, 2.0), (2.5, -1.0)):
        r = minimize(hawkes_nll, np.array([base - 0.5, a0, b0]), args=(t, T0, T1), method='L-BFGS-B',
                     bounds=[(base - 15, base + 5), (-8, 8), (-6, 8)])
        if best is None or r.fun < best.fun:
            best = r
    mu, alpha, beta = np.exp(best.x[0]), 1 / (1 + np.exp(-best.x[1])), np.exp(best.x[2])
    inc_h = hawkes_compensator_increments(mu, alpha, beta, t, T0)
    inc_p = np.diff(np.concatenate([[T0], t])) * n / (T1 - T0)
    # shock at the start of the repo's busiest 1-day window
    j = np.searchsorted(t, t + 1.0, side='right'); tau = t[int(np.argmax(j - np.arange(n)))] - 1e-6
    bs = None
    for a0, b0, s0, g0 in ((best.x[1], best.x[2], np.log(0.2 * n), 1.0), (-1.0, 0.0, np.log(0.5 * n), 3.0), (0.5, 1.0, np.log(0.1 * n), 0.0)):
        r = minimize(shock_nll, np.array([best.x[0], a0, b0, s0, g0]), args=(t, T0, T1, tau), method='L-BFGS-B',
                     bounds=[(base - 15, base + 5), (-8, 8), (-6, 8), (np.log(0.5), np.log(2 * n)), (-6, 8)])
        if bs is None or r.fun < bs.fun:
            bs = r
    smu, sal, sbe, sS, sga = np.exp(bs.x[0]), 1 / (1 + np.exp(-bs.x[1])), np.exp(bs.x[2]), np.exp(bs.x[3]), np.exp(bs.x[4])
    inc_s = shock_compensator_increments(smu, sal, sbe, sS, sga, t, T0, tau)
    return dict(n=n, T=T1 - T0, mu=mu, alpha=alpha, beta=beta, nll=best.fun,
                p_hawkes=kstest(inc_h[1:], 'expon').pvalue, p_poisson=kstest(inc_p[1:], 'expon').pvalue,
                s_mu=smu, s_alpha=sal, s_beta=sbe, s_S=sS, s_gamma=sga, s_tau_rel=(tau - T0) / (T1 - T0), s_nll=bs.fun,
                p_shock=kstest(inc_s[1:], 'expon').pvalue)


def simulate_hawkes(mu, alpha, beta, T, S=0.0, gam=1.0, tau=0.0, nmax=200000):
    """Cluster (branching) representation on [0, T]: immigrants = Poisson baseline
    (+ exogenous shock immigrants at tau), each event has Poisson(alpha) children
    at Exp(beta) delays."""
    imm = [rng.uniform(0, T, rng.poisson(mu * T))]
    if S > 0:
        k = rng.poisson(S * (1 - np.exp(-gam * (T - tau))))
        d = rng.exponential(1 / gam, k); imm.append(tau + d[tau + d < T])
    gen = np.concatenate(imm); out = [gen]; total = gen.size
    while gen.size and total < nmax:
        kids = rng.poisson(min(alpha, 0.999), gen.size)
        ch = np.repeat(gen, kids) + rng.exponential(1 / beta, kids.sum())
        gen = ch[ch < T]; out.append(gen); total += gen.size
    return np.sort(np.concatenate(out))


def burst_features(t):
    """SignalGraph's burst concentration (best 1/7/30-day share) + inter-arrival CV."""
    t = np.sort(t); n = len(t)
    if n < 2:
        return dict(top1d=1.0, top7d=1.0, top30d=1.0, cv=0.0)
    f = {}
    for d, k in ((1, 'top1d'), (7, 'top7d'), (30, 'top30d')):
        j = np.searchsorted(t, t + d, side='right')
        f[k] = float((j - np.arange(n)).max() / n)
    ia = np.diff(t); f['cv'] = float(ia.std() / ia.mean()) if ia.mean() > 0 else 0.0
    return f


fits = []
for k, (repo, t) in enumerate(repos):
    fits.append(fit(t))
    if (k + 1) % 2000 == 0:
        print('  fitted %d/%d [%.0fs]' % (k + 1, len(repos), time.time() - t0))
F_ = pd.DataFrame(fits)
pass_h, pass_p, pass_s = (float((F_[c] > 0.05).mean()) for c in ('p_hawkes', 'p_poisson', 'p_shock'))
print('time-rescaling pass rate (5%%): Poisson %.3f, Hawkes %.3f, Hawkes+shock %.3f; branching ratio median %.2f (with shock %.2f)'
      % (pass_p, pass_h, pass_s, F_.alpha.median(), F_.s_alpha.median()))

# out-of-sample generative check: fit population on half, compare to the other half
idx = rng.permutation(len(repos)); half = len(repos) // 2
train, test = idx[:half], idx[half:]
real_test = pd.DataFrame([burst_features(repos[i][1][:CAP]) for i in test])
syn_h, syn_p, syn_s = [], [], []
for i in rng.choice(train, len(test), replace=True):     # resample whole parameter vectors (keeps correlations)
    p = fits[i]
    th = simulate_hawkes(p['mu'], p['alpha'], p['beta'], p['T'])
    if len(th) >= MIN_STARS:
        syn_h.append(burst_features(th))
    tp = np.sort(rng.uniform(0, p['T'], p['n']))
    syn_p.append(burst_features(tp))
    tsk = simulate_hawkes(p['s_mu'], p['s_alpha'], p['s_beta'], p['T'], p['s_S'], p['s_gamma'], p['s_tau_rel'] * p['T'])
    if len(tsk) >= MIN_STARS:
        syn_s.append(burst_features(tsk))
syn_h, syn_p, syn_s = pd.DataFrame(syn_h), pd.DataFrame(syn_p), pd.DataFrame(syn_s)
ks = {m: {f: float(ks_2samp(real_test[f], d[f]).statistic) for f in real_test} for m, d in (('hawkes', syn_h), ('hawkes_shock', syn_s), ('poisson', syn_p))}
def q(d): return {f: [float(x) for x in np.percentile(d[f], [10, 50, 90])] for f in d}
out = dict(exclude_dumps=EXCLUDE_DUMPS, n_repos=len(repos), min_stars=MIN_STARS, time_rescaling_pass=dict(poisson=pass_p, hawkes=pass_h, hawkes_shock=pass_s),
           params_median=dict(mu_per_day=float(F_.mu.median()), alpha=float(F_.alpha.median()), beta_per_day=float(F_.beta.median())),
           alpha_pct=[float(x) for x in np.percentile(F_.alpha, [10, 50, 90])],
           oos_ks=ks, quantiles=dict(real=q(real_test), hawkes=q(syn_h), hawkes_shock=q(syn_s), poisson=q(syn_p)),
           shock_params_median=dict(alpha=float(F_.s_alpha.median()), S_share=float((F_.s_S / F_.n).median()), gamma_per_day=float(F_.s_gamma.median())),
           n_synthetic=dict(hawkes=len(syn_h), hawkes_shock=len(syn_s), poisson=len(syn_p)))
json.dump(out, open(os.path.join(RES, 'a2_github%s.json' % SUF), 'w'), indent=1)

import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 4, figsize=(18, 4.4))
for j, f in enumerate(['top1d', 'top7d', 'top30d', 'cv']):
    for lab, d, col in (('Real (held-out normal repos)', real_test, '#222'), ('Hawkes (fitted population)', syn_h, '#4C72B0'),
                        ('Hawkes + exogenous shock', syn_s, '#C44E52'), ('Poisson', syn_p, '#DD8452')):
        x = np.sort(d[f].values); ax[j].plot(x, np.arange(len(x)) / len(x), color=col, lw=2, label=lab)
    ax[j].set(xlabel={'top1d': 'Best 1-day share of stars', 'top7d': 'Best 7-day share', 'top30d': 'Best 30-day share',
                      'cv': 'Inter-arrival CV'}[f], ylabel='CDF', title='KS: Hawkes %.2f, +shock %.2f, Poisson %.2f' % (ks['hawkes'][f], ks['hawkes_shock'][f], ks['poisson'][f]))
    if f != 'cv': ax[j].set_xscale('log')
    ax[j].grid(alpha=.3)
ax[0].legend(fontsize=8)
fig.suptitle('Phase A2: organic star-arrival models vs held-out normal repos, on SignalGraph\'s burst features', x=0.01, ha='left')
plt.tight_layout(); plt.savefig(os.path.join(RES, 'a2_github%s.png' % SUF), dpi=130, bbox_inches='tight')
print(json.dumps(out, indent=1)); print('total %.0fs' % (time.time() - t0))
