"""Phase B (Twitter) - can structure and timing reveal fake retweets injected into real cascades?

There are no real manipulation labels for Higgs, so positives are REAL cascades with
synthetic fake retweeters injected; negatives are the same population of real cascades
untouched. Train/test split is by root account, so no cascade appears in both.

Fake accounts are new nodes. Nobody real follows them, so they can never be the inferred
parent of a real adopter: fakes cannot trigger organic downstream spread. Whether the fakes
follow the target (so their retweets look follow-explained) is an attacker choice.

Unoptimised campaign archetypes (Phase C optimises against the detector):
  dump          k fakes within one hour; fresh accounts (0 followers, follow nobody)
  dump_follow   same, fakes follow the target
  drip          spread uniformly over 3 days; follow the target; follower counts drawn
                from the bottom half of real adopters
  mimic         timing copied from the cascade's own organic delays (or the global delay
                distribution); follow the target; follower counts drawn from real
                adopters (bought, aged accounts)
Injection size k in {10, 50, 200}.

Features per cascade (all adopters, fakes included):
  follow_share (adopters with an inferred follow parent), broadcast_share
  (depth-1 share), depth, structural virality, childless_d1 (share of depth-1 adopters
  with no inferred children), top1h_share, frac_lt60s, ia_cv, med_log_followers,
  zero_follower_share, zero_followee_share

Outputs: results/b_twitter.json
"""
import os, gzip, json, time
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, roc_curve

HERE = os.path.dirname(os.path.abspath(__file__))
HIGGS = os.environ.get('HIGGS_DIR', os.path.expanduser('~/projects/blog-artifacts/virality-bayesian-optimization/pipeline'))
RES = os.path.join(HERE, '..', 'results')
MIN_SIZE = 3
rng = np.random.default_rng(31)
t0 = time.time()

z = np.load(os.path.join(HIGGS, 'follower_graph.npz'))
indptr, indices, ids = z['indptr'].astype(np.int64), z['indices'].astype(np.int64), z['ids']
n = len(ids); src = np.repeat(np.arange(n), np.diff(indptr))
followers = np.diff(indptr); followees = np.bincount(indices, minlength=n)
order = np.argsort(indices, kind='stable'); r_ptr = np.zeros(n + 1, np.int64); np.cumsum(followees, out=r_ptr[1:]); r_idx = src[order]

pos = np.full(int(ids.max()) + 1, -1, np.int64); pos[ids] = np.arange(n)
raw = np.loadtxt(gzip.open(os.path.join(HIGGS, 'higgs-activity_time.txt.gz'), 'rt'), dtype=str)
a_, b_, ts, kind = raw[:, 0].astype(np.int64), raw[:, 1].astype(np.int64), raw[:, 2].astype(np.int64), raw[:, 3]
ok = (a_ <= ids.max()) & (b_ <= ids.max()); a_, b_, ts, kind = pos[a_[ok]], pos[b_[ok]], ts[ok], kind[ok]
ok = (a_ >= 0) & (b_ >= 0) & (kind == 'RT') & (a_ != b_); R_a, R_b, R_t = a_[ok], b_[ok], ts[ok].astype(float)
o = np.lexsort((R_t, R_a, R_b)); R_a, R_b, R_t = R_a[o], R_b[o], R_t[o]
first = np.ones(len(R_a), bool); first[1:] = (R_a[1:] != R_a[:-1]) | (R_b[1:] != R_b[:-1])
R_a, R_b, R_t = R_a[first], R_b[first], R_t[first]
uniq, st = np.unique(R_b, return_index=True); en = np.append(st[1:], len(R_b))
roots = [(r, s, e) for r, s, e in zip(uniq, st, en) if e - s >= MIN_SIZE]
print('%d real cascades with >= %d retweeters [%.0fs]' % (len(roots), MIN_SIZE, time.time() - t0))

tadopt = np.full(n, np.inf)
def infer(root, nodes, times):
    tadopt[root] = -np.inf; tadopt[nodes] = times
    par = np.full(len(nodes), -1, np.int64)
    for i, (v, tv) in enumerate(zip(nodes, times)):
        f = r_idx[r_ptr[v]:r_ptr[v + 1]]
        if f.size:
            tf = tadopt[f]; okm = tf < tv
            if okm.any():
                par[i] = f[okm][np.argmax(tf[okm])]
    tadopt[root] = np.inf; tadopt[nodes] = np.inf
    return par

# global organic delay pool (root -> adopter offsets from first adoption) for mimic timing
all_off = np.concatenate([R_t[s:e] - R_t[s:e].min() for _, s, e in roots])
real_fol = followers[R_a]


def feats(root, real_nodes, real_times, real_par, fk_times, fk_follow_root, fk_fol, fk_fee):
    k = len(fk_times)
    times = np.concatenate([real_times, fk_times])
    # parents: real nodes keep inferred parents (fakes are never followed); fakes -> root or external
    par = np.concatenate([real_par, np.where(fk_follow_root, root, -1)])
    is_fake = np.r_[np.zeros(len(real_times), bool), np.ones(k, bool)]
    node_id = np.concatenate([real_nodes, -1 - np.arange(k)])        # fakes get negative ids
    fol = np.concatenate([followers[real_nodes], fk_fol]); fee = np.concatenate([followees[real_nodes], fk_fee])
    N = len(times)
    # tree over adopters connected to root
    o = np.argsort(times, kind='stable'); idx = {root: 0}; depth = {root: 0}; parent_of = {}
    for i in o:
        p = par[i]
        if p >= 0 and p in idx:
            idx[node_id[i]] = len(idx); depth[node_id[i]] = depth[p] + 1; parent_of[node_id[i]] = p
    conn = [v for v in idx if v != root]
    d = np.array([depth[v] for v in conn]) if conn else np.array([0])
    kids = {}
    for v, p in parent_of.items(): kids[p] = kids.get(p, 0) + 1
    d1 = [v for v in conn if depth[v] == 1]
    sub = {v: 1 for v in idx}
    for v in sorted(conn, key=lambda x: -depth[x]): sub[parent_of[v]] += sub[v]
    M = len(idx)
    sv = 2.0 * sum(sub[v] * (M - sub[v]) for v in conn) / (M * (M - 1)) if M > 2 else 0.0
    tt = np.sort(times); ia = np.diff(tt)
    jh = np.searchsorted(tt, tt + 3600, side='right')
    return dict(log_size=np.log(N), follow_share=len(conn) / N, broadcast_share=(len(d1) / len(conn)) if conn else 0.0,
                depth=int(d.max()), sv=sv, childless_d1=(np.mean([kids.get(v, 0) == 0 for v in d1]) if d1 else 1.0),
                top1h_share=(jh - np.arange(N)).max() / N, frac_lt60s=float((ia < 60).mean()) if N > 1 else 0.0,
                ia_cv=float(ia.std() / ia.mean()) if N > 2 and ia.mean() > 0 else 0.0,
                med_log_followers=float(np.median(np.log1p(fol))), zero_follower_share=float((fol == 0).mean()),
                zero_followee_share=float((fee == 0).mean()), fake_share=k / N)


def campaign(kind_, k, real_times):
    t_first = real_times.min()
    if kind_ in ('dump', 'dump_follow'):
        tm = t_first + rng.uniform(0, 6 * 3600) + rng.uniform(0, 3600, k)
        fol = np.zeros(k, int); fee = np.zeros(k, int) if kind_ == 'dump' else np.ones(k, int)
        return tm, kind_ == 'dump_follow', fol, fee
    if kind_ == 'drip':
        tm = t_first + rng.uniform(0, 3 * 86400, k)
        low = real_fol[real_fol <= np.median(real_fol)]
        return tm, True, rng.choice(low, k), np.maximum(1, rng.choice(followees[R_a], k))
    if kind_ == 'mimic':
        off = real_times - t_first
        pool = off if len(off) >= 20 else all_off
        tm = t_first + rng.choice(pool, k) + rng.uniform(0, 600, k)
        return tm, True, rng.choice(real_fol, k), rng.choice(followees[R_a], k)
    raise ValueError(kind_)


KINDS, KS = ['dump', 'dump_follow', 'drip', 'mimic'], [10, 50, 200]
rows = []
for j, (r, s, e) in enumerate(roots):
    nodes, times = R_a[s:e], R_t[s:e]
    par = infer(r, nodes, times)
    base = dict(root=int(r), real_size=int(e - s))
    rows.append(base | feats(r, nodes, times, par, np.array([]), np.zeros(0, bool), np.array([], int), np.array([], int)) | dict(y=0, kind='organic', k=0))
    kd = KINDS[j % len(KINDS)]; kk = KS[(j // len(KINDS)) % len(KS)]      # one campaign per root, balanced design
    tm, fr, fol, fee = campaign(kd, kk, times)
    rows.append(base | feats(r, nodes, times, par, tm, np.full(kk, fr), fol, fee) | dict(y=1, kind=kd, k=kk))
    if (j + 1) % 5000 == 0:
        print('  %d/%d cascades [%.0fs]' % (j + 1, len(roots), time.time() - t0))
D = pd.DataFrame(rows)
# log_size is excluded: injected cascades are always larger than their organic twins, so size
# would leak the label in this paired design (real targets' organic size is unknown).
FEATS = ['follow_share', 'broadcast_share', 'depth', 'sv', 'childless_d1', 'top1h_share', 'frac_lt60s',
         'ia_cv', 'med_log_followers', 'zero_follower_share', 'zero_followee_share']
FAMILIES = {'timing': ['top1h_share', 'frac_lt60s', 'ia_cv'],
            'structure': ['follow_share', 'broadcast_share', 'depth', 'sv', 'childless_d1'],
            'accounts': ['med_log_followers', 'zero_follower_share', 'zero_followee_share'],
            'all': FEATS}

roots_u = D.root.unique(); rng.shuffle(roots_u); test_roots = set(roots_u[: len(roots_u) // 3])
te_mask = D.root.isin(test_roots); tr, te = D[~te_mask], D[te_mask]
def rec1(y, s):
    f, t, _ = roc_curve(y, s); return float(np.interp(0.01, f, t))
out = dict(n_cascades=len(roots), min_size=MIN_SIZE, by_family={}, by_kind={}, generalisation={})
for fam, cols in FAMILIES.items():
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=0).fit(tr[cols], tr.y)
    p = clf.predict_proba(te[cols])[:, 1]
    out['by_family'][fam] = dict(auc=float(roc_auc_score(te.y, p)), recall_at_1pct_fpr=rec1(te.y.values, p))
    if fam == 'all':
        org = te.y.values == 0
        for kd in KINDS:
            for kk in KS:
                msk = (te.kind.values == kd) & (te.k.values == kk)
                yy = np.r_[np.zeros(org.sum()), np.ones(msk.sum())]; ss = np.r_[p[org], p[msk]]
                out['by_kind']['%s_k%d' % (kd, kk)] = dict(n=int(msk.sum()), auc=float(roc_auc_score(yy, ss)), recall_at_1pct_fpr=rec1(yy, ss),
                                                          median_fake_share=float(te.fake_share.values[msk].median() if hasattr(te.fake_share.values[msk], 'median') else np.median(te.fake_share.values[msk])))
    if fam == 'all':
        out['by_fake_share'] = {}
        for lo, hi in ((0, 0.2), (0.2, 0.5), (0.5, 0.8), (0.8, 1.01)):
            msk = (te.y.values == 1) & (te.fake_share.values >= lo) & (te.fake_share.values < hi)
            if msk.sum() >= 20:
                yy = np.r_[np.zeros(org.sum()), np.ones(msk.sum())]; ss = np.r_[p[org], p[msk]]
                out['by_fake_share']['%.1f-%.1f' % (lo, min(hi, 1))] = dict(n=int(msk.sum()), auc=float(roc_auc_score(yy, ss)), recall_at_1pct_fpr=rec1(yy, ss))
    print('%-10s AUC %.3f  recall@1%%FPR %.2f' % (fam, out['by_family'][fam]['auc'], out['by_family'][fam]['recall_at_1pct_fpr']))
for k_, v in out['by_fake_share'].items():
    print('  fake share %-8s n=%4d  AUC %.3f  recall@1%% %.2f' % (k_, v['n'], v['auc'], v['recall_at_1pct_fpr']))
for k_, v in out['by_kind'].items():
    print('  %-16s n=%4d  fake share %.2f  AUC %.3f  recall@1%% %.2f' % (k_, v['n'], v['median_fake_share'], v['auc'], v['recall_at_1pct_fpr']))
# generalisation: train on crude archetypes only, test on mimic
crude = tr[(tr.y == 0) | tr.kind.isin(['dump', 'dump_follow', 'drip'])]
clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=0).fit(crude[FEATS], crude.y)
tm_ = te[(te.y == 0) | (te.kind == 'mimic')]; p = clf.predict_proba(tm_[FEATS])[:, 1]
out['generalisation']['train_crude_test_mimic'] = dict(auc=float(roc_auc_score(tm_.y, p)), recall_at_1pct_fpr=rec1(tm_.y.values, p))
print('trained on crude archetypes, tested on mimic: AUC %.3f recall@1%% %.2f' % (out['generalisation']['train_crude_test_mimic']['auc'],
                                                                                   out['generalisation']['train_crude_test_mimic']['recall_at_1pct_fpr']))
out['feature_medians'] = {g: {f: float(v) for f, v in D[D.kind == g][FEATS].median().items()} for g in ['organic'] + KINDS}
json.dump(out, open(os.path.join(RES, 'b_twitter.json'), 'w'), indent=1)
print('total %.0fs' % (time.time() - t0))
