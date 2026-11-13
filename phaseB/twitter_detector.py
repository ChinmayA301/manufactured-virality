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
MIN_SIZE = 3   # also in twitter_common
rng = np.random.default_rng(31)
t0 = time.time()

import sys; sys.path.insert(0, HERE)
from twitter_common import (roots, R_a, R_b, R_t, followers, followees, real_fol, all_off, cascade, feats,
                            FEATS, FAMILIES, EXTERNAL)
print('%d real cascades with >= %d retweeters [%.0fs]' % (len(roots), MIN_SIZE, time.time() - t0))


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
    nodes, times, par = cascade(r, s, e)
    base = dict(root=int(r), real_size=int(e - s))
    rows.append(base | feats(r, nodes, times, par) | dict(y=0, kind='organic', k=0))
    kd = KINDS[j % len(KINDS)]; kk = KS[(j // len(KINDS)) % len(KS)]      # one campaign per root, balanced design
    tm, fr, fol, fee = campaign(kd, kk, times)
    rows.append(base | feats(r, nodes, times, par, tm, np.full(kk, r if fr else EXTERNAL), fol, fee) | dict(y=1, kind=kd, k=kk))
    if (j + 1) % 5000 == 0:
        print('  %d/%d cascades [%.0fs]' % (j + 1, len(roots), time.time() - t0))
D = pd.DataFrame(rows)
# log_size is excluded from FEATS (twitter_common): injected cascades are always larger than their
# organic twins, so size would leak the label in this paired design.
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
