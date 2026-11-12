"""Phase B (GitHub) - a timing-only detector for star campaigns, under a leakage-aware protocol.

Inputs (MV_DATA/starscout): low_activity_stars (every star of 4.9k repos StarScout
examined, each with its low-activity verdict), clustered_stars (lockstep verdicts),
sample_stars (10k matched normal repos).

Why timing only. Two other feature families leak the label in this data:
  - account profiles: StarScout fetched profiles mostly for suspects (72% of low-activity-
    flagged stargazers have one vs 25% of normal-repo stargazers), so profile-based
    features carry the selection, not just the signal (reported separately, caveated);
  - co-adoption/lockstep against the tables' repo pool: the pool over-represents suspected
    repos, so a fake account's other stars land in the pool and an organic account's don't.
    A fair co-adoption feature needs full GH Archive data (deferred).

Labels (per repo, needs the repo's full star history):
  LA positive   >= 20 stars flagged low-activity and >= 50% of the repo's stars    (headline)
  LA loose      >= 20 flagged and >= 10%
  CL positive   lockstep-flagged repos whose table holds their full history (checked
                against StarScout's n_stars); PARTLY CIRCULAR for timing features, since
                lockstep means many accounts starring in a short window.
  negatives     StarScout's normal sample. Sub-day "dump" repos (>90% of stars within one
                day, ~10% of the sample) are ambiguous: scored and reported, excluded from
                the headline negatives, included in a sensitivity run.
  hard negatives repos StarScout examined whose stars are < 2% low-activity-flagged,
                size-matched to the positives (unmatched, repo size alone separates them).
The lockstep repos with full histories all fall in 2024, so they use stratified 5-fold CV.
Split by time: the repo's busiest day before 2024 = train, 2024 = test.

Features (each repo's own star times):
  top1d/top7d/top30d share        SignalGraph's burst concentration (its baseline score:
                                  0.45 top1d + 0.35 top7d + 0.20 top30d)
  burst_hour_share                share of the busiest day's stars that land in its busiest hour
  burst_min_share                 ... in its busiest 10 minutes
  burst_cv                        inter-arrival CV inside the busiest day (scripts: regular)
  after7_ratio, before7_ratio     stars in the 7 days after / before the busiest day, per
                                  burst star (organic shocks trigger follow-on stars)
  frac_lt60s                      share of inter-arrivals under 60 s
  cv, log_n, active_days_share    overall dispersion, size, share of days with any star

Outputs: results/b_github.json, results/b_github_features.parquet (no repo names are
published from it; it stays local).
"""
import os, json, time
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
from sklearn.model_selection import StratifiedKFold, cross_val_predict

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get('MV_DATA', os.path.expanduser('~/projects/_data/manufactured-virality-data')) + '/starscout'
Z = os.path.expanduser('~/projects/_data/starscout/StarScout/data/250101')
RES = os.path.join(HERE, '..', 'results'); os.makedirs(RES, exist_ok=True)
SPLIT = pd.Timestamp('2024-01-01').timestamp() / 86400
t0 = time.time()


def features(t):
    t = np.sort(t / 86400.0); n = len(t)
    f = {'log_n': np.log(n)}
    for d, k in ((1, 'top1d'), (7, 'top7d'), (30, 'top30d')):
        j = np.searchsorted(t, t + d, side='right'); f[k] = (j - np.arange(n)).max() / n
    j = np.searchsorted(t, t + 1, side='right'); b = int(np.argmax(j - np.arange(n))); b_end = t[b] + 1
    day = t[b:j[b]]
    jh = np.searchsorted(day, day + 1 / 24, side='right'); f['burst_hour_share'] = (jh - np.arange(len(day))).max() / len(day)
    jm = np.searchsorted(day, day + 10 / 1440, side='right'); f['burst_min_share'] = (jm - np.arange(len(day))).max() / len(day)
    ia_d = np.diff(day); f['burst_cv'] = ia_d.std() / ia_d.mean() if len(ia_d) > 1 and ia_d.mean() > 0 else 0.0
    f['after7_ratio'] = ((t >= b_end) & (t < b_end + 7)).sum() / len(day)
    f['before7_ratio'] = ((t < t[b]) & (t >= t[b] - 7)).sum() / len(day)
    ia = np.diff(t)
    f['cv'] = ia.std() / ia.mean() if ia.mean() > 0 else 0.0
    f['frac_lt60s'] = (ia < 60 / 86400).mean()
    span = max(t[-1] - t[0], 1.0); f['active_days_share'] = len(np.unique(np.floor(t))) / np.ceil(span)
    f['burst_day'] = t[b]
    f['sg_burst'] = 0.45 * f['top1d'] + 0.35 * f['top7d'] + 0.20 * f['top30d']
    return f


def repo_table(df, source):
    rows = []
    for repo, g in df.groupby('repo', sort=False):
        tt = g.t.dropna().values.astype(float)
        if len(tt) >= 20:
            r = features(tt); r['repo'] = repo; r['source'] = source; rows.append(r)
    return pd.DataFrame(rows)


# ---- label tables
la = pd.read_parquet(os.path.join(DATA, 'low_activity_stars.parquet')).drop_duplicates(['repo', 'actor'])
la_g = la.groupby('repo').low_activity.agg(['size', 'sum', 'mean'])
cl = pd.read_parquet(os.path.join(DATA, 'clustered_stars.parquet')).drop_duplicates(['repo', 'actor'])
cr = pd.read_csv(os.path.join(Z, 'fake_stars_clustered_repos.csv')).drop_duplicates('repo_name').set_index('repo_name')
cl_g = cl.groupby('repo').clustered.agg(['size', 'sum', 'mean']).join(cr[['n_stars']], how='left')
cl_full = cl_g[(cl_g['sum'] >= 20) & (cl_g['mean'] >= 0.5) & (cl_g['size'] >= 0.9 * cl_g.n_stars)]
print('LA repos %d; CL repos %d, with full history in table %d [%.0fs]' % (len(la_g), len(cl_g), len(cl_full), time.time() - t0))

la_pos = la_g[(la_g['sum'] >= 20) & (la_g['mean'] >= 0.5)].index
la_loose = la_g[(la_g['sum'] >= 20) & (la_g['mean'] >= 0.1)].index
la_hardneg = la_g[la_g['mean'] < 0.02].index

F_la = repo_table(la[la.repo.isin(set(la_loose) | set(la_hardneg))], 'la')
F_cl = repo_table(cl[cl.repo.isin(cl_full.index)], 'cl')
s = pd.read_parquet(os.path.join(DATA, 'sample_stars.parquet')).dropna().drop_duplicates(['repo', 'actor'])
F_s = repo_table(s, 'sample')
F_s['dump'] = F_s.top1d > 0.9
print('features: LA %d, CL %d, sample %d (dumps %d) [%.0fs]' % (len(F_la), len(F_cl), len(F_s), F_s.dump.sum(), time.time() - t0))

FEATS = ['top1d', 'top7d', 'top30d', 'burst_hour_share', 'burst_min_share', 'burst_cv', 'after7_ratio',
         'before7_ratio', 'cv', 'frac_lt60s', 'log_n', 'active_days_share']


def recall_at_fpr(y, score, fpr_target=0.01):
    fpr, tpr, _ = roc_curve(y, score)
    return float(np.interp(fpr_target, fpr, tpr))


def evaluate(pos, neg, label, split='time'):
    P = pos.assign(y=1); N = neg.assign(y=0); D = pd.concat([P, N], ignore_index=True)
    if split == 'time':
        tr, te = D[D.burst_day < SPLIT], D[D.burst_day >= SPLIT]
    else:
        tr = te = D
    res = dict(split=split, n_train=dict(pos=int(tr.y.sum()), neg=int((1 - tr.y).sum())), n_test=dict(pos=int(te.y.sum()), neg=int((1 - te.y).sum())))
    if te.y.nunique() < 2 or tr.y.nunique() < 2:
        res['note'] = 'not enough of both classes in train/test'; return res
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=0)
    if split == 'time':
        clf.fit(tr[FEATS], tr.y); p = clf.predict_proba(te[FEATS])[:, 1]
    else:   # stratified 5-fold, out-of-fold scores
        p = cross_val_predict(clf, D[FEATS], D.y, cv=StratifiedKFold(5, shuffle=True, random_state=0), method='predict_proba')[:, 1]
        clf.fit(D[FEATS], D.y)
    for name, sc in (('signalgraph_burst', te.sg_burst.values), ('timing_gbm', p)):
        res[name] = dict(roc_auc=float(roc_auc_score(te.y, sc)), pr_auc=float(average_precision_score(te.y, sc)),
                         recall_at_1pct_fpr=recall_at_fpr(te.y.values, sc))
    # single-feature AUCs on test (direction-free)
    res['feature_auc'] = {f: float(max(roc_auc_score(te.y, te[f]), 1 - roc_auc_score(te.y, te[f]))) for f in FEATS}
    print('%-45s test pos %4d neg %4d | SignalGraph burst AUC %.3f R@1%% %.2f | timing GBM AUC %.3f R@1%% %.2f'
          % (label, res['n_test']['pos'], res['n_test']['neg'], res['signalgraph_burst']['roc_auc'],
             res['signalgraph_burst']['recall_at_1pct_fpr'], res['timing_gbm']['roc_auc'], res['timing_gbm']['recall_at_1pct_fpr']))
    res['_clf'] = clf
    return res


neg_clean = F_s[~F_s.dump]; neg_all = F_s
out = {}
out['LA_strict_vs_normal'] = evaluate(F_la[F_la.repo.isin(la_pos)], neg_clean, 'LA strict vs normal (no dumps)')
out['LA_loose_vs_normal'] = evaluate(F_la[F_la.repo.isin(la_loose)], neg_clean, 'LA loose vs normal (no dumps)')
out['LA_strict_vs_normal_with_dumps'] = evaluate(F_la[F_la.repo.isin(la_pos)], neg_all, 'LA strict vs normal (dumps as negatives)')
# hard negatives are ~100x larger than positives (size alone gives AUC 1.0): size-matched subset only
pos_la = F_la[F_la.repo.isin(la_pos)]
hn = F_la[F_la.repo.isin(la_hardneg) & (F_la.log_n <= pos_la.log_n.quantile(0.99))]
out['LA_strict_vs_hardneg_sizematched'] = evaluate(pos_la, hn, 'LA strict vs examined-but-clean, size-matched')
# the campaigns that are NOT one-day dumps: what matters once attackers stop being crude
spread = pos_la[pos_la.top1d < 0.9]
out['LA_strict_spread_vs_normal'] = evaluate(spread, neg_clean, 'LA strict, spread-out campaigns only (top1d<0.9)', split='cv')
out['LA_strict_spread_vs_normal']['n_spread_campaigns'] = int(len(spread))
# all full-history lockstep repos fall in 2024: no temporal split possible
out['CL_vs_normal'] = evaluate(F_cl, neg_clean, 'CL (partly circular, 5-fold CV) vs normal', split='cv')

# how the headline detector scores the ambiguous dumps
clf = out['LA_strict_vs_normal'].get('_clf')
if clf is not None:
    thr_src = neg_clean[neg_clean.burst_day >= SPLIT]
    thr = np.quantile(clf.predict_proba(thr_src[FEATS])[:, 1], 0.99)        # 1% FPR threshold on clean normals
    d = F_s[F_s.dump]
    out['dumps_flagged_at_1pct_fpr'] = float((clf.predict_proba(d[FEATS])[:, 1] >= thr).mean())
    print('ambiguous sub-day dumps flagged by the LA-trained detector at 1%% FPR: %.2f' % out['dumps_flagged_at_1pct_fpr'])

# feature profile by class (medians)
prof = {}
for name, D in (('normal', neg_clean), ('dumps', F_s[F_s.dump]), ('LA_strict', F_la[F_la.repo.isin(la_pos)]),
                ('hard_neg', F_la[F_la.repo.isin(la_hardneg)]), ('CL', F_cl)):
    prof[name] = {f: float(D[f].median()) for f in FEATS} | {'n': int(len(D))}
out['feature_medians'] = prof
for k in out:
    if isinstance(out[k], dict): out[k].pop('_clf', None)
json.dump(out, open(os.path.join(RES, 'b_github.json'), 'w'), indent=1)
pd.concat([F_la.assign(label=np.where(F_la.repo.isin(la_pos), 'LA_strict', np.where(F_la.repo.isin(la_hardneg), 'hard_neg', 'LA_loose'))),
           F_cl.assign(label='CL'), F_s.assign(label=np.where(F_s.dump, 'dump', 'normal'))]).to_parquet(os.path.join(DATA, 'b_github_features.parquet'))
print(json.dumps(prof, indent=1)); print('total %.0fs' % (time.time() - t0))
