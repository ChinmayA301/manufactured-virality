"""Phase B (GitHub) - does a fair co-adoption feature add to timing?

Co-adoption (from GH Archive, BigQuery sandbox; query in phaseB/coadopt.sql): for each star,
how many OTHER repositories the same account starred on the same UTC day, across all of
GitHub. This is computed from public events available at the time of the star and does not
depend on StarScout's repo pool, so it avoids the pool leakage noted in github_detector.py.
It is close to the lockstep heuristic's definition, so it is scored against LOW-ACTIVITY labels
(headline) and only descriptively against lockstep labels.

Families compared with the same protocol as github_detector.py (time split; spread-out
campaigns with 5-fold CV):  timing | co-adoption | timing + co-adoption

Outputs: results/b_github_coadopt.json
"""
import os, json
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, cross_val_predict

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get('MV_DATA', os.path.expanduser('~/projects/_data/manufactured-virality-data')) + '/starscout'
RES = os.path.join(HERE, '..', 'results')
SPLIT = pd.Timestamp('2024-01-01').timestamp() / 86400

F = pd.read_parquet(os.path.join(DATA, 'b_github_features.parquet')).drop_duplicates('repo')
C = pd.read_csv(os.path.join(DATA, 'coadopt_by_repo.csv'))
C['log_mean_other'] = np.log1p(C.mean_other_same_day)
D = F.merge(C, on='repo', how='inner')
TIMING = ['top1d', 'top7d', 'top30d', 'burst_hour_share', 'burst_min_share', 'burst_cv', 'after7_ratio',
          'before7_ratio', 'cv', 'frac_lt60s', 'log_n', 'active_days_share']
COAD = ['log_mean_other', 'share_5plus_other_same_day', 'share_20plus_other_same_day']
FAM = {'timing': TIMING, 'coadoption': COAD, 'timing+coadoption': TIMING + COAD}

def rec1(y, s):
    f, t, _ = roc_curve(y, s); return float(np.interp(0.01, f, t))

def run(pos, neg, split):
    d = pd.concat([pos.assign(y=1), neg.assign(y=0)], ignore_index=True)
    res = {}
    for fam, cols in FAM.items():
        clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=0)
        if split == 'time':
            tr, te = d[d.burst_day < SPLIT], d[d.burst_day >= SPLIT]
            p = clf.fit(tr[cols], tr.y).predict_proba(te[cols])[:, 1]; y = te.y.values
        else:
            p = cross_val_predict(clf, d[cols], d.y, cv=StratifiedKFold(5, shuffle=True, random_state=0), method='predict_proba')[:, 1]; y = d.y.values
        res[fam] = dict(auc=float(roc_auc_score(y, p)), recall_at_1pct_fpr=rec1(y, p), n_pos=int(y.sum()), n_neg=int((1 - y).sum()))
    return res

normal = D[D.label == 'normal']
la_strict = D[D.label == 'LA_strict']
out = dict(n_repos_matched=int(len(D)), match_rate=float(C.n_matched.sum() / C.n_stars.sum()))
out['LA_strict_vs_normal'] = run(la_strict, normal, 'time')
out['LA_loose_vs_normal'] = run(D[D.label.isin(['LA_strict', 'LA_loose'])], normal, 'time')
out['LA_strict_spread_vs_normal'] = run(la_strict[la_strict.top1d < 0.9], normal, 'cv')
out['CL_vs_normal_descriptive'] = run(D[D.label == 'CL'], normal, 'cv')
out['medians'] = {lab: {c: float(D.loc[D.label == lab, c].median()) for c in COAD} for lab in ['normal', 'dump', 'LA_strict', 'LA_loose', 'CL']}
for k, v in out.items():
    if isinstance(v, dict) and 'timing' in v:
        print('%-28s ' % k + ' | '.join('%s AUC %.3f R@1%% %.2f' % (f, r['auc'], r['recall_at_1pct_fpr']) for f, r in v.items()))
print(pd.DataFrame(out['medians']).round(3)); print('star match rate in GH Archive: %.3f' % out['match_rate'])
json.dump(out, open(os.path.join(RES, 'b_github_coadopt.json'), 'w'), indent=1)
