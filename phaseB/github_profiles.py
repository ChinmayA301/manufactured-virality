"""Phase B (GitHub) - account-profile features, reported with their coverage problem.

StarScout fetched account profiles (actors.parquet) mostly for suspected accounts, so
"has a profile" itself tracks the label. Every profile feature here is therefore computed
only over stargazers whose profile was fetched, and coverage is reported per class first.

Features per repo (over fetched stargazers):
  coverage            share of the repo's stargazers with a fetched profile     (NOT a feature;
                      reported to show the selection)
  deleted_share       fetched profiles that returned an error (account gone)
  low_activity_share  public_repos <= 2 and followers <= 3 (SignalGraph's rule; nearly the
                      StarScout low-activity heuristic, so scored against LOCKSTEP labels only)
  recent_share        account created <= 14 days before the star (SignalGraph's rule)
Cross-heuristic scoring: low_activity_share vs lockstep labels; recent_share and
deleted_share vs both. 5-fold CV AUC, repos with >= 10 fetched stargazers.

Outputs: results/b_github_profiles.json
"""
import os, json
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get('MV_DATA', os.path.expanduser('~/projects/_data/manufactured-virality-data')) + '/starscout'
RES = os.path.join(HERE, '..', 'results')

F = pd.read_parquet(os.path.join(DATA, 'b_github_features.parquet'))[['repo', 'label']]
act = pd.read_parquet(os.path.join(DATA, 'actors.parquet')).drop_duplicates('actor').set_index('actor')
stars = pd.concat([
    pd.read_parquet(os.path.join(DATA, 'low_activity_stars.parquet'), columns=['repo', 'actor', 't']),
    pd.read_parquet(os.path.join(DATA, 'clustered_stars.parquet'), columns=['repo', 'actor', 't']),
    pd.read_parquet(os.path.join(DATA, 'sample_stars.parquet'))]).dropna()
stars = stars[stars.repo.isin(set(F.repo))].drop_duplicates(['repo', 'actor'])
j = stars.join(act, on='actor', how='left')
j['fetched'] = j.error.notna()
j['ok'] = j.fetched & (j.error == False)
j['low_act'] = j.ok & (j.public_repos.fillna(99) <= 2) & (j.followers.fillna(99) <= 3)
j['recent'] = j.ok & j.created_at.notna() & ((j.t - j.created_at) <= 14 * 86400)
g = j.groupby('repo').agg(n=('actor', 'size'), fetched=('fetched', 'sum'), ok=('ok', 'sum'),
                          deleted=('error', lambda x: (x == True).sum()), low_act=('low_act', 'sum'), recent=('recent', 'sum'))
g['coverage'] = g.fetched / g.n
g['deleted_share'] = g.deleted / g.fetched.clip(lower=1)
g['low_activity_share'] = g.low_act / g.ok.clip(lower=1)
g['recent_share'] = g.recent / g.ok.clip(lower=1)
g = g.join(F.drop_duplicates('repo').set_index('repo'))
out = dict(coverage_by_class={}, medians={}, auc={})
for lab, d in g.groupby('label'):
    out['coverage_by_class'][lab] = dict(repos=int(len(d)), median_coverage=float(d.coverage.median()),
                                         repos_with_10_fetched=int((d.fetched >= 10).sum()))
    out['medians'][lab] = {c: float(d.loc[d.fetched >= 10, c].median()) for c in ('deleted_share', 'low_activity_share', 'recent_share')}
print(pd.DataFrame(out['coverage_by_class']).T); print(pd.DataFrame(out['medians']).T.round(3))

def cv_auc(pos_lab, cols, neg_lab='normal'):
    d = g[(g.fetched >= 10) & g.label.isin([pos_lab, neg_lab])]
    if d.label.nunique() < 2 or (d.label == pos_lab).sum() < 20:
        return dict(note='too few repos with >= 10 fetched stargazers', n=int(len(d)))
    y = (d.label == pos_lab).astype(int).values
    p = cross_val_predict(HistGradientBoostingClassifier(max_iter=200, random_state=0), d[cols], y,
                          cv=StratifiedKFold(5, shuffle=True, random_state=0), method='predict_proba')[:, 1]
    single = {c: float(max(roc_auc_score(y, d[c]), 1 - roc_auc_score(y, d[c]))) for c in cols}
    return dict(n_pos=int(y.sum()), n_neg=int((1 - y).sum()), auc=float(roc_auc_score(y, p)), single=single)
out['auc']['CL_vs_normal_profile_all'] = cv_auc('CL', ['deleted_share', 'low_activity_share', 'recent_share'])
out['auc']['LA_strict_vs_normal_no_lowactivity'] = cv_auc('LA_strict', ['deleted_share', 'recent_share'])
out['auc']['LA_strict_vs_normal_with_lowactivity_CIRCULAR'] = cv_auc('LA_strict', ['deleted_share', 'low_activity_share', 'recent_share'])
for k, v in out['auc'].items(): print(k, v)
json.dump(out, open(os.path.join(RES, 'b_github_profiles.json'), 'w'), indent=1)
