"""Validate SignalGraph's GraphQL co-adoption measure against the GH Archive measure.

SignalGraph (apps/api/app/services/coadoption_ingest.py) counts, per stargazer, the other repos
they starred on the same UTC day by reading their current star list through GitHub GraphQL.
Phase B measured the same quantity from GH Archive WatchEvents. This script runs the
SignalGraph code on the SAME stars Phase B used (bq_stars.csv.gz: repo, actor, day) and compares
each repo's share of stargazers with 20+ other same-day stars.

Sample: repos with 20-400 GH Archive stars, from StarScout "normal" and lockstep (CL) labels,
up to MAX_STARS stars per repo, production settings (max_pages = 3).

Outputs aggregates only (no repository or account names): results/b_graphql_validation.json
Needs: GITHUB_TOKEN (or `gh auth token`), SIGNALGRAPH_DIR pointing at the SignalGraph checkout.
"""
import asyncio, json, os, subprocess, sys, time
from datetime import datetime, timezone
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get('MV_DATA', os.path.expanduser('~/projects/_data/manufactured-virality-data/starscout'))
SG = os.environ.get('SIGNALGRAPH_DIR', os.path.expanduser('~/projects/signalgraph'))
sys.path.insert(0, os.path.join(SG, 'apps', 'api')); sys.path.insert(0, os.path.join(SG, 'packages', 'scoring'))
import httpx
from app.services.coadoption_ingest import enrich_same_day_stars, FARM_THRESHOLD
from signalgraph_scoring.features import StarEventInput

N_PER_LABEL = int(os.environ.get('N_PER_LABEL', 15)); MAX_STARS = int(os.environ.get('MAX_STARS', 40))
rng = np.random.default_rng(7)
token = os.environ.get('GITHUB_TOKEN') or subprocess.run(['gh', 'auth', 'token'], capture_output=True, text=True).stdout.strip()
H = {'Authorization': f'Bearer {token}', 'Content-Type': 'application/json'}

C = pd.read_csv(os.path.join(DATA, 'coadopt_by_repo_v2.csv'))
F = pd.read_parquet(os.path.join(DATA, 'b_github_features.parquet')).drop_duplicates('repo')[['repo', 'label']]
S = pd.read_csv(os.path.join(DATA, 'bq_stars.csv.gz'))
C = C.merge(F, on='repo')
C = C[(C.n_stars >= 20) & (C.n_stars <= 400) & (C.n_matched / C.n_stars > 0.9)]
SURVIVING_ONLY = os.environ.get('SURVIVING_ONLY') == '1'   # only repos that still exist on GitHub
pools = {lab: C[C.label == lab].sample(frac=1, random_state=7) for lab in ('normal', 'CL')}
pick = None if SURVIVING_ONLY else pd.concat([g.head(N_PER_LABEL) for g in pools.values()])


async def repo_ids(client, names):
    q = 'query{' + ' '.join(f'r{i}: repository(owner: {json.dumps(n.split("/")[0])}, name: {json.dumps(n.split("/")[1])}) {{ databaseId }}'
                            for i, n in enumerate(names)) + '}'
    r = await client.post('https://api.github.com/graphql', headers=H, json={'query': q}); d = r.json().get('data') or {}
    return {n: (d.get(f'r{i}') or {}).get('databaseId') for i, n in enumerate(names)}


out_checked = {}


async def main():
    rows = []; t0 = time.time()
    async with httpx.AsyncClient(timeout=60) as client:
        ids = {}
        global pick
        if SURVIVING_ONLY:
            chosen = []
            for lab, g in pools.items():
                got = []
                for k in range(0, min(len(g), 800), 20):
                    batch = g.repo.iloc[k:k + 20].tolist()
                    found = await repo_ids(client, batch); ids |= found
                    got += [n for n in batch if found.get(n) is not None]
                    if len(got) >= N_PER_LABEL:
                        break
                out_checked[lab] = dict(checked=k + 20, surviving=len(got))
                chosen.append(g[g.repo.isin(got[:N_PER_LABEL])])
            pick = pd.concat(chosen)
        else:
            names = pick.repo.tolist()
            for k in range(0, len(names), 20):
                ids |= await repo_ids(client, names[k:k + 20])
        for j, rec in enumerate(pick.itertuples()):
            st = S[S.repo == rec.repo]
            st = st.sample(min(MAX_STARS, len(st)), random_state=7)
            logins = {i: a for i, a in enumerate(st.actor)}
            ev = [StarEventInput(user_id=i, starred_at=datetime.fromisoformat(d).replace(hour=12, tzinfo=timezone.utc))
                  for i, d in enumerate(st.day)]
            res = await enrich_same_day_stars(client, H, ev, logins, ids.get(rec.repo), max_users=MAX_STARS, time_budget_s=900)
            errs = sorted(set(res.errors))
            vals = [e.same_day_other_stars for e in res.events if e.same_day_other_stars is not None]
            age_years = float((pd.Timestamp('2026-11-13') - pd.to_datetime(st.day)).dt.days.median() / 365.25)
            rows.append(dict(label=rec.label, repo_exists=ids.get(rec.repo) is not None, n=len(ev), resolved=len(vals),
                             lower_bounds=res.lower_bounds, queries=res.queries, median_star_age_years=age_years,
                             gha_share20=float(rec.share_20plus_other_same_day),
                             gql_share20=(float(np.mean([v >= FARM_THRESHOLD for v in vals])) if len(vals) >= 8 else None),
                             gql_active=len(vals) >= 8, errors=','.join(errs)))
            print('%2d/%d %-6s exists=%s resolved %2d/%2d  GHA %.2f  GQL %s  [%.0fs]' % (
                j + 1, len(pick), rec.label, rows[-1]['repo_exists'], len(vals), len(ev), rec.share_20plus_other_same_day,
                '%.2f' % rows[-1]['gql_share20'] if rows[-1]['gql_share20'] is not None else ' -- ', time.time() - t0), flush=True)
    R = pd.DataFrame(rows)
    out = dict(n_repos=len(R), surviving_only=SURVIVING_ONLY, survival_check=out_checked, max_stars_per_repo=MAX_STARS,
               repos_with_fetch_errors=int((R.errors != '').sum()), by_label={})
    for lab, g in R.groupby('label'):
        a = g[g.gql_active]
        out['by_label'][lab] = dict(
            n=len(g), repo_still_exists=float(g.repo_exists.mean()),
            resolved_share=float(g.resolved.sum() / g.n.sum()), signal_active_share=float(g.gql_active.mean()),
            median_star_age_years=float(g.median_star_age_years.median()),
            gha_share20_median=float(g.gha_share20.median()),
            gql_share20_median_when_active=(float(a.gql_share20.median()) if len(a) else None),
            abs_diff_median_when_active=(float((a.gql_share20 - a.gha_share20).abs().median()) if len(a) else None),
            flagged_at_0p3_gha=float((g.gha_share20 > 0.3).mean()),
            flagged_at_0p3_gql_when_active=(float((a.gql_share20 > 0.3).mean()) if len(a) else None))
    act = R[R.gql_active]
    if len(act) >= 5:
        out['spearman_gql_vs_gha'] = float(act[['gql_share20', 'gha_share20']].corr(method='spearman').iloc[0, 1])
    out['resolved_vs_age'] = {b: float(g.resolved.sum() / g.n.sum()) for b, g in
                              R.groupby(pd.cut(R.median_star_age_years, [0, 2, 3, 4, 10]).astype(str))}
    out['graphql_queries'] = int(R.queries.sum())
    json.dump(out, open(os.path.join(HERE, '..', 'results', 'b_graphql_validation%s.json' % ('_surviving' if SURVIVING_ONLY else '')), 'w'), indent=1)
    print(json.dumps(out, indent=1))

asyncio.run(main())
