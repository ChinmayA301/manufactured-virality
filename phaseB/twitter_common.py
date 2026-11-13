"""Shared Twitter (Higgs) cascade code for Phase B4 (detector) and Phase C (game).

Loads the follower graph and the follow-explained... (actually: all first retweets per
(retweeter, source) pair), groups them into real cascades per source, infers parents from the
follow graph, and computes the detector's cascade features with optional injected fakes.

Fake accounts are new nodes. Real users never follow them, so a fake can never be the parent
of a real adopter. A fake's parent is set by the attacker: the target (it follows the target),
an earlier fake (sockpuppets following each other, which lets fakes form chains), or none
(external). Node ids: real >= 0; EXTERNAL = -1; fake i = -(i + 2), so no fake id collides
with the external marker.
"""
import os, gzip
import numpy as np

HIGGS = os.environ.get('HIGGS_DIR', os.path.expanduser('~/projects/blog-artifacts/virality-bayesian-optimization/pipeline'))
EXTERNAL = -1
MIN_SIZE = 3


def fake_id(i):
    return -(np.asarray(i) + 2)


z = np.load(os.path.join(HIGGS, 'follower_graph.npz'))
indptr, indices, ids = z['indptr'].astype(np.int64), z['indices'].astype(np.int64), z['ids']
n = len(ids); src = np.repeat(np.arange(n), np.diff(indptr))
followers = np.diff(indptr); followees = np.bincount(indices, minlength=n)
_order = np.argsort(indices, kind='stable'); r_ptr = np.zeros(n + 1, np.int64); np.cumsum(followees, out=r_ptr[1:]); r_idx = src[_order]

_pos = np.full(int(ids.max()) + 1, -1, np.int64); _pos[ids] = np.arange(n)
_raw = np.loadtxt(gzip.open(os.path.join(HIGGS, 'higgs-activity_time.txt.gz'), 'rt'), dtype=str)
_a, _b, _ts, _k = _raw[:, 0].astype(np.int64), _raw[:, 1].astype(np.int64), _raw[:, 2].astype(np.int64), _raw[:, 3]
_ok = (_a <= ids.max()) & (_b <= ids.max()); _a, _b, _ts, _k = _pos[_a[_ok]], _pos[_b[_ok]], _ts[_ok], _k[_ok]
_ok = (_a >= 0) & (_b >= 0) & (_k == 'RT') & (_a != _b); R_a, R_b, R_t = _a[_ok], _b[_ok], _ts[_ok].astype(float)
_o = np.lexsort((R_t, R_a, R_b)); R_a, R_b, R_t = R_a[_o], R_b[_o], R_t[_o]
_first = np.ones(len(R_a), bool); _first[1:] = (R_a[1:] != R_a[:-1]) | (R_b[1:] != R_b[:-1])
R_a, R_b, R_t = R_a[_first], R_b[_first], R_t[_first]
_u, _st = np.unique(R_b, return_index=True); _en = np.append(_st[1:], len(R_b))
roots = [(int(r), int(s), int(e)) for r, s, e in zip(_u, _st, _en) if e - s >= MIN_SIZE]
real_fol = followers[R_a]; real_fee = followees[R_a]
all_off = np.concatenate([R_t[s:e] - R_t[s:e].min() for _, s, e in roots])     # organic offsets from first adoption

_tadopt = np.full(n, np.inf)


def infer(root, nodes, times):
    """Parent of each real adopter = most recent followee adopted before it (root adopts first)."""
    _tadopt[root] = -np.inf; _tadopt[nodes] = times
    par = np.full(len(nodes), EXTERNAL, np.int64)
    for i, (v, tv) in enumerate(zip(nodes, times)):
        f = r_idx[r_ptr[v]:r_ptr[v + 1]]
        if f.size:
            tf = _tadopt[f]; okm = tf < tv
            if okm.any():
                par[i] = f[okm][np.argmax(tf[okm])]
    _tadopt[root] = np.inf; _tadopt[nodes] = np.inf
    return par


def cascade(r, s, e):
    nodes, times = R_a[s:e], R_t[s:e]
    return nodes, times, infer(r, nodes, times)


FEATS = ['follow_share', 'broadcast_share', 'depth', 'sv', 'childless_d1', 'top1h_share', 'frac_lt60s',
         'ia_cv', 'med_log_followers', 'zero_follower_share', 'zero_followee_share']
FAMILIES = {'timing': ['top1h_share', 'frac_lt60s', 'ia_cv'],
            'structure': ['follow_share', 'broadcast_share', 'depth', 'sv', 'childless_d1'],
            'accounts': ['med_log_followers', 'zero_follower_share', 'zero_followee_share'],
            'all': FEATS}


def feats(root, real_nodes, real_times, real_par, fk_times=None, fk_par=None, fk_fol=None, fk_fee=None):
    """Cascade features. fk_par: per fake, the root id, a fake id (fake_id(j), j earlier), or EXTERNAL."""
    if fk_times is None or len(fk_times) == 0:
        fk_times, fk_par = np.array([]), np.array([], np.int64)
        fk_fol, fk_fee = np.array([], int), np.array([], int)
    k = len(fk_times)
    times = np.concatenate([real_times, fk_times])
    par = np.concatenate([real_par, fk_par]).astype(np.int64)
    node_id = np.concatenate([real_nodes, fake_id(np.arange(k))]).astype(np.int64)
    fol = np.concatenate([followers[real_nodes], fk_fol]); fee = np.concatenate([followees[real_nodes], fk_fee])
    N = len(times)
    o = np.argsort(times, kind='stable'); idx = {root: 0}; depth = {root: 0}; parent_of = {}
    for i in o:
        p = int(par[i])
        if p != EXTERNAL and p in idx:
            v = int(node_id[i]); idx[v] = len(idx); depth[v] = depth[p] + 1; parent_of[v] = p
    conn = [v for v in idx if v != root]
    d = np.array([depth[v] for v in conn]) if conn else np.array([0])
    kids = {}
    for v, p in parent_of.items():
        kids[p] = kids.get(p, 0) + 1
    d1 = [v for v in conn if depth[v] == 1]
    sub = {v: 1 for v in idx}
    for v in sorted(conn, key=lambda x: -depth[x]):
        sub[parent_of[v]] += sub[v]
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
