"""Stream the StarScout MongoDB dump (Zenodo 17009694, CC-BY-4.0) into compact Parquet.

The dump is 5.6 GB. Nothing is stored locally: the zip is read over HTTP range
requests and each .bson.gz member is decompressed and decoded on the fly,
keeping only the columns this study uses.

Outputs (in $MV_DATA, default: Google Drive "manufactured-virality-data/starscout"):
  low_activity_stars.parquet  repo, actor, t, low_activity   every star of repos StarScout examined
                                                             with its low-activity verdict
  clustered_stars.parquet     repo, actor, t, clustered      same, lockstep (CopyCatch) verdict
  sample_stars.parquet        repo, actor, t                 WatchEvents of the matched normal-repo sample
  actors.parquet              actor, created_at, public_repos, followers, following, n_stars, error

Disclaimer carried from StarScout: labels mark *suspected* fake stars; individual
repos/users may be false positives. Use for statistics, never to name anyone.
"""
import os, sys, zlib, ast, time
from datetime import datetime, timezone
import bson
import pyarrow as pa, pyarrow.parquet as pq
from remotezip import RemoteZip

URL = 'https://zenodo.org/api/records/17009694/files/mongodb.zip/content'
DRIVE = os.path.expanduser('~/Library/CloudStorage/GoogleDrive-chinmayarora2001@gmail.com/My Drive')
OUT = os.environ.get('MV_DATA', os.path.join(DRIVE, 'manufactured-virality-data')) + '/starscout'
os.makedirs(OUT, exist_ok=True)
CHUNK = 500_000


def ts(s):
    if not s:
        return None
    s = str(s).replace(' UTC', '').replace('Z', '').replace('T', ' ')
    try:
        return int(datetime.fromisoformat(s).replace(tzinfo=timezone.utc).timestamp())
    except ValueError:
        return None


def docs(z, name):
    """Yield decoded BSON documents from a gzipped mongodump member, streaming."""
    f = z.open('mongodb/fake_stars/%s.bson.gz' % name)
    d = zlib.decompressobj(16 + zlib.MAX_WBITS)
    buf = b''
    while True:
        raw = f.read(1 << 20)
        if raw:
            buf += d.decompress(raw)
        elif not buf:
            return
        while len(buf) >= 4:
            n = int.from_bytes(buf[:4], 'little')
            if len(buf) < n:
                break
            yield bson.decode(buf[:n])
            buf = buf[n:]
        if not raw:
            if buf:
                raise RuntimeError('trailing bytes in %s' % name)
            return


def write(name, schema, rows_iter):
    path = os.path.join(OUT, name + '.parquet')
    tmp = path + '.part'
    w = pq.ParquetWriter(tmp, schema, compression='zstd')
    cols = {f.name: [] for f in schema}
    n = 0; t0 = time.time()
    for row in rows_iter:
        for k in cols:
            cols[k].append(row[k])
        n += 1
        if n % CHUNK == 0:
            w.write_table(pa.table(cols, schema=schema)); cols = {k: [] for k in cols}
            print('  %s: %dk rows [%.0fs]' % (name, n // 1000, time.time() - t0), flush=True)
    if cols[schema[0].name]:
        w.write_table(pa.table(cols, schema=schema))
    w.close(); os.replace(tmp, path)
    print('wrote %s: %d rows, %.1f MB [%.0fs]' % (path, n, os.path.getsize(path) / 1e6, time.time() - t0), flush=True)


def star_rows(z, coll, flag):
    for d in docs(z, coll):
        yield dict(repo=d.get('repo'), actor=d.get('actor'), t=ts(d.get('starred_at')),
                   **{flag: str(d.get(flag)) == 'True'})


def sample_rows(z):
    for d in docs(z, 'sample_repo_events'):
        if d.get('type') == 'WatchEvent':
            yield dict(repo=d.get('repo'), actor=d.get('actor'), t=ts(d.get('created_at')))


def actor_rows(z):
    for d in docs(z, 'actors'):
        info = d.get('info')
        try:
            info = ast.literal_eval(info) if isinstance(info, str) else (info or {})
        except (ValueError, SyntaxError):
            info = {}
        stars = d.get('stars')
        try:
            n_stars = len(ast.literal_eval(stars)) if isinstance(stars, str) else len(stars or [])
        except (ValueError, SyntaxError):
            n_stars = None
        info = info if isinstance(info, dict) else {}
        yield dict(actor=d.get('actor'), created_at=ts(info.get('created_at')),
                   public_repos=info.get('public_repos'), followers=info.get('followers'),
                   following=info.get('following'), n_stars=n_stars, error=str(d.get('error')) == 'True')


S = pa.string(); I = pa.int64()
JOBS = {
    'sample_stars': (pa.schema([('repo', S), ('actor', S), ('t', I)]), sample_rows),
    'clustered_stars': (pa.schema([('repo', S), ('actor', S), ('t', I), ('clustered', pa.bool_())]),
                        lambda z: star_rows(z, 'clustered_stars', 'clustered')),
    'low_activity_stars': (pa.schema([('repo', S), ('actor', S), ('t', I), ('low_activity', pa.bool_())]),
                           lambda z: star_rows(z, 'low_activity_stars', 'low_activity')),
    'actors': (pa.schema([('actor', S), ('created_at', I), ('public_repos', I), ('followers', I),
                          ('following', I), ('n_stars', I), ('error', pa.bool_())]), actor_rows),
}

if __name__ == '__main__':
    todo = sys.argv[1:] or list(JOBS)
    print('output:', OUT)
    for name in todo:
        if os.path.exists(os.path.join(OUT, name + '.parquet')):
            print('skip (exists):', name); continue
        schema, fn = JOBS[name]
        with RemoteZip(URL) as z:
            write(name, schema, fn(z))
