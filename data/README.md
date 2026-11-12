# Data

Nothing here is committed. Working files live outside the repo, outside iCloud.

| Where | What |
|---|---|
| `~/projects/_data/manufactured-virality-data/starscout/` (`$MV_DATA`) | Working copy: Parquet extracted from StarScout's Mongo dump, about 1.1 GB |
| `~/projects/_data/starscout/StarScout/` | StarScout code and CSVs (`StarScout.zip`, 164 MB) |
| Google Drive `manufactured-virality-data/` | Archive copy only |
| `~/projects/blog-artifacts/virality-bayesian-optimization/pipeline/` (`$HIGGS_DIR`) | Higgs follower graph, activity log and calibration (from the BO study) |

**Why a local working copy.** On 12 Nov 2026, three of the four Parquet files written into the Google Drive mount failed to decode within minutes ("ZSTD decompression failed", corrupt page headers), while the folder occupied about 100 MB on disk against 1.1 GB of files. The streamed Drive mount evicted or replaced data mid-sync, the same failure mode as iCloud's dataless files. Treat Drive as an archive: copy files there after writing, and verify them by reading them back. Don't read from the mount in analysis code.

## Sources

- **StarScout replication package:** He et al., ICSE 2026. Zenodo 17009694, CC-BY-4.0. `extract_starscout.py` streams the 5.6 GB `mongodb.zip` over HTTP range requests and keeps only the columns used here. Labels mark *suspected* fake stars; never name individual repositories or accounts.
- **SNAP Higgs Twitter dataset:** follower graph and activity log.
