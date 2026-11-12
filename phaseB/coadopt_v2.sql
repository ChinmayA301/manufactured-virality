WITH actors AS (SELECT DISTINCT actor FROM `polyrag-demo-ca.mv_coadopt.stars`),
gh AS (
  SELECT actor.login AS actor, DATE(created_at) AS day, COUNT(DISTINCT repo.name) AS c
  FROM `githubarchive.month.20*`
  WHERE type = 'WatchEvent' AND _TABLE_SUFFIX BETWEEN '1907' AND '2501'
    AND actor.login IN (SELECT actor FROM actors)
  GROUP BY 1, 2)
SELECT s.repo,
       COUNT(*) AS n_stars,
       COUNTIF(gh.c IS NOT NULL) AS n_matched,
       AVG(GREATEST(IFNULL(gh.c, 1) - 1, 0)) AS mean_other_same_day,
       COUNTIF(IFNULL(gh.c, 1) >= 6) / COUNT(*) AS share_5plus_other_same_day,
       COUNTIF(IFNULL(gh.c, 1) >= 21) / COUNT(*) AS share_20plus_other_same_day,
       COUNTIF(IFNULL(gh.c, 1) = 1) / COUNT(*) AS share_zero_other_same_day,
       COUNTIF(IFNULL(gh.c, 1) BETWEEN 2 AND 5) / COUNT(*) AS share_1to4_other_same_day
FROM `polyrag-demo-ca.mv_coadopt.stars` s
LEFT JOIN gh USING (actor, day)
GROUP BY s.repo
