-- Starter queries for the photo-tagger telemetry dataset.
--
-- Run these against the Cloudflare Analytics Engine SQL API (or paste into the dashboard's SQL
-- console). All of these (including quantileWeighted, toStartOfInterval, and COUNT(DISTINCT
-- index1)) are verified to run on the live dataset. AE speaks a subset of the ClickHouse SQL
-- dialect, so consult the AE SQL reference if you extend them.
--
-- Every row counts SUM(_sample_interval) rather than COUNT(): once volume is high enough that AE
-- starts sampling, _sample_interval is the per-row weight (it is 1 while no sampling happens), so
-- weighting keeps the totals honest. The same reason is why the medians use quantileWeighted.
--
-- Column map (see worker.js):
--   index1  = install_id          blob1  = app_version    blob2  = interface (cli|gui)
--   blob3   = provider            blob4  = model          blob5  = arch
--   blob6   = os                  blob7  = os_release     blob8  = python_version
--   blob9   = output_language     blob10 = ui_language    blob11 = file_types (e.g. "cr3,jpg")
--   double1 = schema_version      double2 = batch_size    double3 = duration_seconds


-- 1. Most-used models (histogram), last 30 days.
SELECT blob4 AS model, SUM(_sample_interval) AS runs
FROM photo_tagger_telemetry
WHERE timestamp > NOW() - INTERVAL '30' DAY
GROUP BY model
ORDER BY runs DESC;


-- 2. Typical batch size: the median number of photos per run.
SELECT quantileWeighted(0.5)(double2, _sample_interval) AS median_batch_size
FROM photo_tagger_telemetry
WHERE timestamp > NOW() - INTERVAL '30' DAY;


-- 3. CLI vs GUI split.
SELECT blob2 AS interface, SUM(_sample_interval) AS runs
FROM photo_tagger_telemetry
WHERE timestamp > NOW() - INTERVAL '30' DAY
GROUP BY interface
ORDER BY runs DESC;


-- 4. Performance by CPU architecture: median run duration per arch.
-- Restricted to interface='cli' because a CLI run's duration is the batch's wall time, while a GUI
-- session's duration includes idle review time and is not comparable.
SELECT
  blob5 AS arch,
  quantileWeighted(0.5)(double3, _sample_interval) AS median_seconds,
  SUM(_sample_interval) AS runs
FROM photo_tagger_telemetry
WHERE timestamp > NOW() - INTERVAL '30' DAY AND blob2 = 'cli'
GROUP BY arch
ORDER BY runs DESC;


-- 5. OS distribution.
SELECT blob6 AS os, SUM(_sample_interval) AS runs
FROM photo_tagger_telemetry
WHERE timestamp > NOW() - INTERVAL '30' DAY
GROUP BY os
ORDER BY runs DESC;


-- 6a. Active installs (distinct install ids) over the last 30 days.
SELECT COUNT(DISTINCT index1) AS active_installs
FROM photo_tagger_telemetry
WHERE timestamp > NOW() - INTERVAL '30' DAY;

-- 6b. Active installs per day, for a trend line over the last 90 days.
SELECT
  toStartOfInterval(timestamp, INTERVAL '1' DAY) AS day,
  COUNT(DISTINCT index1) AS active_installs
FROM photo_tagger_telemetry
WHERE timestamp > NOW() - INTERVAL '90' DAY
GROUP BY day
ORDER BY day;


-- 7. File-type combinations per run. blob11 is the sorted set of extensions in one batch (e.g.
-- "cr3,jpg"), so this groups by the whole combination. AE SQL has no splitByChar to explode it into
-- one row per format; the dashboard does that split client-side. This still shows RAW vs JPEG usage.
SELECT blob11 AS file_types, SUM(_sample_interval) AS runs
FROM photo_tagger_telemetry
WHERE timestamp > NOW() - INTERVAL '30' DAY AND blob11 != ''
GROUP BY file_types
ORDER BY runs DESC;


-- 8. Metadata (output) language vs UI language.
SELECT blob9 AS output_language, blob10 AS ui_language, SUM(_sample_interval) AS runs
FROM photo_tagger_telemetry
WHERE timestamp > NOW() - INTERVAL '30' DAY
GROUP BY output_language, ui_language
ORDER BY runs DESC;
