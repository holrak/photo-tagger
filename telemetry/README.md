# photo-tagger telemetry collector

This directory is the **entire** server side of photo-tagger's anonymous usage telemetry: a small
Cloudflare Worker that records one Analytics Engine data point per run. There is no third-party
analytics service. It is kept in the repo so anyone can read exactly what the collector does.

What the client sends, when, and how to turn it off is documented in the
[Telemetry section of the main README](../README.md#telemetry). The client side lives in
[`src/photo_tagger/telemetry.py`](../src/photo_tagger/telemetry.py).

## Files

| File                             | Purpose                                                          |
| -------------------------------- | ---------------------------------------------------------------- |
| [`worker.js`](worker.js)         | The Worker: validates a POST beacon, writes one AE data point.   |
| [`wrangler.toml`](wrangler.toml) | Deploy config (Worker name, AE binding, custom domain route).    |
| [`queries.sql`](queries.sql)     | Starter SQL answering each question the telemetry exists to ask. |
| [`dashboard.py`](dashboard.py)   | A marimo dashboard that runs those queries and charts them.      |

## What it stores

One data point per run, holding only the fields in
[`build_payload`](../src/photo_tagger/telemetry.py): app version, interface (cli/gui), provider,
model, CPU arch, OS, OS release, Python version, batch size, run duration, and a random install id.

It does **not** store IP addresses, set cookies, or read anything back; the endpoint is write-only
and replies `204 No Content`. Strings are length-clamped so a malformed client cannot bloat a point.

## Deploy

Prerequisites: a Cloudflare account, `tagger.photo` as a zone on it, and
[Wrangler](https://developers.cloudflare.com/workers/wrangler/) installed.

```bash
cd telemetry
export CLOUDFLARE_ACCOUNT_ID=...   # or run `wrangler login`
wrangler deploy
```

Wrangler creates the `photo_tagger_telemetry` Analytics Engine dataset on first write and provisions
the DNS record and TLS certificate for `telemetry.tagger.photo`.

## Query

Use the
[Analytics Engine SQL API](https://developers.cloudflare.com/analytics/analytics-engine/sql-api/)
(or the Cloudflare dashboard SQL console) with the statements in [`queries.sql`](queries.sql). They
cover most-used models, median batch size, CLI vs GUI split, duration by CPU arch, OS distribution,
and active installs over time.

## Dashboard

[`dashboard.py`](dashboard.py) is a [marimo](https://marimo.io) notebook that runs those queries and
turns them into KPI tiles, a daily active-installs trend, and ranked bar charts (models, interface,
OS, arch, and app/Python versions), plus a free-form SQL console. It reads the SQL API directly and
never writes.

It is a self-contained [PEP 723](https://peps.python.org/pep-0723/) script, so
[uv](https://docs.astral.sh/uv/) resolves its dependencies into a throwaway venv. It needs two
environment variables: the same `CLOUDFLARE_ACCOUNT_ID` as the deploy step, and a
[Cloudflare API token](https://dash.cloudflare.com/profile/api-tokens) scoped to **Account
Analytics: Read** (read-only, nothing else):

```bash
export CLOUDFLARE_ACCOUNT_ID=...
export CLOUDFLARE_API_TOKEN=...
uvx marimo run --sandbox telemetry/dashboard.py   # read-only app; use `marimo edit` to tinker
```

If the variables are missing the dashboard opens to setup instructions instead of failing. A window
control (last 7/30/90 days) scopes every chart, and each chart has a table view so no value is gated
behind color or a hover.
