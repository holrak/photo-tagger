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
(or the dashboard SQL console) with the statements in [`queries.sql`](queries.sql). They cover
most-used models, median batch size, CLI vs GUI split, duration by CPU arch, OS distribution, and
active installs over time.
