# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "altair>=6.2.2",
#     "httpx>=0.27",
#     "marimo>=0.23",
#     "pandas>=2.3.3",
# ]
# ///
# mypy: ignore-errors
# The dependencies above live only in marimo's --sandbox venv, never in the project venv, so zuban
# cannot resolve them here (same escape hatch as gui.py). Altair 6 is required: 5.x fails to import
# on Python 3.14 (it declares `closed=True` TypedDicts that 3.14's stdlib typing does not accept).
"""
Marimo dashboard over the photo-tagger telemetry dataset.

Reads the Cloudflare Analytics Engine SQL API directly; nothing here can write. Each chart answers
one of the questions in queries.sql, plus release-adoption views, and a free-form SQL console.

Run it (read-only app view, dependencies resolved into an isolated venv):

    uvx marimo run --sandbox telemetry/dashboard.py

or open it as an editable notebook with `marimo edit` instead of `marimo run`. Requires
CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN (Account Analytics: Read) in the environment; see
telemetry/README.md.
"""

import marimo


__generated_with = "0.23.13"
app = marimo.App(width="full", app_title="photo-tagger telemetry")


@app.cell
def _(mo):
    mo.md(
        """
        # photo-tagger telemetry
        Anonymous usage beacons from the [collector](README.md), one data point per run.
        All totals weight rows by `_sample_interval`, so they stay honest once Analytics Engine
        starts sampling.
        """,
    )


@app.cell
def _():
    import os
    import re
    from contextlib import suppress

    import altair as alt
    import httpx
    import marimo as mo
    import pandas as pd

    return alt, httpx, mo, os, pd, re, suppress


@app.cell
def _(mo):
    # Validated categorical/ink palette, stepped per theme. Marks wear the hue; text and chrome
    # wear the ink tokens, never the series color.
    _dark = mo.app_meta().theme == "dark"
    COLORS = {
        "hue": "#3987e5" if _dark else "#2a78d6",
        "surface": "#1a1a19" if _dark else "#fcfcfb",
        "grid": "#2c2c2a" if _dark else "#e1e0d9",
        "baseline": "#383835" if _dark else "#c3c2b7",
        "ink": "#ffffff" if _dark else "#0b0b0b",
        "ink2": "#c3c2b7" if _dark else "#52514e",
        "muted": "#898781",
    }
    return (COLORS,)


@app.cell
def _(mo, os):
    account_id = os.environ.get("CLOUDFLARE_ACCOUNT_ID", "")
    api_token = os.environ.get("CLOUDFLARE_API_TOKEN", "")
    mo.stop(
        not account_id or not api_token,
        mo.md(
            """
            **Credentials missing.** The dashboard reads the Analytics Engine SQL API, which needs:

            1. `CLOUDFLARE_ACCOUNT_ID`: the account id shown on the Cloudflare dashboard overview.
            2. `CLOUDFLARE_API_TOKEN`: an [API token](https://dash.cloudflare.com/profile/api-tokens)
               with the **Account Analytics: Read** permission and nothing else.

            Export both and restart:

            ```bash
            export CLOUDFLARE_ACCOUNT_ID=...
            export CLOUDFLARE_API_TOKEN=...
            uvx marimo run --sandbox telemetry/dashboard.py
            ```
            """,
        ).callout(kind="warn"),
    )
    return account_id, api_token


@app.cell
def _(mo):
    window = mo.ui.dropdown(
        options={"Last 7 days": 7, "Last 30 days": 30, "Last 90 days": 90},
        value="Last 30 days",
        label="Window",
    )
    refresh = mo.ui.button(label="Refresh", value=0, on_click=lambda count: count + 1)
    mo.hstack([window, refresh], justify="start", align="center", gap=1)
    return refresh, window


@app.cell
def _(window):
    days = int(window.value)
    return (days,)


@app.cell
def _(account_id, api_token, httpx, pd, re, refresh, suppress):
    _ = refresh.value  # Depending on the button makes query re-run (and thus re-fetch) on click.

    def query(sql: str) -> pd.DataFrame:
        """Run one statement against the Analytics Engine SQL API and return the rows."""
        # Drop a trailing ';' (the queries.sql samples have one) so the appended clause stays
        # part of the same statement, then ask for JSON unless the caller already chose a format.
        sql = sql.strip().removesuffix(";")
        if not re.search(r"\bFORMAT\b", sql, flags=re.IGNORECASE):
            sql += "\nFORMAT JSON"
        response = httpx.post(
            f"https://api.cloudflare.com/client/v4/accounts/{account_id}/analytics_engine/sql",
            content=sql,
            headers={"Authorization": f"Bearer {api_token}"},
            timeout=30.0,
        )
        if response.status_code != httpx.codes.OK:
            msg = f"AE SQL API returned {response.status_code}: {response.text[:500]}"
            raise RuntimeError(msg)
        frame = pd.DataFrame(response.json().get("data", []))
        for column in frame.columns:
            # Numeric blobs (counts, durations) become numbers; text blobs stay strings.
            with suppress(ValueError, TypeError):
                frame[column] = pd.to_numeric(frame[column])
        return frame

    return (query,)


@app.cell
def _(pd, re):
    def fmt_int(value: float) -> str:
        return "n/a" if pd.isna(value) else f"{value:,.0f}"

    def fmt_secs(value: float) -> str:
        return "n/a" if pd.isna(value) else f"{value:,.1f} s"

    def version_key(text: str) -> tuple[int, ...]:
        """Sort key so '0.10.2' orders above '0.9.0' (string sort would not)."""
        return tuple(int(part) for part in re.findall(r"\d+", str(text))[:4]) or (0,)

    return fmt_int, fmt_secs, version_key


@app.cell
def _(COLORS, alt, mo, pd):
    def _themed(chart: alt.LayerChart, title: str, subtitle: str) -> alt.LayerChart:
        return (
            chart.properties(
                width="container",
                title=alt.TitleParams(
                    title,
                    # None fails Altair's schema validation; Undefined omits the subtitle instead.
                    subtitle=subtitle or alt.Undefined,
                    anchor="start",
                    color=COLORS["ink"],
                    subtitleColor=COLORS["muted"],
                    fontSize=13,
                    fontWeight=600,
                    subtitleFontSize=11,
                    offset=14,
                ),
                padding={"left": 4, "right": 48, "top": 4, "bottom": 4},
            )
            .configure(background="transparent", font="system-ui, sans-serif")
            .configure_view(stroke=None)
            .configure_axis(
                gridColor=COLORS["grid"],
                gridWidth=1,
                domainColor=COLORS["baseline"],
                tickColor=COLORS["baseline"],
                labelColor=COLORS["muted"],
                labelFontSize=11,
            )
        )

    def hbar(
        frame: pd.DataFrame,
        cat: str,
        val: str,
        *,
        title: str,
        subtitle: str = "",
        label: str | None = None,
        tooltips: list | None = None,
    ) -> alt.LayerChart:
        """
        Ranked horizontal bars in a single hue, magnitude read by length.

        Values are labeled at the tips, so the value axis and its grid can go; each bar has a
        4px rounded data-end and stays square at the baseline.
        """
        y = alt.Y(
            f"{cat}:N",
            sort=None,  # Keep the frame's order; the SQL or the caller already ranked it.
            title=None,
            axis=alt.Axis(ticks=False, domain=False, labelLimit=220, labelPadding=8),
        )
        x = alt.X(f"{val}:Q", axis=None)
        bars = (
            alt.Chart(frame)
            .mark_bar(size=18, cornerRadiusEnd=4, color=COLORS["hue"])
            .encode(
                y=y,
                x=x,
                tooltip=tooltips
                or [
                    alt.Tooltip(f"{cat}:N"),
                    alt.Tooltip(f"{val}:Q", format=",.0f"),
                ],
            )
        )
        # A pre-formatted label column rides as-is; otherwise format the raw value at the tip.
        text = alt.Text(f"{label}:N") if label else alt.Text(f"{val}:Q", format=",.0f")
        tips = (
            alt.Chart(frame)
            .mark_text(align="left", dx=6, color=COLORS["ink2"], fontSize=11)
            .encode(y=y, x=x, text=text)
        )
        return _themed((bars + tips).properties(height=alt.Step(30)), title, subtitle)

    def trend_line(frame: pd.DataFrame, *, title: str, subtitle: str = "") -> alt.LayerChart:
        """
        Single-series daily trend as a 2px line over a 10% area wash.

        A hover tooltip and end-dot mark the reading, and the last value is labeled (no legend,
        since the title already names the one series).
        """
        x = alt.X("day:T", title=None, axis=alt.Axis(grid=False, format="%b %d"))
        y = alt.Y(
            "active_installs:Q",
            title=None,
            axis=alt.Axis(domain=False, ticks=False, tickCount=4, tickMinStep=1, format=","),
        )
        base = alt.Chart(frame).encode(x=x, y=y)
        tooltip = [
            alt.Tooltip("day:T", title="Day", format="%b %d"),
            alt.Tooltip("active_installs:Q", title="Active installs", format=","),
        ]
        hover = alt.selection_point(fields=["day"], nearest=True, on="pointerover", empty=False)
        line = base.mark_line(
            strokeWidth=2,
            strokeCap="round",
            strokeJoin="round",
            color=COLORS["hue"],
        )
        wash = base.mark_area(opacity=0.1, color=COLORS["hue"])
        # Invisible wide targets carry the hover selection and the tooltip, so the reader never
        # has to land on the 2px line itself.
        targets = base.mark_point(size=400, opacity=0).encode(tooltip=tooltip).add_params(hover)
        dot = base.mark_point(
            filled=True,
            size=80,
            color=COLORS["hue"],
            stroke=COLORS["surface"],
            strokeWidth=2,
        ).encode(opacity=alt.condition(hover, alt.value(1), alt.value(0)))
        last = frame.tail(1)
        end_dot = (
            alt.Chart(last)
            .mark_point(
                filled=True,
                size=70,
                color=COLORS["hue"],
                stroke=COLORS["surface"],
                strokeWidth=2,
            )
            .encode(x=x, y=y)
        )
        end_label = (
            alt.Chart(last)
            .mark_text(align="left", dx=8, fontWeight=600, color=COLORS["ink2"], fontSize=11)
            .encode(x=x, y=y, text=alt.Text("active_installs:Q", format=","))
        )
        chart = (wash + line + targets + dot + end_dot + end_label).properties(height=260)
        return _themed(chart, title, subtitle)

    def chart_card(chart: alt.LayerChart, frame: pd.DataFrame) -> object:
        """Pair every chart with its table twin, so no value is gated behind color or hover."""
        return mo.ui.tabs({"Chart": chart, "Table": mo.ui.table(frame, page_size=10)})

    return chart_card, hbar, trend_line


@app.cell
def _(days, fmt_int, fmt_secs, mo, query):
    _totals = query(
        f"""
        SELECT
          COUNT(DISTINCT index1) AS installs,
          SUM(_sample_interval) AS runs,
          quantileWeighted(0.5)(double2, _sample_interval) AS median_batch
        FROM photo_tagger_telemetry
        WHERE timestamp > NOW() - INTERVAL '{days}' DAY
        """,
    ).iloc[0]
    _cli = query(
        f"""
        SELECT quantileWeighted(0.5)(double3, _sample_interval) AS median_seconds
        FROM photo_tagger_telemetry
        WHERE timestamp > NOW() - INTERVAL '{days}' DAY AND blob2 = 'cli'
        """,
    ).iloc[0]
    mo.hstack(
        [
            mo.stat(
                fmt_int(_totals["installs"]),
                label="Active installs",
                caption=f"last {days} days",
                bordered=True,
            ),
            mo.stat(
                fmt_int(_totals["runs"]),
                label="Runs",
                caption=f"weighted, last {days} days",
                bordered=True,
            ),
            mo.stat(
                fmt_int(_totals["median_batch"]),
                label="Median batch size",
                caption="photos per run",
                bordered=True,
            ),
            mo.stat(
                fmt_secs(_cli["median_seconds"]),
                label="Median run duration",
                caption="CLI runs only",
                bordered=True,
            ),
        ],
        widths="equal",
        gap=1,
        wrap=True,
    )


@app.cell
def _(chart_card, days, mo, pd, query, trend_line):
    installs_frame = query(
        f"""
        SELECT
          toStartOfInterval(timestamp, INTERVAL '1' DAY) AS day,
          COUNT(DISTINCT index1) AS active_installs
        FROM photo_tagger_telemetry
        WHERE timestamp > NOW() - INTERVAL '{days}' DAY
        GROUP BY day
        ORDER BY day
        """,
    )
    mo.stop(installs_frame.empty, mo.md("_No data points in this window yet._"))
    installs_frame["day"] = pd.to_datetime(installs_frame["day"])
    chart_card(
        trend_line(
            installs_frame,
            title="Active installs per day",
            subtitle="distinct install ids seen each day",
        ),
        installs_frame,
    )


@app.cell
def _(chart_card, days, hbar, mo, query):
    models_frame = query(
        f"""
        SELECT blob4 AS model, SUM(_sample_interval) AS runs
        FROM photo_tagger_telemetry
        WHERE timestamp > NOW() - INTERVAL '{days}' DAY
        GROUP BY model
        ORDER BY runs DESC
        LIMIT 12
        """,
    )
    mo.stop(models_frame.empty, mo.md("_No model data in this window._"))
    models_view = chart_card(
        hbar(
            models_frame,
            "model",
            "runs",
            title="Most-used models",
            subtitle=f"top 12 by weighted runs, last {days} days",
        ),
        models_frame,
    )
    return (models_view,)


@app.cell
def _(chart_card, days, hbar, mo, query):
    interface_frame = query(
        f"""
        SELECT blob2 AS interface, SUM(_sample_interval) AS runs
        FROM photo_tagger_telemetry
        WHERE timestamp > NOW() - INTERVAL '{days}' DAY
        GROUP BY interface
        ORDER BY runs DESC
        """,
    )
    mo.stop(interface_frame.empty, mo.md("_No interface data in this window._"))
    interface_frame["interface"] = interface_frame["interface"].str.upper()
    _total = interface_frame["runs"].sum()
    interface_frame["share"] = interface_frame["runs"] / _total if _total else 0.0
    interface_frame["label"] = interface_frame.apply(
        lambda row: f"{row['runs']:,.0f} ({row['share']:.0%})",
        axis=1,
    )
    interface_view = chart_card(
        hbar(
            interface_frame,
            "interface",
            "runs",
            title="CLI vs GUI",
            subtitle=f"weighted runs, last {days} days",
            label="label",
        ),
        interface_frame,
    )
    return (interface_view,)


@app.cell
def _(interface_view, mo, models_view):
    mo.hstack([models_view, interface_view], widths="equal", gap=1, wrap=True)


@app.cell
def _(chart_card, days, hbar, mo, query):
    os_frame = query(
        f"""
        SELECT blob6 AS os, SUM(_sample_interval) AS runs
        FROM photo_tagger_telemetry
        WHERE timestamp > NOW() - INTERVAL '{days}' DAY
        GROUP BY os
        ORDER BY runs DESC
        """,
    )
    mo.stop(os_frame.empty, mo.md("_No OS data in this window._"))
    os_view = chart_card(
        hbar(
            os_frame,
            "os",
            "runs",
            title="Operating systems",
            subtitle=f"weighted runs, last {days} days",
        ),
        os_frame,
    )
    return (os_view,)


@app.cell
def _(alt, chart_card, days, hbar, mo, query):
    arch_frame = query(
        f"""
        SELECT
          blob5 AS arch,
          quantileWeighted(0.5)(double3, _sample_interval) AS median_seconds,
          SUM(_sample_interval) AS runs
        FROM photo_tagger_telemetry
        WHERE timestamp > NOW() - INTERVAL '{days}' DAY AND blob2 = 'cli'
        GROUP BY arch
        ORDER BY runs DESC
        """,
    )
    mo.stop(arch_frame.empty, mo.md("_No CLI runs in this window._"))
    arch_frame["label"] = arch_frame["median_seconds"].map(lambda s: f"{s:,.1f} s")
    arch_view = chart_card(
        hbar(
            arch_frame,
            "arch",
            "median_seconds",
            title="Median run duration by CPU architecture",
            subtitle="CLI only: GUI durations include idle review time",
            label="label",
            tooltips=[
                alt.Tooltip("arch:N"),
                alt.Tooltip("median_seconds:Q", title="Median seconds", format=",.1f"),
                alt.Tooltip("runs:Q", title="Runs", format=",.0f"),
            ],
        ),
        arch_frame,
    )
    return (arch_view,)


@app.cell
def _(arch_view, mo, os_view):
    mo.hstack([os_view, arch_view], widths="equal", gap=1, wrap=True)


@app.cell
def _(chart_card, days, hbar, mo, query, version_key):
    app_version_frame = query(
        f"""
        SELECT blob1 AS app_version, SUM(_sample_interval) AS runs
        FROM photo_tagger_telemetry
        WHERE timestamp > NOW() - INTERVAL '{days}' DAY
        GROUP BY app_version
        ORDER BY runs DESC
        LIMIT 12
        """,
    ).sort_values("app_version", key=lambda s: s.map(version_key), ascending=False)
    mo.stop(app_version_frame.empty, mo.md("_No version data in this window._"))
    app_version_view = chart_card(
        hbar(
            app_version_frame,
            "app_version",
            "runs",
            title="photo-tagger versions",
            subtitle="newest first: is the latest release being picked up?",
        ),
        app_version_frame,
    )
    return (app_version_view,)


@app.cell
def _(chart_card, days, hbar, mo, query, version_key):
    python_version_frame = query(
        f"""
        SELECT blob8 AS python_version, SUM(_sample_interval) AS runs
        FROM photo_tagger_telemetry
        WHERE timestamp > NOW() - INTERVAL '{days}' DAY
        GROUP BY python_version
        ORDER BY runs DESC
        LIMIT 12
        """,
    ).sort_values("python_version", key=lambda s: s.map(version_key), ascending=False)
    mo.stop(python_version_frame.empty, mo.md("_No Python version data in this window._"))
    python_version_view = chart_card(
        hbar(
            python_version_frame,
            "python_version",
            "runs",
            title="Python versions",
            subtitle="newest first: informs when old interpreters can be dropped",
        ),
        python_version_frame,
    )
    return (python_version_view,)


@app.cell
def _(app_version_view, mo, python_version_view):
    mo.hstack([app_version_view, python_version_view], widths="equal", gap=1, wrap=True)


@app.cell
def _(chart_card, days, hbar, mo, query):
    _formats_raw = query(
        f"""
        SELECT blob11 AS file_types, SUM(_sample_interval) AS runs
        FROM photo_tagger_telemetry
        WHERE timestamp > NOW() - INTERVAL '{days}' DAY AND blob11 != ''
        GROUP BY file_types
        ORDER BY runs DESC
        """,
    )
    mo.stop(_formats_raw.empty, mo.md("_No file-type data in this window yet._"))
    # blob11 is a per-run set like "cr3,jpg"; AE SQL cannot split it, so explode client-side into
    # one row per format. runs then counts runs that used each format (a CR3+JPEG run hits both).
    formats_frame = (
        _formats_raw.assign(fmt=_formats_raw["file_types"].str.split(","))
        .explode("fmt")
        .groupby("fmt", as_index=False)["runs"]
        .sum()
        .sort_values("runs", ascending=False)
    )
    chart_card(
        hbar(
            formats_frame,
            "fmt",
            "runs",
            title="File formats",
            subtitle="runs that included each extension (a CR3+JPEG run counts toward both)",
        ),
        formats_frame,
    )


@app.cell
def _(chart_card, days, hbar, mo, query):
    output_language_frame = query(
        f"""
        SELECT blob9 AS output_language, SUM(_sample_interval) AS runs
        FROM photo_tagger_telemetry
        WHERE timestamp > NOW() - INTERVAL '{days}' DAY AND blob9 != ''
        GROUP BY output_language
        ORDER BY runs DESC
        LIMIT 12
        """,
    )
    mo.stop(output_language_frame.empty, mo.md("_No metadata-language data in this window yet._"))
    output_language_view = chart_card(
        hbar(
            output_language_frame,
            "output_language",
            "runs",
            title="Metadata languages",
            subtitle="language the model writes titles and keywords in",
        ),
        output_language_frame,
    )
    return (output_language_view,)


@app.cell
def _(chart_card, days, hbar, mo, query):
    ui_language_frame = query(
        f"""
        SELECT blob10 AS ui_language, SUM(_sample_interval) AS runs
        FROM photo_tagger_telemetry
        WHERE timestamp > NOW() - INTERVAL '{days}' DAY AND blob10 != ''
        GROUP BY ui_language
        ORDER BY runs DESC
        LIMIT 12
        """,
    )
    mo.stop(ui_language_frame.empty, mo.md("_No UI-language data in this window yet._"))
    ui_language_view = chart_card(
        hbar(
            ui_language_frame,
            "ui_language",
            "runs",
            title="UI languages",
            subtitle="the app interface language actually in effect",
        ),
        ui_language_frame,
    )
    return (ui_language_view,)


@app.cell
def _(mo, output_language_view, ui_language_view):
    mo.hstack([output_language_view, ui_language_view], widths="equal", gap=1, wrap=True)


@app.cell
def _(mo):
    _default_sql = """\
    -- Column map (see worker.js):
    --   index1  = install_id          blob1  = app_version    blob2  = interface (cli|gui)
    --   blob3   = provider            blob4  = model          blob5  = arch
    --   blob6   = os                  blob7  = os_release     blob8  = python_version
    --   blob9   = output_language     blob10 = ui_language    blob11 = file_types
    --   double1 = schema_version      double2 = batch_size    double3 = duration_seconds
    SELECT blob3 AS provider, SUM(_sample_interval) AS runs
    FROM photo_tagger_telemetry
    WHERE timestamp > NOW() - INTERVAL '30' DAY
    GROUP BY provider
    ORDER BY runs DESC;
    """
    sql_editor = mo.ui.code_editor(value=_default_sql, language="sql")
    run_sql = mo.ui.run_button(label="Run query")
    mo.vstack(
        [
            mo.md(
                "### SQL console\n"
                "Anything the [AE SQL dialect]("
                "https://developers.cloudflare.com/analytics/analytics-engine/sql-reference/"
                ") supports; `FORMAT JSON` is appended automatically. Starter statements "
                "live in `queries.sql`.",
            ),
            sql_editor,
            run_sql,
        ],
        gap=1,
    )
    return run_sql, sql_editor


@app.cell
def _(mo, query, run_sql, sql_editor):
    mo.stop(not run_sql.value, mo.md("_Edit the SQL above and click **Run query**._"))
    result_frame = query(sql_editor.value)
    mo.stop(result_frame.empty, mo.md("_The query returned no rows._"))
    mo.ui.table(result_frame, page_size=15)


@app.cell
def _(mo):
    mo.md(
        """
        ---
        Counts are `SUM(_sample_interval)` and medians are `quantileWeighted`, so numbers stay
        correct once Analytics Engine samples (the weight is 1 until then). The collector stores
        no IPs and no cookies; see [README.md](README.md) in this directory.
        """,
    )


if __name__ == "__main__":
    app.run()
