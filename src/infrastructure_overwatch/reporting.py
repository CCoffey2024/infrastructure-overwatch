"""Analyst-facing reporting: a Seaborn "report card" static figure for
briefing documents, and a Plotly interactive dashboard for live monitoring --
both built from the same alerts/events tables `pipeline.PipelineResult`
produces (`PipelineResult.alerts_frame()` / `.events_frame()`).

Adapted from a general Seaborn/Plotly technique reference and re-pointed at
this project's alert/detection/event data instead of unrelated data. Requires
the `dataviz` extra (`pip install -e ".[dataviz]"`); lazy-imported so the
rest of the project has no hard dependency on matplotlib/seaborn/plotly.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

TRIAGE_BAND_ORDER = ("auto_confirm", "analyst_review", "auto_discard")
SEVERITY_ORDER = ("high", "medium", "low")


def _counts_by(series: pd.Series, order: tuple[str, ...] | None = None) -> pd.Series:
    """Value counts as a Series, reindexed to a fixed category order when
    given (missing categories show as 0 rather than being dropped) --
    matters for a report card where a stable panel layout across runs is
    more useful than an order that shuffles with the data."""
    counts = series.value_counts()
    if order is not None:
        counts = counts.reindex(order, fill_value=0)
    return counts


def build_report_card(alerts_df: pd.DataFrame, events_df: pd.DataFrame, out_path: str | Path | None = None):
    """A single static PNG-able figure: detections by threat class, triage-
    band breakdown, confidence distribution, and events by severity -- the
    four numbers an analyst supervisor would want in a shift-handoff report.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    fig.suptitle("Infrastructure Overwatch — Report Card", fontsize=14, fontweight="bold")

    ax = axes[0, 0]
    if len(alerts_df):
        counts = _counts_by(alerts_df["label"])
        sns.barplot(x=counts.index, y=counts.values, hue=counts.index, palette="Blues_d", legend=False, ax=ax)
    ax.set_title("Detections by threat class")
    ax.set_xlabel("")
    ax.set_ylabel("count")
    ax.tick_params(axis="x", rotation=30)

    ax = axes[0, 1]
    if len(alerts_df) and "triage_band" in alerts_df and alerts_df["triage_band"].notna().any():
        counts = _counts_by(alerts_df["triage_band"], order=TRIAGE_BAND_ORDER)
        colors = {"auto_confirm": "#54A24B", "analyst_review": "#F58518", "auto_discard": "#B0B0B0"}
        sns.barplot(x=counts.index, y=counts.values, hue=counts.index, palette=colors, legend=False, ax=ax)
        ax.set_title("Alert triage bands")
    else:
        ax.set_title("Alert triage bands (uncalibrated run)")
    ax.set_xlabel("")
    ax.set_ylabel("count")

    ax = axes[1, 0]
    if len(alerts_df):
        sns.histplot(alerts_df["confidence"], bins=15, kde=False, color="#4C78A8", ax=ax)
    ax.set_title("Raw detector confidence distribution")
    ax.set_xlabel("confidence")
    ax.set_xlim(0, 1)

    ax = axes[1, 1]
    if len(events_df):
        counts = _counts_by(events_df["severity"], order=SEVERITY_ORDER)
        colors = {"high": "#E45756", "medium": "#F58518", "low": "#54A24B"}
        sns.barplot(x=counts.index, y=counts.values, hue=counts.index, palette=colors, legend=False, ax=ax)
    ax.set_title("Events by severity")
    ax.set_xlabel("")
    ax.set_ylabel("count")

    plt.tight_layout()
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
    return fig


ALERTS_TABLE_COLUMNS = ["frame_id", "label", "confidence", "calibrated_confidence", "triage_band", "track_id"]
EVENTS_TABLE_COLUMNS = ["frame_id", "timestamp_s", "event_type", "severity", "track_id", "score", "description"]


def _add_table(fig, df: pd.DataFrame, columns: list[str], header_color: str, row: int, max_rows: int):
    """Adds a `go.Table` trace for whichever of `columns` are actually present in `df` --
    an alerts/events table from a minimal or synthetic-test frame shouldn't raise just
    because it's missing a column a full pipeline run would always populate."""
    import plotly.graph_objects as go

    present = [c for c in columns if c in df.columns]
    if not len(df) or not present:
        return
    shown = df[present].head(max_rows)
    fig.add_trace(
        go.Table(
            header=dict(values=present, fill_color=header_color, font=dict(color="white"), align="left"),
            cells=dict(values=[shown[c] for c in present], align="left"),
        ),
        row=row,
        col=1,
    )


def build_dashboard(
    alerts_df: pd.DataFrame,
    events_df: pd.DataFrame,
    out_path: str | Path | None = None,
    gif_path: str | Path | None = None,
    max_table_rows: int = 300,
):
    """The same four summary panels as `build_report_card`, as an interactive Plotly
    dashboard instead of a static image, plus an alerts table and an events table (an
    operator wants to see the actual rows, not just their distribution) -- for a live
    monitoring view rather than a printed briefing.

    `gif_path`, if given, is embedded above the charts as an `<img>` element pointing at an
    already-rendered annotated sequence (see `viz.build_annotated_gif`) -- this function
    only ever reads `alerts_df`/`events_df`, it never touches pixels itself.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=4,
        cols=2,
        specs=[
            [{"type": "xy"}, {"type": "xy"}],
            [{"type": "xy"}, {"type": "xy"}],
            [{"type": "table", "colspan": 2}, None],
            [{"type": "table", "colspan": 2}, None],
        ],
        row_heights=[0.18, 0.18, 0.32, 0.32],
        vertical_spacing=0.06,
        subplot_titles=(
            "Detections by threat class",
            "Alert triage bands",
            "Raw detector confidence distribution",
            "Events by severity",
            "Alerts",
            "Events",
        ),
    )

    if len(alerts_df):
        counts = _counts_by(alerts_df["label"])
        fig.add_trace(
            go.Bar(x=counts.index, y=counts.values, marker_color="#4C78A8", name="threat class"), row=1, col=1
        )

    if len(alerts_df) and "triage_band" in alerts_df and alerts_df["triage_band"].notna().any():
        counts = _counts_by(alerts_df["triage_band"], order=TRIAGE_BAND_ORDER)
        colors = ["#54A24B", "#F58518", "#B0B0B0"]
        fig.add_trace(go.Bar(x=counts.index, y=counts.values, marker_color=colors, name="triage band"), row=1, col=2)

    if len(alerts_df):
        fig.add_trace(
            go.Histogram(x=alerts_df["confidence"], nbinsx=15, marker_color="#4C78A8", name="confidence"), row=2, col=1
        )

    if len(events_df):
        counts = _counts_by(events_df["severity"], order=SEVERITY_ORDER)
        colors = ["#E45756", "#F58518", "#54A24B"]
        fig.add_trace(go.Bar(x=counts.index, y=counts.values, marker_color=colors, name="severity"), row=2, col=2)

    _add_table(fig, alerts_df, ALERTS_TABLE_COLUMNS, "#4C78A8", row=3, max_rows=max_table_rows)
    _add_table(fig, events_df, EVENTS_TABLE_COLUMNS, "#E45756", row=4, max_rows=max_table_rows)

    fig.update_layout(
        title_text="Infrastructure Overwatch — Live Dashboard",
        showlegend=False,
        height=1500,
        width=1000,
    )

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if gif_path is None:
            fig.write_html(str(out_path))
        else:
            _write_dashboard_with_gif(fig, out_path, Path(gif_path))
    return fig


def _write_dashboard_with_gif(fig, out_path: Path, gif_path: Path) -> None:
    """`fig.write_html` produces a full standalone page on its own, so embedding an
    `<img>` tag alongside it means building the page around `fig.to_html(full_html=False)`
    instead. The GIF is referenced by a path relative to `out_path`'s own directory, so the
    dashboard still finds it if both files are moved together."""
    plot_html = fig.to_html(full_html=False, include_plotlyjs="cdn")
    try:
        gif_src = gif_path.relative_to(out_path.parent).as_posix()
    except ValueError:
        gif_src = gif_path.as_posix()

    html = f"""<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><title>Infrastructure Overwatch Dashboard</title></head>
<body style="font-family: sans-serif; margin: 24px;">
<h1>Infrastructure Overwatch — Live Dashboard</h1>
<h2>Annotated sequence</h2>
<img src="{gif_src}" alt="Annotated sequence" style="max-width: 100%; background: #000;">
{plot_html}
</body>
</html>"""
    out_path.write_text(html, encoding="utf-8")
