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


def build_dashboard(alerts_df: pd.DataFrame, events_df: pd.DataFrame, out_path: str | Path | None = None):
    """The same four panels as `build_report_card`, as an interactive Plotly
    dashboard instead of a static image -- for a live monitoring view rather
    than a printed briefing."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Detections by threat class",
            "Alert triage bands",
            "Raw detector confidence distribution",
            "Events by severity",
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

    fig.update_layout(
        title_text="Infrastructure Overwatch — Live Dashboard",
        showlegend=False,
        height=700,
        width=900,
    )

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(out_path))
    return fig
