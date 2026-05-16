from __future__ import annotations
 
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
 
 
# ── Number formatters ─────────────────────────────────────────────────────────
 
def human_int(x) -> str:
    if pd.isna(x):
        return "—"
    return f"{int(x):,}"
 
 
def pct(x) -> str:
    if pd.isna(x):
        return "—"
    return f"{x * 100:.1f}%"
 
 
# ── Chart styling ─────────────────────────────────────────────────────────────
 
def style_bar_chart(fig: go.Figure) -> go.Figure:
    """Apply consistent light/dark-aware styling to any Plotly figure."""
    is_dark = st.get_option("theme.base") == "dark"
 
    bg     = "#0f172a" if is_dark else "white"
    grid   = "#334155" if is_dark else "#e5e7eb"
    border = "#334155" if is_dark else "#cfcfcf"
    font   = "#ffffff" if is_dark else "#111827"
    legend_bg = "rgba(255,255,255,0.05)" if is_dark else "rgba(255,255,255,0.8)"
 
    fig.update_layout(
        plot_bgcolor=bg,
        paper_bgcolor=bg,
        font=dict(color=font),
        margin=dict(l=20, r=20, t=60, b=20),
        xaxis=dict(showgrid=True, gridcolor=grid, zeroline=False,
                   showline=True, linecolor=border),
        yaxis=dict(showgrid=True, gridcolor=grid, zeroline=False,
                   showline=True, linecolor=border),
        legend=dict(bgcolor=legend_bg, bordercolor=border, borderwidth=1),
        shapes=[dict(
            type="rect", xref="paper", yref="paper",
            x0=0, y0=0, x1=1, y1=1,
            line=dict(color=border, width=1.5),
            fillcolor="rgba(0,0,0,0)",
        )],
    )
    return fig
 
 
# ── UI helpers ────────────────────────────────────────────────────────────────
 
def section_header(title: str, subtitle: str = "") -> None:
    st.markdown(f"### {title}")
    if subtitle:
        st.caption(subtitle)
 
 
def shorten(text: str, max_len: int = 60) -> str:
    if not text or pd.isna(text):
        return ""
    s = str(text)
    return s if len(s) <= max_len else s[:max_len].rsplit(" ", 1)[0] + "…"
 
 
# ── Stats helpers ─────────────────────────────────────────────────────────────
 
def top_share(counts: pd.Series, frac: float) -> float:
    """Fraction of total in the top `frac` of entities."""
    if counts.empty:
        return np.nan
    n = max(1, int(np.ceil(len(counts) * frac)))
    return float(counts.sort_values(ascending=False).head(n).sum() / counts.sum())
 
 
def cumulative_share_curve(counts: pd.Series, entity_label: str) -> pd.DataFrame:
    s = counts.sort_values(ascending=False).reset_index(drop=True)
    if s.empty:
        return pd.DataFrame(columns=[f"{entity_label}_pct", "review_pct"])
    return pd.DataFrame({
        f"{entity_label}_pct": np.arange(1, len(s) + 1) / len(s),
        "review_pct":           s.cumsum() / s.sum(),
    })
 