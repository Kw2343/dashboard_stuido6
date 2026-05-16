from __future__ import annotations
 
from typing import Optional
 
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
 
from config import SCATTER_FILE, TOP_ORDER
from data_loader import load_scatter
 
 
# ── Chart builder ─────────────────────────────────────────────────────────────
 
def _build_scatter(df: pd.DataFrame) -> go.Figure:
    top  = df[df["Group"].isin(TOP_ORDER)].copy()
    near = df[df["Group"] == "Near"]
    far  = df[df["Group"] == "Far"]
    rand = df[df["Group"] == "Random"]
 
    top["_order"] = top["Group"].map({g: i for i, g in enumerate(TOP_ORDER)})
    top = top.sort_values("_order")
 
    fig = go.Figure()
 
    # background cloud
    fig.add_trace(go.Scatter(
        x=rand["MaxCosine"], y=rand["Predicted_Rating"],
        mode="markers", name="All",
        marker=dict(size=6, color="rgba(120,120,120,0.25)"),
        hoverinfo="skip",
    ))
 
    # near / far clusters
    fig.add_trace(go.Scatter(
        x=near["MaxCosine"], y=near["Predicted_Rating"],
        mode="markers", name="Near",
        marker=dict(size=10, color="green"),
    ))
    fig.add_trace(go.Scatter(
        x=far["MaxCosine"], y=far["Predicted_Rating"],
        mode="markers", name="Far",
        marker=dict(size=10, color="red"),
    ))
 
    # glow halo for top 5
    fig.add_trace(go.Scatter(
        x=top["MaxCosine"], y=top["Predicted_Rating"],
        mode="markers", name="Top glow",
        marker=dict(size=26, color="rgba(59,130,246,0.22)"),
        hoverinfo="skip", showlegend=False,
    ))
 
    # top 5 connected line + labels
    fig.add_trace(go.Scatter(
        x=top["MaxCosine"], y=top["Predicted_Rating"],
        mode="lines+markers+text",
        text=top["DisplayLabel"], textposition="top center",
        name="Top 5",
        line=dict(color="#3b82f6", width=3),
        marker=dict(size=14, color="#3b82f6"),
    ))
 
    fig.update_layout(
        title="Recommendation Scatter Plot",
        height=650,
        xaxis_title="Cosine Similarity",
        yaxis_title="Predicted Rating",
    )
    return fig
 
 
# ── Tab entry point ───────────────────────────────────────────────────────────
 
def show_scatter_tab() -> None:
    st.header("📊 Product Recommendation Scatter Plot")
 
    df: Optional[pd.DataFrame] = load_scatter()
 
    if df is None:
        st.warning(f"Scatter data not found at `{SCATTER_FILE}`. Please place the Excel file in the data folder.")
        return
 
    user_id = st.text_input("Search by User ID", placeholder="Enter User ID…")
 
    if not user_id.strip():
        st.info("Enter a User ID to view personalised recommendations.")
        return
 
    plot_df = df[df["User_ID"].astype(str) == user_id.strip()].copy()
 
    if plot_df.empty:
        st.warning(f"No data found for user **{user_id}**.")
        return
 
    # ── Top-5 table ───────────────────────────────────────────────────────────
    top = plot_df[plot_df["Group"].isin(TOP_ORDER)].copy()
    if not top.empty:
        top["_order"] = top["Group"].map({g: i for i, g in enumerate(TOP_ORDER)})
        top = top.sort_values("_order")
 
        st.subheader("Top 5 Product Recommendations")
        st.dataframe(
            top[["DisplayLabel", "MaxCosine", "Predicted_Rating"]]
            .rename(columns={
                "DisplayLabel":    "Product",
                "MaxCosine":       "Cosine Similarity",
                "Predicted_Rating": "Predicted Rating",
            })
            .assign(**{
                "Cosine Similarity": lambda d: d["Cosine Similarity"].round(3),
                "Predicted Rating":  lambda d: d["Predicted Rating"].round(2),
            }),
            use_container_width=True,
            hide_index=True,
            height=220,
        )
 
    st.markdown("<br>", unsafe_allow_html=True)
    st.plotly_chart(_build_scatter(plot_df), use_container_width=True)