from __future__ import annotations
 
import pandas as pd
import plotly.express as px
import streamlit as st
 
from data_loader import load_bought_together
from utils import style_bar_chart, shorten
 
 
def show_bought_together_tab(products_lookup: pd.DataFrame) -> None:
    st.header("🛒 Top Products Bought Together")
 
    pairs_df = load_bought_together()
 
    if pairs_df is None:
        st.error("Could not load bought-together data. Check that the Excel file exists in the data folder.")
        return
 
    # ── Merge product titles ──────────────────────────────────────────────────
    titles = products_lookup[["parent_asin", "title"]].copy()
    top    = pairs_df.sort_values("count", ascending=False).head(10).copy()
 
    top = (
        top
        .merge(titles, left_on="parent_asin_1", right_on="parent_asin", how="left")
        .rename(columns={"title": "Product_1"}).drop(columns=["parent_asin"], errors="ignore")
        .merge(titles, left_on="parent_asin_2", right_on="parent_asin", how="left")
        .rename(columns={"title": "Product_2"}).drop(columns=["parent_asin"], errors="ignore")
    )
 
    top["Pair"] = (
        top["Product_1"].fillna(top["parent_asin_1"]) + " + " +
        top["Product_2"].fillna(top["parent_asin_2"])
    )
    top["Pair_short"] = top["Pair"].apply(shorten)
 
    # ── Chart ─────────────────────────────────────────────────────────────────
    fig = px.bar(
        top.sort_values("count"),
        x="count", y="Pair_short", orientation="h",
        title="Top 10 Frequently Bought Together",
        hover_data={"Pair": True, "Pair_short": False, "count": True},
    )
    fig.update_layout(height=500, yaxis_title="", xaxis_title="Co-purchase count")
    st.plotly_chart(style_bar_chart(fig), use_container_width=True)
 
    st.subheader("Details")
    st.dataframe(top[["Pair", "count"]], use_container_width=True, hide_index=True)
 