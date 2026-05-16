from __future__ import annotations
 
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
 
from utils import style_bar_chart, section_header, human_int
 
 
def show_overview_tab(
    reviews: pd.DataFrame,
    products: pd.DataFrame,
    users: pd.DataFrame,
    filtered_reviews: pd.DataFrame,
) -> None:
 
    section_header("Dataset snapshot")
 
    o1, o2, o3, o4 = st.columns(4)
    o1.metric("All reviews",  human_int(len(reviews)))
    o2.metric("All products", human_int(products["parent_asin"].nunique()))
    o3.metric("All users",    human_int(users["user_id"].nunique()))
    o4.metric(
        "Avg words / filtered review",
        f"{filtered_reviews['review_length_words'].mean():.1f}",
    )
 
    # ── Reviews per year ─────────────────────────────────────────────────────
    yearly = (
        filtered_reviews.groupby("review_year", as_index=False)
        .size()
        .rename(columns={"size": "reviews"})
    )
 
    left, right = st.columns(2)
 
    with left:
        fig = px.bar(yearly, x="review_year", y="reviews", title="Reviews per year")
        fig.update_layout(height=420)
        st.plotly_chart(style_bar_chart(fig), use_container_width=True)
 
    # ── Rating distribution ───────────────────────────────────────────────────
    with right:
        rating_counts = (
            filtered_reviews["rating"]
            .value_counts()
            .sort_index()
            .reset_index()
            .rename(columns={"index": "rating", "rating": "count", "count": "count"})
        )
        # value_counts returns (rating, count) in newer pandas
        rating_counts.columns = ["rating", "count"]
        fig = px.bar(rating_counts, x="rating", y="count", title="Rating distribution")
        fig.update_layout(height=420)
        st.plotly_chart(style_bar_chart(fig), use_container_width=True)
 
    # ── User & product review-count distributions ─────────────────────────────
    bins   = [0, 1, 5, 10, 20, 50, np.inf]
    labels = ["1", "2–5", "6–10", "11–20", "21–50", "51+"]
 
    left2, right2 = st.columns(2)
 
    def _bin_chart(series: pd.Series, x_title: str, y_col: str, chart_title: str):
        binned = pd.cut(series, bins=bins, labels=labels, include_lowest=True)
        counts = binned.value_counts().sort_index().reset_index()
        counts.columns = ["reviews_range", y_col]
        counts["pct"] = (counts[y_col] / counts[y_col].sum() * 100).round(1).astype(str) + "%"
        fig = px.bar(counts, x="reviews_range", y=y_col, title=chart_title, text="pct")
        fig.update_traces(textposition="outside")
        fig.update_layout(height=420, xaxis_title=x_title, yaxis_title=f"Number of {y_col.split('_')[0]}s")
        return style_bar_chart(fig)
 
    with left2:
        fig = _bin_chart(
            filtered_reviews.groupby("user_id").size(),
            "Reviews per user", "users", "Reviews written per user",
        )
        st.plotly_chart(fig, use_container_width=True)
 
    with right2:
        fig = _bin_chart(
            filtered_reviews.groupby("parent_asin").size(),
            "Reviews per product", "products", "Reviews received per product",
        )
        st.plotly_chart(fig, use_container_width=True)
 