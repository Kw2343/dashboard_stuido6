from __future__ import annotations
 
import pandas as pd
import plotly.express as px
import streamlit as st
 
from utils import style_bar_chart
 
 
def show_products_tab(
    filtered_reviews: pd.DataFrame,
    products_lookup: pd.DataFrame,
    products: pd.DataFrame,
) -> None:
 
    st.markdown("### Product exploration")
 
    product_counts = (
        filtered_reviews.groupby("parent_asin", as_index=False)
        .size()
        .rename(columns={"size": "filtered_review_count"})
        .merge(products_lookup, on="parent_asin", how="left")
        .sort_values(["filtered_review_count", "average_rating"], ascending=[False, False])
    )
 
    top_n = st.slider("Top products to show", 10, 100, 25, key="top_products_n")
 
    chart_data = product_counts.head(top_n).sort_values("filtered_review_count").copy()
    chart_data["chart_title"] = chart_data.get("display_title", chart_data["title"])
 
    fig = px.bar(
        chart_data,
        x="filtered_review_count", y="chart_title", orientation="h",
        hover_data=["parent_asin", "store_clean", "average_rating", "price", "title"],
        title=f"Top {top_n} products by filtered review count",
    )
    fig.update_layout(height=420, yaxis_title="Product")
    fig.update_yaxes(tickfont=dict(size=10))
    st.plotly_chart(style_bar_chart(fig), use_container_width=True)
 
    # ── Metadata coverage ─────────────────────────────────────────────────────
    completeness = pd.DataFrame({
        "Field":    ["Price", "Description", "Features", "Store", "Categories"],
        "Coverage": [
            products["has_price"].mean(),
            products["has_description"].mean(),
            products["has_features"].mean(),
            products["has_store"].mean(),
            products["has_categories"].mean(),
        ],
    })
    fig = px.bar(completeness, x="Field", y="Coverage", title="Metadata coverage in products file")
    fig.update_layout(height=400, yaxis_tickformat=".0%")
    st.plotly_chart(style_bar_chart(fig), use_container_width=True)
 
    # ── Search ────────────────────────────────────────────────────────────────
    st.markdown("#### Search products")
    query = st.text_input("Search by product title or store")
 
    table = product_counts[
        product_counts["store_clean"].notna()
        & (product_counts["store_clean"] != "(missing store)")
    ].copy()
 
    if query.strip():
        q = query.strip().lower()
        mask = (
            table["title"].fillna("").str.lower().str.contains(q, na=False)
            | table["store_clean"].fillna("").str.lower().str.contains(q, na=False)
        )
        table = table[mask]
 
    st.dataframe(table.head(250), use_container_width=True)