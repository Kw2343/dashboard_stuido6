from __future__ import annotations
import pandas as pd
import plotly.express as px
import streamlit as st
from feature_engineering import feature_engineering, product_feature_importance
from utils import style_bar_chart

def _progress_row(label: str, value: float) -> None:
    st.markdown(f"""
    <div style="margin-bottom:14px;">
        <div style="font-weight:600;margin-bottom:6px;">{label} — {value:.1f}%</div>
        <div style="width:100%;height:14px;background:#e5e7eb;border-radius:999px;overflow:hidden;">
            <div style="width:{value}%;height:100%;background:#3b82f6;border-radius:999px;"></div>
        </div>
    </div>""", unsafe_allow_html=True)

def show_feature_tab(
    reviews: pd.DataFrame,
    products: pd.DataFrame,
    features_df: pd.DataFrame,
) -> None:
    st.markdown("## 🧠 Feature Engineering Overview")
    st.markdown("### Product Metadata Completeness")

    for _, row in product_feature_importance(products).iterrows():
        _progress_row(row["feature"], row["value"])

    st.divider()
    st.header("🧠 Engineered Product Features")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Avg Popularity Score",   f"{features_df['score'].mean():.2f}")
    c2.metric("Avg Purchase Frequency", f"{features_df['purchase_frequency'].mean():.1f}")
    c3.metric("Avg Unique Users",       f"{features_df['unique_users'].mean():.1f}")
    c4.metric("Avg Product Rating",     f"{features_df['avg_rating'].mean():.2f}")

    st.divider()
    st.subheader("🔥 Top Products by Popularity Score")

    top = (
        features_df.sort_values("score", ascending=False).head(20)
        .merge(products[["parent_asin", "title"]], on="parent_asin", how="left")
        .assign(short_title=lambda d: d["title"].fillna("Unknown").str[:50])
    )
    fig = px.bar(
        top.sort_values("score"), x="score", y="short_title", orientation="h",
        title="Top 20 Engineered Product Scores",
        hover_data=["purchase_count", "purchase_frequency", "avg_rating", "unique_users"],
    )
    fig.update_layout(height=600)
    st.plotly_chart(style_bar_chart(fig), use_container_width=True)

    st.subheader("⏳ Most Recently Purchased Products")
    recency = (
        features_df.sort_values("days_since_last_purchase").head(30)
        .merge(products[["parent_asin", "title"]], on="parent_asin", how="left")
        .assign(short_title=lambda d: d["title"].fillna("Unknown").str[:40])
    )
    fig2 = px.bar(
        recency.sort_values("days_since_last_purchase"),
        x="days_since_last_purchase", y="short_title", orientation="h",
        title="30 Most Recently Purchased Products",
    )
    fig2.update_layout(height=700)
    st.plotly_chart(style_bar_chart(fig2), use_container_width=True)

    st.subheader("📊 Engineered Features Table")
    st.dataframe(top[[
        "parent_asin","title","purchase_count","unique_users",
        "avg_rating","purchase_frequency","days_since_last_purchase","score",
    ]], use_container_width=True)