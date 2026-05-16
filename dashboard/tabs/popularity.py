from __future__ import annotations
 
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
 
from utils import style_bar_chart, shorten
 
 
# ── Popularity score ──────────────────────────────────────────────────────────
 
def _compute_popularity(products: pd.DataFrame, m: int = 50) -> pd.DataFrame:
    df = products.copy()
    C  = df["average_rating"].mean()
 
    df["rating_number"]       = pd.to_numeric(df["rating_number"],       errors="coerce").fillna(0)
    df["purchase_frequency"]  = pd.to_numeric(df.get("purchase_frequency", 0), errors="coerce").fillna(0)
 
    v      = df["rating_number"]
    R      = df["average_rating"]
    f      = df["purchase_frequency"]
    f_norm = np.log1p(f) / (np.log1p(f.max() + 1) or 1)
 
    rating_score           = (v / (v + m)) * R + (m / (v + m)) * C
    df["popularity_score"] = (0.7 * rating_score) + (0.3 * f_norm * 5)
 
    return df.sort_values("popularity_score", ascending=False)
 
 
def _make_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
 
    def _label(row):
        title  = shorten(str(row["title"]), 40)
        rating = row.get("average_rating")
        store  = row.get("store_clean", "")
        if pd.notna(rating):
            title = f"{title} ⭐ {rating:.1f}"
        if pd.notna(store) and store not in ("", "(missing store)"):
            title = f"{title} ({store})"
        return title
 
    df["short_label"] = df.apply(_label, axis=1)
    return df
 
 
# ── Tab entry point ───────────────────────────────────────────────────────────
 
def show_popularity_tab(products: pd.DataFrame) -> None:
    st.header("🔥 Most Popular Products")
 
    pop_df = _make_labels(_compute_popularity(products))
 
    top_n = st.slider("Top N products", 10, 100, 20, key="popularity_top_n")
    top   = pop_df.head(top_n)
 
    fig = px.bar(
        top.sort_values("popularity_score"),
        x="popularity_score", y="short_label", orientation="h",
        hover_data=["average_rating", "rating_number", "purchase_frequency"],
        title=f"Top {top_n} Popular Products",
    )
    fig.update_layout(height=500, yaxis_title="")
    st.plotly_chart(style_bar_chart(fig), use_container_width=True)
 
    st.subheader("📊 Top Products Table")
    table_cols = ["parent_asin", "title", "average_rating",
                  "rating_number", "purchase_frequency", "popularity_score"]
    pop_table = top[[c for c in table_cols if c in top.columns]]
 
    st.data_editor(pop_table, use_container_width=True, hide_index=True, num_rows="fixed")
 
    csv = pop_table.to_csv(index=False).encode("utf-8")
    st.download_button(
        label=f"⬇ Download Top {top_n} as CSV",
        data=csv,
        file_name=f"top_{top_n}_popular_products.csv",
        mime="text/csv",
    )
 
    if st.button("Save Table to Server", type="primary"):
        pop_df.to_csv("popularity_table.csv", index=False)
        st.success("Saved to popularity_table.csv ✓")
 