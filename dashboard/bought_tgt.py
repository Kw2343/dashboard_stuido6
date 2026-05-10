import pandas as pd
import plotly.express as px
import streamlit as st
from pathlib import Path



def show_bought_together_chart(products_lookup):
    st.header("Top Products Bought Together")

    # ---------- LOAD DATA ----------
    try:
        BASE_DIR = Path(__file__).resolve().parent
        file_path = BASE_DIR / "data" / "products_bought_together_pair_counts.xlsx"

        pairs_df = pd.read_excel(file_path)
    except Exception as e:
        st.error(f"Error loading Excel file: {e}")
        return

    # ---------- CLEAN COLUMN NAMES ----------
    pairs_df.columns = [c.strip().lower() for c in pairs_df.columns]

    # ---------- STANDARDIZE COLUMN NAMES ----------
    pairs_df = pairs_df.rename(columns={
        "parent_asin_a": "parent_asin_1",
        "parent_asin_b": "parent_asin_2",
        "frequency": "count",
        "pair_count": "count"
    })

    required_cols = {"parent_asin_1", "parent_asin_2", "count"}
    if not required_cols.issubset(set(pairs_df.columns)):
        st.error(f"Missing required columns. Found: {pairs_df.columns}")
        return

    # ---------- TOP 10 ----------
    top_pairs = pairs_df.sort_values("count", ascending=False).head(10).copy()

    # ---------- MERGE PRODUCT TITLES ----------
    products_small = products_lookup[["parent_asin", "title"]].copy()

    top_pairs = top_pairs.merge(
        products_small,
        left_on="parent_asin_1",
        right_on="parent_asin",
        how="left"
    ).rename(columns={"title": "Product_1"}).drop(columns=["parent_asin"])

    top_pairs = top_pairs.merge(
        products_small,
        left_on="parent_asin_2",
        right_on="parent_asin",
        how="left"
    ).rename(columns={"title": "Product_2"}).drop(columns=["parent_asin"])

    # ---------- LABEL ----------
    top_pairs["Pair"] = (
        top_pairs["Product_1"].fillna(top_pairs["parent_asin_1"]) +
        " + " +
        top_pairs["Product_2"].fillna(top_pairs["parent_asin_2"])
    )

    # ---------- SHORT LABEL FUNCTION ----------
    def shorten(text, max_len=60):
        if pd.isna(text):
            return ""
        return text if len(text) <= max_len else text[:max_len].rsplit(" ", 1)[0] + "..."

    top_pairs["Pair_short"] = top_pairs["Pair"].apply(shorten)

    # ---------- CHART ----------
    fig = px.bar(
        top_pairs.sort_values("count"),
        x="count",
        y="Pair_short",
        orientation="h",
        title="Top 10 Frequently Bought Together",
        hover_data={
            "Pair": True,          # full text on hover
            "Pair_short": False,
            "count": True
        }
    )

    fig.update_layout(
        height=500,
        yaxis_title="",
        xaxis_title="Count"
    )
   

    st.plotly_chart(fig, use_container_width=True)

    # ---------- TABLE ----------
    st.subheader("Details")
    st.dataframe(
        top_pairs[["Pair", "count"]],
        use_container_width=True,
        hide_index=True
    )