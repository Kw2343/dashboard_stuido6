import pandas as pd
import plotly.express as px
import streamlit as st


def show_products_tab(
    filtered_reviews,
    products_lookup,
    products,
    style_bar_chart
):

    st.markdown("### Product exploration")

    product_counts = (
        filtered_reviews.groupby("parent_asin", as_index=False)
        .size()
        .rename(columns={"size": "filtered_review_count"})
        .merge(products_lookup, on="parent_asin", how="left")
        .sort_values(
            ["filtered_review_count", "average_rating"],
            ascending=[False, False]
        )
    )

    top_n = st.slider(
        "Top products to show",
        10,
        100,
        25,
        key="top_products_n"
    )

    chart_data = (
        product_counts.head(top_n)
        .sort_values("filtered_review_count")
        .copy()
    )

    chart_data["chart_title"] = (
        chart_data.get("display_title", chart_data["title"])
    )

    fig = px.bar(
        chart_data,
        x="filtered_review_count",
        y="chart_title",
        orientation="h",
        hover_data=[
            "parent_asin",
            "store_clean",
            "average_rating",
            "price",
            "title"
        ],
        title=f"Top {top_n} products by filtered review count",
    )

    fig.update_layout(
        height=420,
        yaxis_title="Product"
    )

    fig.update_yaxes(
        tickfont=dict(size=10)
    )

    fig = style_bar_chart(fig)

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # ---------- METADATA COVERAGE ----------
    completeness = pd.DataFrame(
        {
            "Field": [
                "Price",
                "Description",
                "Features",
                "Store",
                "Categories"
            ],
            "Coverage": [
                products["has_price"].mean(),
                products["has_description"].mean(),
                products["has_features"].mean(),
                products["has_store"].mean(),
                products["has_categories"].mean(),
            ],
        }
    )

    fig = px.bar(
        completeness,
        x="Field",
        y="Coverage",
        title="Metadata coverage in products file"
    )

    fig.update_layout(
        height=500,
        yaxis_tickformat=".0%"
    )

    fig = style_bar_chart(fig)

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # ---------- SEARCH ----------
    st.markdown("#### Search products")

    query = st.text_input(
        "Search by product title or store"
    )

    filtered_product_table = product_counts.copy()

    filtered_product_table = filtered_product_table[
        (
            filtered_product_table["store_clean"].notna()
        )
        &
        (
            filtered_product_table["store_clean"]
            != "(missing store)"
        )
    ]

    if query.strip():

        q = query.strip().lower()

        search_mask = (
            filtered_product_table["title"]
            .fillna("")
            .str.lower()
            .str.contains(q, na=False)
        ) | (
            filtered_product_table["store_clean"]
            .fillna("")
            .str.lower()
            .str.contains(q, na=False)
        )

        filtered_product_table = (
            filtered_product_table[search_mask]
        )

    st.dataframe(
        filtered_product_table.head(250),
        use_container_width=True
    )