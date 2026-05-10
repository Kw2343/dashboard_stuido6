import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


def show_overview_tab(
    reviews,
    products,
    users,
    filtered_reviews,
    human_int,
    style_bar_chart,
    section_header,
):

    section_header("Dataset snapshot")

    o1, o2, o3, o4 = st.columns(4)

    o1.metric("All reviews", human_int(len(reviews)))
    o2.metric("All products", human_int(products["parent_asin"].nunique()))
    o3.metric("All users", human_int(users["user_id"].nunique()))
    o4.metric(
        "Avg words per filtered review",
        f"{filtered_reviews['review_length_words'].mean():.1f}"
    )

    left, right = st.columns(2)

    with left:
        yearly = (
            filtered_reviews.groupby("review_year", as_index=False)
            .size()
            .rename(columns={"size": "reviews"})
        )

        fig = px.bar(
            yearly,
            x="review_year",
            y="reviews",
            title="Reviews per year"
        )

        fig.update_layout(height=420)

        fig = style_bar_chart(fig)

        st.plotly_chart(fig, use_container_width=True)

    with right:

        is_dark = st.session_state.get("dark_mode", False)

        bg = "#0f172a" if is_dark else "white"
        grid = "#334155" if is_dark else "#e5e7eb"
        font = "#ffffff" if is_dark else "#111827"

        fig = px.bar(
            yearly,
            x="review_year",
            y="reviews",
            title="Reviews per year"
        )

        fig.update_layout(
            height=420,
            plot_bgcolor=bg,
            paper_bgcolor=bg,
            font=dict(color=font),
            xaxis=dict(gridcolor=grid),
            yaxis=dict(gridcolor=grid),
        )

        fig = style_bar_chart(fig)

        st.plotly_chart(fig, use_container_width=True)

    left2, right2 = st.columns(2)

    with left2:

        reviews_per_user = filtered_reviews.groupby("user_id").size()

        user_bins = pd.cut(
            reviews_per_user,
            bins=[0, 1, 5, 10, 20, 50, np.inf],
            labels=["1", "2-5", "6-10", "11-20", "21-50", "51+"],
            include_lowest=True,
        )

        user_bin_counts = (
            user_bins.value_counts()
            .sort_index()
            .reset_index()
        )

        user_bin_counts.columns = ["reviews_range", "user_count"]

        user_bin_counts["percentage"] = (
            user_bin_counts["user_count"]
            / user_bin_counts["user_count"].sum()
            * 100
        ).round(1)

        user_bin_counts["text"] = (
            user_bin_counts["percentage"].astype(str) + "%"
        )

        fig = px.bar(
            user_bin_counts,
            x="reviews_range",
            y="user_count",
            title="Reviews written per user",
            text="text"
        )

        fig.update_traces(textposition="outside")

        fig.update_layout(
            height=420,
            xaxis_title="Reviews per user",
            yaxis_title="Number of users"
        )

        fig = style_bar_chart(fig)

        st.plotly_chart(fig, use_container_width=True)

    with right2:

        reviews_per_product = filtered_reviews.groupby("parent_asin").size()

        product_bins = pd.cut(
            reviews_per_product,
            bins=[0, 1, 5, 10, 20, 50, np.inf],
            labels=["1", "2-5", "6-10", "11-20", "21-50", "51+"],
            include_lowest=True,
        )

        product_bin_counts = (
            product_bins.value_counts()
            .sort_index()
            .reset_index()
        )

        product_bin_counts.columns = [
            "reviews_range",
            "product_count"
        ]

        product_bin_counts["percentage"] = (
            product_bin_counts["product_count"]
            / product_bin_counts["product_count"].sum()
            * 100
        ).round(1)

        product_bin_counts["text"] = (
            product_bin_counts["percentage"].astype(str) + "%"
        )

        fig = px.bar(
            product_bin_counts,
            x="reviews_range",
            y="product_count",
            title="Reviews received per product",
            text="text"
        )

        fig.update_traces(textposition="outside")

        fig.update_layout(
            height=420,
            xaxis_title="Reviews per product",
            yaxis_title="Number of products"
        )

        fig = style_bar_chart(fig)

        st.plotly_chart(fig, use_container_width=True)