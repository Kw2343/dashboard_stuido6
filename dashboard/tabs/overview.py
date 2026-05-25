from __future__ import annotations
 
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

 
from utils import style_bar_chart, section_header, human_int
from insights import rating_insights, review_trend_insight
 
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
        st.markdown(review_trend_insight(yearly))
 
    # ── Rating distribution ───────────────────────────────────────────────────
    with right:
        rating_counts = (
            filtered_reviews["rating"]
            .value_counts()
            .sort_index()
            .reset_index()
            .rename(columns={"index": "rating", "rating": "count", "count": "count"})
        )
        rating_counts.columns = ["rating", "count"]
        fig = px.bar(rating_counts, x="rating", y="count", title="Rating distribution")
        fig.update_layout(height=420)
        st.plotly_chart(style_bar_chart(fig), use_container_width=True)
        st.markdown(rating_insights(filtered_reviews))
        
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

        # Dynamic user insights
        reviews_per_user = filtered_reviews.groupby("user_id").size()
        total_users = reviews_per_user.nunique()
        heavy_users = (reviews_per_user > 10).sum()
        heavy_users_pct = (reviews_per_user > 10).mean() * 100
        super_users = (reviews_per_user > 50).sum()
        avg_reviews_per_user = reviews_per_user.mean()
        median_reviews_per_user = reviews_per_user.median()
        
        # Calculate top contributor impact
        top_10_pct_users = int(total_users * 0.1)
        top_users_review_count = reviews_per_user.nlargest(top_10_pct_users).sum()
        top_users_contribution = (top_users_review_count / len(filtered_reviews)) * 100

        # Dynamic recommendation based on data
        if heavy_users_pct > 15:
            engagement_level = "high"
            recommendation = "Leverage your active user base with exclusive benefits, early access to new products, and community features"
        elif heavy_users_pct > 5:
            engagement_level = "moderate"
            recommendation = "Grow your power user community with gamification, leaderboards, and recognition programs"
        else:
            engagement_level = "low"
            recommendation = "Incentivize repeat reviews with rewards programs and simplified review processes"

        st.markdown(f"""
        **Insights**
        - **{heavy_users_pct:.1f}%** of users ({human_int(heavy_users)} users) write more than 10 reviews
        - **Top 10%** of users contribute **{top_users_contribution:.1f}%** of all reviews
        - Average: **{avg_reviews_per_user:.1f}** reviews per user | Median: **{median_reviews_per_user:.0f}**
        {f"- **{super_users}** super users have written 50+ reviews each" if super_users > 0 else ""}

        **Interpretation**
        - User engagement is **{engagement_level}** - {'a dedicated core drives most activity' if heavy_users_pct > 10 else 'most users are casual reviewers'}
        - {'Heavy concentration suggests dependency on few contributors' if top_users_contribution > 50 else 'Review distribution is relatively balanced across users'}

        **Recommendation**
        - {recommendation}
        - {'Diversify engagement to reduce dependency on power users' if top_users_contribution > 60 else 'Continue building a loyal reviewer community'}
        """)
 
    with right2:
        fig = _bin_chart(
            filtered_reviews.groupby("parent_asin").size(),
            "Reviews per product", "products", "Reviews received per product",
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Dynamic product insights
        reviews_per_product = filtered_reviews.groupby("parent_asin").size()
        total_products = reviews_per_product.nunique()
        well_reviewed_products = (reviews_per_product > 20).sum()
        well_reviewed_pct = (reviews_per_product > 20).mean() * 100
        lonely_products = (reviews_per_product == 1).sum()
        lonely_products_pct = (reviews_per_product == 1).mean() * 100
        avg_reviews_per_product = reviews_per_product.mean()
        median_reviews_per_product = reviews_per_product.median()
        
        # Calculate visibility distribution
        top_20_pct_products = int(total_products * 0.2)
        top_products_review_count = reviews_per_product.nlargest(top_20_pct_products).sum()
        top_products_visibility = (top_products_review_count / len(filtered_reviews)) * 100

        # Dynamic recommendation based on data
        if lonely_products_pct > 30:
            visibility_issue = "severe"
            recommendation = "Urgent: Implement review request campaigns and incentivize first reviews for new products"
        elif lonely_products_pct > 15:
            visibility_issue = "moderate"
            recommendation = "Focus on products with 0-5 reviews through targeted email campaigns and review prompts"
        else:
            visibility_issue = "low"
            recommendation = "Maintain current review generation strategies while optimizing for underperforming products"

        st.markdown(f"""
        **Insights**
        - **{well_reviewed_pct:.1f}%** of products ({human_int(well_reviewed_products)}) have 20+ reviews
        - **{lonely_products_pct:.1f}%** of products ({human_int(lonely_products)}) have only 1 review
        - Average: **{avg_reviews_per_product:.1f}** reviews per product | Median: **{median_reviews_per_product:.0f}**
        - **Top 20%** of products receive **{top_products_visibility:.1f}%** of all reviews

        **Interpretation**
        - Product visibility is **{visibility_issue}** - {'most products lack social proof' if lonely_products_pct > 20 else 'review distribution shows room for improvement'}
        - {'High concentration on few products limits catalog engagement' if top_products_visibility > 70 else 'Reviews are moderately distributed across catalog'}

        **Recommendation**
        - {recommendation}
        - {'Implement automated review requests post-purchase for low-visibility items' if lonely_products_pct > 20 else 'Balance review acquisition across product portfolio'}
        - Consider featuring less-reviewed products on homepage and in email campaigns
        """)