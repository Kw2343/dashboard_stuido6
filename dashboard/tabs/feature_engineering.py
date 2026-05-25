from __future__ import annotations
import pandas as pd
import plotly.express as px
import streamlit as st
from feature_engineering import feature_engineering, product_feature_importance
from utils import style_bar_chart, human_int

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
    st.markdown("## Feature Engineering Overview")
    st.markdown("### Product Metadata Completeness")

    importance_df = product_feature_importance(products)
    
    for _, row in importance_df.iterrows():
        _progress_row(row["feature"], row["value"])
    
    # ── Dynamic Metadata Completeness Insights ────────────────────────────────
    avg_completeness = importance_df["value"].mean()
    lowest = importance_df.sort_values("value").iloc[0]
    highest = importance_df.sort_values("value").iloc[-1]
    
    critical_features = importance_df[importance_df["value"] < 70]
    excellent_features = importance_df[importance_df["value"] >= 90]
    
    total_products = len(products)
    lowest_missing = int(total_products * (100 - lowest["value"]) / 100)
    
    if avg_completeness >= 90:
        completeness_level = "excellent"
        
    elif avg_completeness >= 75:
        completeness_level = "strong"
        
    elif avg_completeness >= 60:
        completeness_level = "moderate"
        
    else:
        completeness_level = "poor"
        
    
    insights = [
        f"**Overall Completeness:** {avg_completeness:.1f}%",
        f"**Best Feature:** {highest['feature']} at {highest['value']:.1f}%",
        f"**Weakest Feature:** {lowest['feature']} at {lowest['value']:.1f}% ({human_int(lowest_missing)} products missing)",
    ]
    
    if len(excellent_features) > 0:
        insights.append(f"**{len(excellent_features)}** features exceed 90% completeness")
    if len(critical_features) > 0:
        insights.append(f" **{len(critical_features)}** features below 70% threshold")
    
    interpretations = []
    
    if avg_completeness >= 85:
        interpretations.append(f"**{completeness_level.capitalize()} metadata quality**  - Strong foundation for features")
    elif avg_completeness >= 70:
        interpretations.append(f"**{completeness_level.capitalize()} metadata quality**  - Acceptable but improvable")
    else:
        interpretations.append(f"**{completeness_level.capitalize()} metadata quality**  - Critical gaps limiting feature engineering")
    
    if lowest["value"] < 50:
        interpretations.append(f"**Critical gap in {lowest['feature']}** - Over half of products missing this data")
    
    interpretations.append("Complete metadata enables better scoring, recommendations, and search relevance")
    
    recommendations = []
    
    if lowest["value"] < 70:
        recommendations.append(f" **Priority:** Improve {lowest['feature']} from {lowest['value']:.0f}% (add to {human_int(lowest_missing)} products)")
    
    for _, row in critical_features.iterrows():
        missing = int(total_products * (100 - row["value"]) / 100)
        recommendations.append(f" **Fix:** {row['feature']} coverage ({row['value']:.0f}% → target 85%+)")
    
    if avg_completeness < 80:
        recommendations.append(" **Data enrichment campaign** - Systematically fill metadata gaps")
    
    if len(excellent_features) > 0:
        recommendations.append(f" **Maintain standards** - {len(excellent_features)} features show best practices")

    st.markdown(f"""
**Insights** 
{chr(10).join(f"- {i}" for i in insights)}

**Interpretation**
{chr(10).join(f"- {i}" for i in interpretations)}

**Recommendations**
{chr(10).join(f"- {r}" for r in recommendations)}
    """)

    st.divider()
    st.header("Engineered Product Features")

    c1, c2, c3, c4 = st.columns(4)
    avg_score = features_df['score'].mean()
    avg_freq = features_df['purchase_frequency'].mean()
    avg_users = features_df['unique_users'].mean()
    avg_rating = features_df['avg_rating'].mean()
    
    c1.metric("Avg Popularity Score",   f"{avg_score:.2f}")
    c2.metric("Avg Purchase Frequency", f"{avg_freq:.1f}")
    c3.metric("Avg Unique Users",       f"{avg_users:.1f}")
    c4.metric("Avg Product Rating",     f"{avg_rating:.2f}")
    
    # ── Dynamic Feature Quality Insights ──────────────────────────────────────
    score_std = features_df["score"].std()
    high_score_products = (features_df["score"] >= 4.0).sum()
    low_score_products = (features_df["score"] < 2.0).sum()
    total_feature_products = len(features_df)
    
    high_score_pct = (high_score_products / total_feature_products * 100) if total_feature_products > 0 else 0
    low_score_pct = (low_score_products / total_feature_products * 100) if total_feature_products > 0 else 0
    
    # Rating quality
    high_rated = (features_df["avg_rating"] >= 4.5).sum()
    low_rated = (features_df["avg_rating"] < 3.5).sum()
    high_rated_pct = (high_rated / total_feature_products * 100) if total_feature_products > 0 else 0
    
    # Engagement metrics
    high_engagement = (features_df["unique_users"] >= 100).sum()
    low_engagement = (features_df["unique_users"] < 10).sum()
    low_engagement_pct = (low_engagement / total_feature_products * 100) if total_feature_products > 0 else 0
    
    if avg_score >= 3.5:
        portfolio_quality = "strong"
      
    elif avg_score >= 2.5:
        portfolio_quality = "moderate"
      
    else:
        portfolio_quality = "weak"
        
    
    insights = [
        f"**Average Popularity Score:** {avg_score:.2f} / 5.00",
        f"**Score Distribution:** Std Dev = {score_std:.2f}",
        f"**High Performers (≥4.0):** {high_score_products} ({high_score_pct:.1f}%)",
        f"**Low Performers (<2.0):** {low_score_products} ({low_score_pct:.1f}%)",
        f"**Well-Rated Products (≥4.5⭐):** {high_rated} ({high_rated_pct:.1f}%)",
        f"**Low Engagement (<10 users):** {low_engagement} ({low_engagement_pct:.1f}%)"
    ]
    
    interpretations = []
    
    if avg_score >= 3.5:
        interpretations.append(f"**{portfolio_quality.capitalize()} portfolio**  - Overall product performance is solid")
    elif avg_score >= 2.5:
        interpretations.append(f"**{portfolio_quality.capitalize()} portfolio**  - Room for improvement exists")
    else:
        interpretations.append(f"**{portfolio_quality.capitalize()} portfolio**  - Significant quality issues")
    
    if high_score_pct >= 30:
        interpretations.append(f"**Strong top-tier** - {high_score_pct:.0f}% of products are high performers")
    elif high_score_pct < 15:
        interpretations.append(f" **Limited winners** - Only {high_score_pct:.0f}% are high performers")
    
    if low_score_pct > 20:
        interpretations.append(f" **Quality crisis** - {low_score_pct:.0f}% of products severely underperform")
    
    if score_std > 1.0:
        interpretations.append("**High variance** - Wide performance gap between best and worst products")
    else:
        interpretations.append("**Consistent performance** - Products perform at similar levels")
    
    if high_rated_pct < 30:
        interpretations.append(" **Rating concerns** - Few products achieve excellence (4.5+ stars)")
    
    recommendations = []
    
    if low_score_pct > 15:
        recommendations.append(f" **Urgent action:** Review and improve {low_score_products} low-scoring products")
        recommendations.append(" **Consider removal:** Discontinue products that can't be improved")
    
    if avg_score < 3.0:
        recommendations.append(" **Portfolio optimization** - Focus on quality over quantity")
        recommendations.append(" **Root cause analysis** - Identify why products score poorly")
    
    if high_score_pct < 20:
        recommendations.append(" **Develop winners** - Invest in creating more high-performing products")
    else:
        recommendations.append(f" **Leverage success** - Promote your {high_score_products} high-scoring products")
    
    if low_engagement_pct > 30:
        recommendations.append(f" **Marketing gap:** {low_engagement_pct:.0f}% of products have <10 users - visibility issue")
        recommendations.append(" **Promotion needed:** Launch campaigns for low-engagement products")
    
    if high_rated_pct >= 25:
        recommendations.append(" **Quality messaging:** Highlight your highly-rated products in marketing")
    else:
        recommendations.append(" **Quality improvement:** Work to increase number of 4.5+ rated products")
    
    recommendations.append(" **Benchmarking:** Use top performers as templates for new products")
    recommendations.append(" **Continuous monitoring:** Track score changes over time to measure improvements")

    st.markdown(f"""
**Insights** 
{chr(10).join(f"- {i}" for i in insights)}

**Interpretation**
{chr(10).join(f"- {i}" for i in interpretations)}

**Recommendations**
{chr(10).join(f"- {r}" for r in recommendations)}
    """)

    st.divider()
    st.subheader("Top Products by Popularity Score")

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
    
    # ── Dynamic Top Products Insights ─────────────────────────────────────────
    top_product = top.iloc[-1]  # Highest score (last after sorting)
    second_product = top.iloc[-2] if len(top) > 1 else None
    
    score_gap = top_product["score"] - second_product["score"] if second_product is not None else 0
    avg_top_score = top["score"].mean()
    
    # Analyze what drives success
    top_avg_rating = top_product.get("avg_rating", 0)
    top_users = top_product.get("unique_users", 0)
    top_frequency = top_product.get("purchase_frequency", 0)
    
    insights = [
        f"**#1 Product:** {top_product['title'][:60]}",
        f"**Score:** {top_product['score']:.2f} / 5.00"
    ]
    
    if pd.notna(top_avg_rating):
        insights.append(f"**Rating:** {top_avg_rating:.2f}⭐")
    if pd.notna(top_users):
        insights.append(f"**Unique Users:** {human_int(top_users)}")
    if pd.notna(top_frequency):
        insights.append(f"**Purchase Frequency:** {top_frequency:.1f}")
    
    if second_product is not None:
        insights.append(f"**Gap to #2:** {score_gap:.2f} points")
    
    insights.append(f"**Average (Top 20):** {avg_top_score:.2f}")
    
    interpretations = []
    
    if top_product["score"] >= 4.5:
        interpretations.append("**Exceptional performer** - This product excels across all dimensions")
    elif top_product["score"] >= 4.0:
        interpretations.append("**Strong performer** - Well-balanced rating, engagement, and purchase behavior")
    else:
        interpretations.append("**Moderate performer** - Top of catalog but room for improvement")
    
    # Identify success drivers
    if pd.notna(top_avg_rating) and top_avg_rating >= 4.5 and pd.notna(top_users) and top_users >= 100:
        interpretations.append("**Quality + Volume winner** - High ratings with broad appeal")
    elif pd.notna(top_avg_rating) and top_avg_rating >= 4.5:
        interpretations.append("**Quality-driven** - Excellent ratings are main success factor")
    elif pd.notna(top_users) and top_users >= 200:
        interpretations.append("**Engagement-driven** - Large user base drives popularity")
    
    if score_gap > 0.5:
        interpretations.append("**Clear leader** - Significantly outperforms #2")
    else:
        interpretations.append("**Competitive top tier** - Multiple products perform similarly")
    
    recommendations = []
    
    if top_product["score"] >= 4.0:
        recommendations.append(f" **Hero product:** Feature '{top_product['title'][:40]}' as flagship offering")
        recommendations.append(" **Premium placement:** Homepage, email headers, category featured spots")
    
    if pd.notna(top_avg_rating) and top_avg_rating >= 4.5:
        recommendations.append(" **Social proof:** Prominently display rating in all touchpoints")
    
    recommendations.append(" **Benchmark analysis:** Study what makes this product succeed")
    recommendations.append(" **Cross-sell:** Bundle with complementary products to boost revenue")
    recommendations.append(" **Replicate success:** Apply winning patterns to other products")
    recommendations.append(" **Protect performance:** Monitor closely and maintain quality standards")

    st.markdown(f"""
**Insights**
{chr(10).join(f"- {i}" for i in insights)}

**Interpretation**
{chr(10).join(f"- {i}" for i in interpretations)}

**Recommendations**
{chr(10).join(f"- {r}" for r in recommendations)}
    """)

    st.subheader("Most Recently Purchased Products")
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
    
    # ── Dynamic Recency Insights ──────────────────────────────────────────────
    recent_product = recency.iloc[0]
    days_ago = recent_product["days_since_last_purchase"]
    
    # Calculate trending metrics
    very_recent = (recency["days_since_last_purchase"] <= 7).sum()
    recent_count = (recency["days_since_last_purchase"] <= 30).sum()
    avg_recency = recency["days_since_last_purchase"].mean()
    
    if days_ago <= 1:
        recency_status = "today/yesterday"
       
    elif days_ago <= 7:
        recency_status = "within last week"
        
    elif days_ago <= 30:
        recency_status = "within last month"
    
    else:
        recency_status = "over a month ago"
    
    
    insights = [
        f"**Most Recent Purchase:** {recent_product['title'][:60]}",
        f"**Last Purchased:** {days_ago:.0f} days ago ({recency_status}) ",
        f"**Purchased in Last Week:** {very_recent} products",
        f"**Purchased in Last Month:** {recent_count} products",
        f"**Average Recency (Top 30):** {avg_recency:.0f} days"
    ]
    
    interpretations = []
    
    if days_ago <= 7:
        interpretations.append("**Active demand** - Recent purchases indicate current market interest")
    elif days_ago <= 30:
        interpretations.append("**Recent activity** - Products have recent but not immediate demand")
    else:
        interpretations.append(" **Stale recency** - Most recent purchase is quite old")
    
    if very_recent >= 10:
        interpretations.append(f"**Strong current activity** - {very_recent} products purchased in last week")
    elif very_recent < 5:
        interpretations.append(" **Low recent activity** - Few products purchased in last week")
    
    if avg_recency <= 30:
        interpretations.append("**Fresh catalog** - Top products have recent purchase activity")
    else:
        interpretations.append(" **Aging activity** - Recent purchases are not very recent on average")
    
    recommendations = []
    
    if days_ago <= 7:
        recommendations.append(f" **Promote trending:** Feature '{recent_product['title'][:40]}' as 'Hot Right Now'")
        recommendations.append(" **Strike while hot:** Push this product in email/social campaigns immediately")
    elif days_ago <= 30:
        recommendations.append(f" **Capitalize on interest:** Promote '{recent_product['title'][:40]}' as recently popular")
    else:
        recommendations.append(" **Refresh needed:** Consider why no very recent purchases exist")
    
    if very_recent >= 8:
        recommendations.append(f" **'Trending Now' section:** Showcase {very_recent} recently purchased products")
    
    recommendations.append(" **Real-time marketing:** Use recency data for dynamic homepage updates")
    recommendations.append(" **Momentum tracking:** Monitor purchase recency to catch emerging trends")
    recommendations.append(" **Urgency messaging:** Use 'Recently purchased by others' to drive conversions")
    
    if avg_recency > 45:
        recommendations.append(" **Activity boost needed:** Recent purchase lag is concerning - launch promotions")
    
    recommendations.append(" **Restock alerts:** Ensure recently popular items have adequate inventory")

    st.markdown(f"""
**Insights** 
{chr(10).join(f"- {i}" for i in insights)}

**Interpretation**
{chr(10).join(f"- {i}" for i in interpretations)}

**Recommendations**
{chr(10).join(f"- {r}" for r in recommendations)}
    """)

    st.subheader("Engineered Features Table")
    st.dataframe(top[[
        "parent_asin","title","purchase_count","unique_users",
        "avg_rating","purchase_frequency","days_since_last_purchase","score",
    ]], use_container_width=True)