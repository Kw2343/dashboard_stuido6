from __future__ import annotations
 
import pandas as pd
import plotly.express as px
import streamlit as st
 
from utils import style_bar_chart, human_int
 
 
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
    
    # ── Dynamic Product Insights ──────────────────────────────────────────────
    top_product = chart_data.iloc[-1]
    total_reviews = product_counts["filtered_review_count"].sum()
    top_product_reviews = top_product["filtered_review_count"]
    top_product_share = (top_product_reviews / total_reviews * 100) if total_reviews > 0 else 0
    
    # Calculate concentration metrics
    top_10_products = product_counts.head(10)
    top_10_reviews = top_10_products["filtered_review_count"].sum()
    top_10_concentration = (top_10_reviews / total_reviews * 100) if total_reviews > 0 else 0
    
    # Get product details
    avg_rating = top_product.get("average_rating", 0)
    price = top_product.get("price", 0)
    store = top_product.get("store_clean", "Unknown")
    
    # Calculate review distribution
    median_reviews = product_counts["filtered_review_count"].median()
    avg_reviews = product_counts["filtered_review_count"].mean()
    
    # Determine concentration level
    if top_10_concentration > 50:
        concentration = "high"
        diversity = "low"
    elif top_10_concentration > 30:
        concentration = "moderate"
        diversity = "moderate"
    else:
        concentration = "low"
        diversity = "high"
    
    insights = [
        f"Most reviewed: **{top_product['chart_title']}** with **{human_int(top_product_reviews)}** reviews (**{top_product_share:.1f}%** of all filtered reviews)"
    ]
    
    if pd.notna(avg_rating):
        insights.append(f"Rating: **{avg_rating:.2f}**⭐")
    if pd.notna(price) and price > 0:
        insights.append(f"Price: **${price:.2f}**")
    if pd.notna(store) and store != "(missing store)":
        insights.append(f"Seller: **{store}**")
    
    insights.append(f"Top 10 products capture **{top_10_concentration:.1f}%** of all reviews")
    insights.append(f"Median reviews per product: **{median_reviews:.0f}** | Average: **{avg_reviews:.1f}**")
    
    interpretations = []
    
    if top_product_share > 10:
        interpretations.append(f"**Dominant product** - single item drives {top_product_share:.1f}% of customer engagement")
    elif top_product_share > 5:
        interpretations.append("**Clear leader** - product has strong market presence")
    else:
        interpretations.append("**Balanced catalog** - reviews distributed across products")
    
    if concentration == "high":
        interpretations.append(f"**{concentration.capitalize()} concentration** - majority of engagement on few products (potential risk)")
    elif concentration == "moderate":
        interpretations.append(f"**{concentration.capitalize()} concentration** - engagement somewhat concentrated")
    else:
        interpretations.append(f"**{diversity.capitalize()} diversity** - reviews well-distributed across catalog")
    
    if pd.notna(avg_rating) and avg_rating >= 4.5:
        interpretations.append("**High customer satisfaction** - strong performance indicator")
    elif pd.notna(avg_rating) and avg_rating < 3.5:
        interpretations.append(" **Quality concerns** - low rating despite high engagement")
    
    recommendations = []
    
    if top_product_share > 15 or top_10_concentration > 60:
        recommendations.append(" **Reduce dependency** - over-reliance on few products is risky")
        recommendations.append(" **Diversify offerings** - promote mid-tier products to balance portfolio")
    
    if pd.notna(avg_rating) and avg_rating >= 4.0:
        recommendations.append(" **Feature prominently** - showcase this product on homepage and in campaigns")
        recommendations.append(" **Maximize visibility** - use in email marketing and social proof")
    else:
        recommendations.append(" **Quality review needed** - investigate why high volume doesn't match ratings")
    
    recommendations.append(" **Stock optimization** - ensure adequate inventory based on engagement levels")
    
    if concentration == "high":
        recommendations.append(" **Strategic promotion** - invest in marketing for products ranked 11-50")
        recommendations.append(" **Bundle opportunities** - create packages combining popular and lesser-known items")
    
    if median_reviews < 5:
        recommendations.append(" **Review generation** - many products lack social proof (median < 5 reviews)")

    st.markdown(f"""
**Insights**
{chr(10).join(f"- {i}" for i in insights)}

**Interpretation**
{chr(10).join(f"- {i}" for i in interpretations)}

**Recommendations**
{chr(10).join(f"- {r}" for r in recommendations)}
    """)
 
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
    
    # ── Dynamic Metadata Insights ─────────────────────────────────────────────
    lowest = completeness.sort_values("Coverage").iloc[0]
    highest = completeness.sort_values("Coverage").iloc[-1]
    avg_coverage = completeness["Coverage"].mean() * 100
    
    # Identify critical gaps (below 80%)
    critical_gaps = completeness[completeness["Coverage"] < 0.8]
    
    # Calculate impact
    total_products = len(products)
    lowest_missing = int(total_products * (1 - lowest["Coverage"]))
    
    insights = [
        f"Average metadata coverage: **{avg_coverage:.1f}%**",
        f"Best field: **{highest['Field']}** at **{highest['Coverage']*100:.1f}%**",
        f"Worst field: **{lowest['Field']}** at **{lowest['Coverage']*100:.1f}%** (**{human_int(lowest_missing)}** products missing data)"
    ]
    
    if len(critical_gaps) > 0:
        insights.append(f"**{len(critical_gaps)}** fields below 80% coverage threshold")
    
    interpretations = []
    
    if avg_coverage >= 90:
        interpretations.append("**Excellent metadata quality** - strong product information foundation")
    elif avg_coverage >= 75:
        interpretations.append("**Good metadata quality** - most products have complete information")
    elif avg_coverage >= 60:
        interpretations.append("**Moderate metadata quality** - significant gaps exist")
    else:
        interpretations.append(" **Poor metadata quality** - critical data gaps impacting customer trust")
    
    if lowest["Coverage"] < 0.5:
        interpretations.append(f"**Critical gap in {lowest['Field']}** - over half of products missing this data")
    elif lowest["Coverage"] < 0.7:
        interpretations.append(f"**Significant gap in {lowest['Field']}** - nearly a third of products incomplete")
    
    interpretations.append("Missing metadata reduces conversion rates, SEO performance, and customer confidence")
    
    recommendations = []
    
    if lowest["Coverage"] < 0.7:
        recommendations.append(f" **Priority fix: {lowest['Field']}** - fill missing data for **{human_int(lowest_missing)}** products")
    
    for _, row in critical_gaps.iterrows():
        missing = int(total_products * (1 - row["Coverage"]))
        recommendations.append(f" **Improve {row['Field']}** coverage from {row['Coverage']*100:.0f}% (add data for {human_int(missing)} products)")
    
    if avg_coverage < 80:
        recommendations.append(" **Data enrichment campaign** - systematically fill metadata gaps")
        recommendations.append(" **Vendor compliance** - require complete product info from suppliers")
    
    recommendations.append(" **SEO optimization** - complete metadata improves search visibility")
    recommendations.append(" **Quality standards** - establish minimum coverage thresholds (e.g., 90%+)")
    
    if highest["Coverage"] >= 0.95:
        recommendations.append(f" **Maintain excellence** - {highest['Field']} field shows best practices working")

    st.markdown(f"""
**Insights**
{chr(10).join(f"- {i}" for i in insights)}

**Interpretation**
{chr(10).join(f"- {i}" for i in interpretations)}

**Recommendations**
{chr(10).join(f"- {r}" for r in recommendations)}
    """)
 
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