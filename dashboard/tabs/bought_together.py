from __future__ import annotations
 
import pandas as pd
import plotly.express as px
import streamlit as st
 
from data_loader import load_bought_together
from utils import style_bar_chart, shorten, human_int
 
 
def show_bought_together_tab(products_lookup: pd.DataFrame) -> None:
    st.header("Top Products Bought Together")
 
    pairs_df = load_bought_together()
 
    if pairs_df is None:
        st.error("Could not load bought-together data. Check that the Excel file exists in the data folder.")
        return
 
    # ── Merge product titles ──────────────────────────────────────────────────
    titles = products_lookup[["parent_asin", "title"]].copy()
    top    = pairs_df.sort_values("count", ascending=False).head(10).copy()
 
    top = (
        top
        .merge(titles, left_on="parent_asin_1", right_on="parent_asin", how="left")
        .rename(columns={"title": "Product_1"}).drop(columns=["parent_asin"], errors="ignore")
        .merge(titles, left_on="parent_asin_2", right_on="parent_asin", how="left")
        .rename(columns={"title": "Product_2"}).drop(columns=["parent_asin"], errors="ignore")
    )
 
    top["Pair"] = (
        top["Product_1"].fillna(top["parent_asin_1"]) + " + " +
        top["Product_2"].fillna(top["parent_asin_2"])
    )
    top["Pair_short"] = top["Pair"].apply(shorten)
 
    # ── Chart ─────────────────────────────────────────────────────────────────
    fig = px.bar(
        top.sort_values("count"),
        x="count", y="Pair_short", orientation="h",
        title="Top 10 Frequently Bought Together",
        hover_data={"Pair": True, "Pair_short": False, "count": True},
    )
    fig.update_layout(height=500, yaxis_title="", xaxis_title="Co-purchase count")
    st.plotly_chart(style_bar_chart(fig), use_container_width=True)
    
    # ── Dynamic Co-Purchase Insights ──────────────────────────────────────────
    top_pair = top.iloc[0]  # Highest count (first after sorting desc)
    
    # Calculate metrics
    total_pairs = len(pairs_df)
    total_copurchases = pairs_df["count"].sum()
    top_10_copurchases = top["count"].sum()
    top_10_share = (top_10_copurchases / total_copurchases * 100) if total_copurchases > 0 else 0
    
    avg_copurchase = top["count"].mean()
    median_copurchase = top["count"].median()
    
    # Analyze strength of associations
    strong_pairs = (top["count"] >= 50).sum()
    moderate_pairs = ((top["count"] >= 20) & (top["count"] < 50)).sum()
    weak_pairs = (top["count"] < 20).sum()
    
    # Calculate gap between top pairs
    count_gap = top.iloc[0]["count"] - top.iloc[1]["count"] if len(top) > 1 else 0
    count_range = top["count"].max() - top["count"].min()
    
    insights = [
        f"**Top Pair:** {shorten(top_pair['Product_1'], 35)} + {shorten(top_pair['Product_2'], 35)}",
        f"**Co-purchased:** {human_int(top_pair['count'])} times"
    ]
    
    if len(top) > 1:
        insights.append(f"**Gap to #2:** {human_int(count_gap)} co-purchases ({(count_gap/top_pair['count']*100):.1f}%)")
    
    insights.append(f"**Total unique pairs analyzed:** {human_int(total_pairs)}")
    insights.append(f"**Top 10 represent:** {top_10_share:.1f}% of all co-purchases")
    insights.append(f"**Average (Top 10):** {human_int(avg_copurchase)} | **Median:** {human_int(median_copurchase)}")
    
    if strong_pairs > 0:
        insights.append(f"**{strong_pairs}** strong associations (50+ co-purchases)")
    if moderate_pairs > 0:
        insights.append(f"**{moderate_pairs}** moderate associations (20-49 co-purchases)")
    
    interpretations = []
    
    # Analyze association strength
    if top_pair["count"] >= 100:
        interpretations.append("**Extremely strong pairing** - these products are heavily associated in customer minds")
    elif top_pair["count"] >= 50:
        interpretations.append("**Strong pairing** - clear purchase pattern between these products")
    elif top_pair["count"] >= 20:
        interpretations.append("**Moderate pairing** - noticeable but not dominant purchase pattern")
    else:
        interpretations.append("**Weak associations** - co-purchase patterns are not strongly defined")
    
    # Market dominance
    if count_gap / top_pair["count"] > 0.3:
        interpretations.append("**Dominant pair** - #1 significantly outperforms other combinations")
    elif count_range > 50:
        interpretations.append("**Varied association strength** - wide range in co-purchase frequency")
    else:
        interpretations.append("**Balanced patterns** - multiple product pairs show similar co-purchase rates")
    
    # Portfolio implications
    if top_10_share > 50:
        interpretations.append("**Concentrated behavior** - majority of co-purchases involve just 10 pairs")
    elif top_10_share > 30:
        interpretations.append("**Moderate concentration** - top pairs drive significant but not dominant share")
    else:
        interpretations.append("**Diverse behavior** - co-purchases spread across many product combinations")
    
    if strong_pairs >= 5:
        interpretations.append(f"**Rich bundling opportunities** - {strong_pairs} pairs show strong customer preferences")
    elif strong_pairs == 0:
        interpretations.append(" **Weak pairing signals** - no strong co-purchase patterns detected")
    
    recommendations = []
    
    # Bundling strategy
    if top_pair["count"] >= 50:
        recommendations.append(f" **Create bundle:** Package '{shorten(top_pair['Product_1'], 30)}' + '{shorten(top_pair['Product_2'], 30)}' together")
        recommendations.append(f" **Bundle discount:** Offer 10-15% off when purchased together ({human_int(top_pair['count'])} customers already do this)")
    
    if strong_pairs >= 3:
        recommendations.append(f" **Multiple bundles:** Create {strong_pairs} distinct product bundles based on strong pairs")
    elif strong_pairs == 0:
        recommendations.append(" **Bundle testing needed:** Current co-purchase patterns are weak - test suggested pairings")
    
    # Cross-selling recommendations
    recommendations.append(" **'Frequently bought together':** Display these pairs on product pages")
    recommendations.append(" **Email upsells:** Target customers who bought Product A with offers for Product B")
    
    if top_10_share < 30:
        recommendations.append(" **Expand analysis:** Low concentration suggests more pairing opportunities beyond top 10")
    
    # Inventory and placement
    if top_pair["count"] >= 50:
        recommendations.append(" **Co-location strategy:** Place paired products near each other in warehouse/store")
        recommendations.append(" **Joint inventory planning:** Forecast demand for pairs together to avoid stockouts")
    
    # Marketing opportunities
    recommendations.append(" **Cross-promotion:** Advertise Product B to customers viewing Product A")
    recommendations.append(" **Complete the set':** Market pairs as solutions or complementary items")
    
    if moderate_pairs + strong_pairs >= 5:
        recommendations.append(f" **Bundle campaign:** Launch marketing campaign around {moderate_pairs + strong_pairs} proven product combinations")
    
    # Analytics
    recommendations.append(" **Track bundle performance:** Monitor conversion rates when pairs are shown together")
    recommendations.append(" **A/B test pricing:** Experiment with bundle discounts to optimize revenue")
    
    if top_pair["count"] < 20:
        recommendations.append(" **Incentivize bundles:** Current patterns are weak - use discounts to strengthen associations")

    st.markdown(f"""
**Insights**
{chr(10).join(f"- {i}" for i in insights)}

**Interpretation**
{chr(10).join(f"- {i}" for i in interpretations)}

**Recommendations**
{chr(10).join(f"- {r}" for r in recommendations)}
    """)
 
    st.subheader("Details")
    st.dataframe(top[["Pair", "count"]], use_container_width=True, hide_index=True)