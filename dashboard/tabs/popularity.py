from __future__ import annotations
 
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
 
from utils import style_bar_chart, shorten, human_int
 
 
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
    st.header("Most Popular Products")
 
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
    
    # ── Dynamic Popularity Insights ───────────────────────────────────────────
    top_product = top.iloc[0]  # Highest score (first after sorting desc)
    
    # Calculate distribution metrics
    score_range = top["popularity_score"].max() - top["popularity_score"].min()
    score_gap = top.iloc[0]["popularity_score"] - top.iloc[1]["popularity_score"] if len(top) > 1 else 0
    avg_score = top["popularity_score"].mean()
    median_score = top["popularity_score"].median()
    
    # Rating analysis
    avg_rating = top_product.get("average_rating", 0)
    rating_count = top_product.get("rating_number", 0)
    purchase_freq = top_product.get("purchase_frequency", 0)
    
    # Analyze score components
    high_rating_products = (top["average_rating"] >= 4.5).sum() if "average_rating" in top.columns else 0
    high_rating_pct = (high_rating_products / len(top) * 100) if len(top) > 0 else 0
    
    # Calculate concentration
    top_5_avg_score = top.head(5)["popularity_score"].mean() if len(top) >= 5 else avg_score
    score_concentration = (top_5_avg_score / avg_score - 1) * 100 if avg_score > 0 else 0
    
    insights = [
        f"**#1 Product:** {shorten(top_product['title'], 60)}",
        f"**Popularity Score:** {top_product['popularity_score']:.2f} / 5.00"
    ]
    
    if pd.notna(avg_rating):
        insights.append(f"**Rating:** {avg_rating:.2f}⭐ from {human_int(rating_count)} reviews")
    
    if pd.notna(purchase_freq) and purchase_freq > 0:
        insights.append(f"**Purchase Frequency:** {human_int(purchase_freq)}")
    
    if len(top) > 1:
        insights.append(f"**Gap to #2:** {score_gap:.2f} points ({(score_gap/top_product['popularity_score']*100):.1f}%)")
    
    insights.append(f"**Average score (Top {top_n}):** {avg_score:.2f} | **Median:** {median_score:.2f}")
    insights.append(f"**{high_rating_products}** products ({high_rating_pct:.0f}%) have 4.5+ star ratings")
    
    interpretations = []
    
    # Analyze dominance
    if score_gap > 0.5:
        interpretations.append("**Clear market leader** - #1 product significantly outperforms competition")
    elif score_gap > 0.2:
        interpretations.append("**Strong leader** - #1 product has solid advantage")
    else:
        interpretations.append("**Competitive market** - top products closely matched in popularity")
    
    # Analyze quality vs volume
    rating_weight = 0.7  # From algorithm
    if pd.notna(avg_rating) and avg_rating >= 4.5 and rating_count > 100:
        interpretations.append("**Quality + Volume winner** - combines excellent ratings with high purchase frequency")
    elif pd.notna(avg_rating) and avg_rating >= 4.5:
        interpretations.append("**Quality-driven success** - high ratings compensate for lower volume")
    elif rating_count > 200:
        interpretations.append("**Volume-driven popularity** - high purchase frequency drives score")
    
    # Overall market health
    if high_rating_pct >= 70:
        interpretations.append(f"**Strong product quality** - {high_rating_pct:.0f}% of top products are highly rated")
    elif high_rating_pct < 40:
        interpretations.append(f" **Quality concerns** - only {high_rating_pct:.0f}% of popular products have excellent ratings")
    
    if score_concentration > 20:
        interpretations.append("**Top-heavy distribution** - few products dominate popularity landscape")
    elif score_concentration < 10:
        interpretations.append("**Balanced competition** - popularity distributed evenly across products")
    
    recommendations = []
    
    # Top product recommendations
    if top_product["popularity_score"] >= 4.0:
        recommendations.append(f" **Hero product strategy** - make '{shorten(top_product['title'], 40)}' your flagship offering")
        recommendations.append(" **Premium placement** - feature on homepage, category pages, and email campaigns")
    else:
        recommendations.append(" **No standout winner** - consider product portfolio optimization")
    
    if pd.notna(avg_rating) and avg_rating >= 4.5:
        recommendations.append(" **Leverage social proof** - prominently display ratings in all marketing materials")
        recommendations.append(" **Customer testimonials** - extract and promote positive reviews")
    elif pd.notna(avg_rating) and avg_rating < 4.0:
        recommendations.append(" **Quality improvement needed** - investigate and address rating concerns")
    
    # Inventory and marketing
    if pd.notna(purchase_freq) and purchase_freq > 1000:
        recommendations.append(" **Inventory priority** - ensure consistent stock availability for high-demand items")
    
    recommendations.append(" **Cross-sell opportunities** - bundle top products with complementary items")
    
    # Portfolio strategy
    if score_concentration > 30:
        recommendations.append(" **Diversify portfolio** - reduce dependency on few products")
        recommendations.append(" **Promote mid-tier** - invest in products ranked 6-20 to balance catalog")
    
    if high_rating_pct < 50:
        recommendations.append(" **Quality audit** - review why popular products lack high ratings")
        recommendations.append(" **Customer feedback loop** - actively collect and address product concerns")
    
    # Growth opportunities
    recommendations.append(f" **Replicate success** - analyze what makes #{len(top)+1}-{len(top)+10} perform well")
    recommendations.append(" **New product development** - use popularity patterns to inform future offerings")

    st.markdown(f"""
**Insights**
{chr(10).join(f"- {i}" for i in insights)}

**Interpretation**
{chr(10).join(f"- {i}" for i in interpretations)}

**Recommendations**
{chr(10).join(f"- {r}" for r in recommendations)}
    """)
 
    st.subheader("Top Products Table")
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