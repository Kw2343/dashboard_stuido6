from __future__ import annotations
 
from typing import Optional
 
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
 
from config import SCATTER_FILE, TOP_ORDER
from data_loader import load_scatter
from utils import human_int
 
 
# ── Chart builder ─────────────────────────────────────────────────────────────
 
def _build_scatter(df: pd.DataFrame) -> go.Figure:
    top  = df[df["Group"].isin(TOP_ORDER)].copy()
    near = df[df["Group"] == "Near"]
    far  = df[df["Group"] == "Far"]
    rand = df[df["Group"] == "Random"]
 
    top["_order"] = top["Group"].map({g: i for i, g in enumerate(TOP_ORDER)})
    top = top.sort_values("_order")
 
    fig = go.Figure()
 
    # background cloud
    fig.add_trace(go.Scatter(
        x=rand["MaxCosine"], y=rand["Predicted_Rating"],
        mode="markers", name="All",
        marker=dict(size=6, color="rgba(120,120,120,0.25)"),
        hoverinfo="skip",
    ))
 
    # near / far clusters
    fig.add_trace(go.Scatter(
        x=near["MaxCosine"], y=near["Predicted_Rating"],
        mode="markers", name="Near",
        marker=dict(size=10, color="green"),
    ))
    fig.add_trace(go.Scatter(
        x=far["MaxCosine"], y=far["Predicted_Rating"],
        mode="markers", name="Far",
        marker=dict(size=10, color="red"),
    ))
 
    # glow halo for top 5
    fig.add_trace(go.Scatter(
        x=top["MaxCosine"], y=top["Predicted_Rating"],
        mode="markers", name="Top glow",
        marker=dict(size=26, color="rgba(59,130,246,0.22)"),
        hoverinfo="skip", showlegend=False,
    ))
 
    # top 5 connected line + labels
    fig.add_trace(go.Scatter(
        x=top["MaxCosine"], y=top["Predicted_Rating"],
        mode="lines+markers+text",
        text=top["DisplayLabel"], textposition="top center",
        name="Top 5",
        line=dict(color="#3b82f6", width=3),
        marker=dict(size=14, color="#3b82f6"),
    ))
 
    fig.update_layout(
        title="Recommendation Scatter Plot",
        height=650,
        xaxis_title="Cosine Similarity",
        yaxis_title="Predicted Rating",
    )
    return fig
 
 
# ── Tab entry point ───────────────────────────────────────────────────────────
 
def show_scatter_tab() -> None:
    st.header("Product Recommendation Scatter Plot")
 
    df: Optional[pd.DataFrame] = load_scatter()
 
    if df is None:
        st.warning(f"Scatter data not found at `{SCATTER_FILE}`. Please place the Excel file in the data folder.")
        return
 
    user_id = st.text_input("Search by User ID", placeholder="Enter User ID…")
 
    if not user_id.strip():
        st.info("Enter a User ID to view personalised recommendations.")
        return
 
    plot_df = df[df["User_ID"].astype(str) == user_id.strip()].copy()
 
    if plot_df.empty:
        st.warning(f"No data found for user **{user_id}**.")
        return
 
    # ── Top-5 table ───────────────────────────────────────────────────────────
    top = plot_df[plot_df["Group"].isin(TOP_ORDER)].copy()
    if not top.empty:
        top["_order"] = top["Group"].map({g: i for i, g in enumerate(TOP_ORDER)})
        top = top.sort_values("_order")
 
        st.subheader("Top 5 Product Recommendations")
        st.dataframe(
            top[["DisplayLabel", "MaxCosine", "Predicted_Rating"]]
            .rename(columns={
                "DisplayLabel":    "Product",
                "MaxCosine":       "Cosine Similarity",
                "Predicted_Rating": "Predicted Rating",
            })
            .assign(**{
                "Cosine Similarity": lambda d: d["Cosine Similarity"].round(3),
                "Predicted Rating":  lambda d: d["Predicted Rating"].round(2),
            }),
            use_container_width=True,
            hide_index=True,
            height=220,
        )
        
        # ── Dynamic Top Recommendation Insights ───────────────────────────────
        best = top.iloc[0]
        worst_in_top5 = top.iloc[-1]
        
        # Calculate quality metrics
        avg_similarity_top5 = top["MaxCosine"].mean()
        avg_rating_top5 = top["Predicted_Rating"].mean()
        
        # Calculate score spread
        similarity_range = top["MaxCosine"].max() - top["MaxCosine"].min()
        rating_range = top["Predicted_Rating"].max() - top["Predicted_Rating"].min()
        
        # Quality assessment
        if best["MaxCosine"] >= 0.8:
            similarity_quality = "excellent"
        elif best["MaxCosine"] >= 0.6:
            similarity_quality = "strong"
        elif best["MaxCosine"] >= 0.4:
            similarity_quality = "moderate"
        else:
            similarity_quality = "weak"
        
        if best["Predicted_Rating"] >= 4.5:
            rating_quality = "excellent"
        elif best["Predicted_Rating"] >= 4.0:
            rating_quality = "strong"
        elif best["Predicted_Rating"] >= 3.5:
            rating_quality = "moderate"
        else:
            rating_quality = "poor"
        
        insights = [
            f"**#1 Recommendation:** {best['DisplayLabel']}",
            f"**Cosine Similarity:** {best['MaxCosine']:.3f} ({similarity_quality})",
            f"**Predicted Rating:** {best['Predicted_Rating']:.2f}⭐ ({rating_quality})",
            f"**Top 5 Average Similarity:** {avg_similarity_top5:.3f}",
            f"**Top 5 Average Predicted Rating:** {avg_rating_top5:.2f}⭐"
        ]
        
        if similarity_range > 0.2:
            insights.append(f"**Quality Variation:** Wide spread ({similarity_range:.2f}) between #1 and #5")
        
        interpretations = []
        
        # Overall quality assessment
        if best["MaxCosine"] >= 0.7 and best["Predicted_Rating"] >= 4.0:
            interpretations.append("**Excellent match** - High similarity AND high predicted satisfaction")
            interpretations.append("Strong confidence in recommendation quality")
        elif best["MaxCosine"] >= 0.5 and best["Predicted_Rating"] >= 3.5:
            interpretations.append("**Good match** - Reasonable similarity with acceptable predicted rating")
        elif best["MaxCosine"] < 0.4 or best["Predicted_Rating"] < 3.0:
            interpretations.append(" **Weak match** - Low confidence in recommendation quality")
            interpretations.append("Model may need more user data or feature tuning")
        
        # Consistency check
        if similarity_range < 0.15 and rating_range < 0.5:
            interpretations.append("**Consistent recommendations** - All top 5 are similarly strong")
        elif similarity_range > 0.3 or rating_range > 1.0:
            interpretations.append("**Variable quality** - Significant gap between best and worst recommendations")
        
        # Balance check
        if best["MaxCosine"] >= 0.7 and best["Predicted_Rating"] < 3.5:
            interpretations.append(" **Similarity-rating mismatch** - High similarity but low predicted enjoyment")
        elif best["MaxCosine"] < 0.5 and best["Predicted_Rating"] >= 4.0:
            interpretations.append(" **Interesting pattern** - Lower similarity but high predicted rating (diverse taste)")
        
        recommendations = []
        
        # Primary recommendation action
        if best["MaxCosine"] >= 0.6 and best["Predicted_Rating"] >= 4.0:
            recommendations.append(f" **Display prominently:** Show '{best['DisplayLabel']}' as primary suggestion")
            recommendations.append(" **High confidence message:** Use phrases like 'Perfect match for you'")
        elif best["MaxCosine"] >= 0.4:
            recommendations.append(f" **Soft recommendation:** Present '{best['DisplayLabel']}' with 'You might like this'")
        else:
            recommendations.append(" **Explore alternatives:** Current top match is weak - consider showing popular items instead")
        
        # Top 5 strategy
        if avg_similarity_top5 >= 0.6:
            recommendations.append(" **Show all 5:** Strong overall quality - display full recommendation set")
        elif avg_similarity_top5 >= 0.4:
            recommendations.append(" **Show top 3:** Focus on highest-quality recommendations only")
        else:
            recommendations.append(" **Limit display:** Show only #1 or switch to popularity-based recommendations")
        
        # Model improvement
        if best["MaxCosine"] < 0.5:
            recommendations.append(" **Improve model:** Collect more user preference data or add features")
            recommendations.append(" **Feature engineering:** Enhance similarity calculation with additional signals")
        
        if worst_in_top5["Predicted_Rating"] < 3.0:
            recommendations.append(" **Filter threshold:** Exclude recommendations below 3.0 rating prediction")
        
        # Testing and optimization
        recommendations.append(" **A/B test:** Track click-through and conversion rates for these recommendations")
        recommendations.append(" **Feedback loop:** Collect user reactions to improve future predictions")
        
        if similarity_range > 0.3:
            recommendations.append(" **Quality control:** Consider setting minimum similarity threshold (e.g., 0.5)")

        st.markdown(f"""
**Insights**
{chr(10).join(f"- {i}" for i in insights)}

**Interpretation**
{chr(10).join(f"- {i}" for i in interpretations)}

**Recommendations**
{chr(10).join(f"- {r}" for r in recommendations)}
        """)
 
    st.markdown("<br>", unsafe_allow_html=True)
    st.plotly_chart(_build_scatter(plot_df), use_container_width=True)
    
    # ── Dynamic Overall System Insights ───────────────────────────────────────
    avg_similarity = plot_df["MaxCosine"].mean()
    median_similarity = plot_df["MaxCosine"].median()
    avg_rating = plot_df["Predicted_Rating"].mean()
    median_rating = plot_df["Predicted_Rating"].median()
    
    # Calculate distribution
    high_quality = (plot_df["MaxCosine"] >= 0.6).sum()
    medium_quality = ((plot_df["MaxCosine"] >= 0.4) & (plot_df["MaxCosine"] < 0.6)).sum()
    low_quality = (plot_df["MaxCosine"] < 0.4).sum()
    total_recommendations = len(plot_df)
    
    high_quality_pct = (high_quality / total_recommendations * 100) if total_recommendations > 0 else 0
    low_quality_pct = (low_quality / total_recommendations * 100) if total_recommendations > 0 else 0
    
    # Near/Far analysis
    near_count = len(plot_df[plot_df["Group"] == "Near"])
    far_count = len(plot_df[plot_df["Group"] == "Far"])
    
    # System performance level
    if avg_similarity >= 0.6:
        system_performance = "excellent"
       
    elif avg_similarity >= 0.5:
        system_performance = "strong"
     
    elif avg_similarity >= 0.4:
        system_performance = "moderate"
  
    else:
        system_performance = "weak"
        
    
    insights = [
        f"**Average Similarity:** {avg_similarity:.3f} (Median: {median_similarity:.3f})",
        f"**Average Predicted Rating:** {avg_rating:.2f}⭐ (Median: {median_rating:.2f})",
        f"**Total Candidate Products:** {human_int(total_recommendations)}",
        f"**High Quality (≥0.6):** {high_quality} ({high_quality_pct:.1f}%)",
        f"**Medium Quality (0.4-0.6):** {medium_quality}",
        f"**Low Quality (<0.4):** {low_quality} ({low_quality_pct:.1f}%)"
    ]
    
    if near_count > 0 or far_count > 0:
        insights.append(f"**Clustering:** {near_count} near matches, {far_count} far mismatches")
    
    interpretations = []
    
    # Overall system assessment
    if avg_similarity >= 0.6:
        interpretations.append(f"**{system_performance.capitalize()} recommendation engine** - System is performing very well")
        interpretations.append("Strong personalization signals detected")
    elif avg_similarity >= 0.5:
        interpretations.append(f"**{system_performance.capitalize()} performance**  - System is working adequately")
        interpretations.append("Decent personalization with room for improvement")
    elif avg_similarity >= 0.4:
        interpretations.append(f"**{system_performance.capitalize()} performance**  - System needs optimization")
        interpretations.append("Weak personalization signals - recommendations may not match user preferences well")
    else:
        interpretations.append(f"**{system_performance.capitalize()} performance**  - Critical issues detected")
        interpretations.append("Very poor personalization - system may not have enough data or features")
    
    # Quality distribution
    if high_quality_pct >= 40:
        interpretations.append(f"**Rich candidate pool** - {high_quality_pct:.0f}% of products are strong matches")
    elif high_quality_pct < 20:
        interpretations.append(f" **Limited matches** - Only {high_quality_pct:.0f}% of products are strong candidates")
    
    if low_quality_pct > 50:
        interpretations.append(" **Poor filtering** - Majority of candidates are low quality")
    
    # Rating vs similarity check
    if avg_rating >= 4.0 and avg_similarity < 0.5:
        interpretations.append(" **Interesting pattern** - High predicted ratings despite low similarity (possibly popular items)")
    elif avg_rating < 3.5 and avg_similarity >= 0.6:
        interpretations.append(" **Quality concern** - High similarity but low predicted ratings")
    
    recommendations = []
    
    # System-level actions
    if avg_similarity < 0.5:
        recommendations.append(" **Model retraining needed** - Current algorithm is underperforming")
        recommendations.append(" **Feature enhancement** - Add more user/product features to improve matching")
        recommendations.append(" **Data collection** - Gather more user interaction data for better personalization")
    elif avg_similarity < 0.6:
        recommendations.append(" **Optimization opportunity** - System is adequate but can be improved")
        recommendations.append(" **A/B test features** - Experiment with additional signals")
    else:
        recommendations.append(f" **Maintain quality** - System is performing well, continue monitoring")
    
    # Filtering recommendations
    if low_quality_pct > 40:
        recommendations.append(" **Implement thresholds** - Filter out recommendations below 0.4 similarity")
        recommendations.append(" **Focus on quality** - Show only top-tier matches to users")
    
    if high_quality_pct >= 30:
        recommendations.append(" **Leverage strong matches** - Prioritize displaying high-quality recommendations")
    else:
        recommendations.append(" **Expand catalog relevance** - Limited strong matches suggests narrow product range")
    
    # User experience
    recommendations.append(" **UI strategy:** " + (
        "Use confidence indicators when showing recommendations" if avg_similarity >= 0.5 
        else "Consider hybrid approach (personalized + popular items)"
    ))
    
    recommendations.append(" **Continuous improvement:** Monitor recommendation acceptance rates and adjust model")
    
    if near_count > 0:
        recommendations.append(f" **Validate clusters:** {near_count} products flagged as 'near' matches - investigate patterns")

    st.markdown(f"""
**Overall System Insights** 
{chr(10).join(f"- {i}" for i in insights)}

**Interpretation**
{chr(10).join(f"- {i}" for i in interpretations)}

**Recommendations**
{chr(10).join(f"- {r}" for r in recommendations)}
    """)