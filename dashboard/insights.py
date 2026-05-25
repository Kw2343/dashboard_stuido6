import pandas as pd
import numpy as np

def rating_insights(df: pd.DataFrame) -> str:
    """Generate dynamic insights based on rating distribution."""
    avg = df["rating"].mean()
    median = df["rating"].median()
    mode = df["rating"].mode().iloc[0] if not df["rating"].mode().empty else 0
    
    # Calculate distribution percentages
    low_pct = (df["rating"] <= 2).mean() * 100
    mid_pct = (df["rating"] == 3).mean() * 100
    high_pct = (df["rating"] >= 4).mean() * 100
    
    # Calculate specific ratings
    five_star_pct = (df["rating"] == 5).mean() * 100
    one_star_pct = (df["rating"] == 1).mean() * 100
    
    # Determine polarization
    polarization = (df["rating"] <= 2).mean() + (df["rating"] == 5).mean()
    is_polarized = polarization > 0.7
    
    # Calculate variance for sentiment stability
    variance = df["rating"].var()
    is_consistent = variance < 1.0
    
    # Dynamic satisfaction level
    if avg >= 4.5:
        satisfaction = "excellent"
        
    elif avg >= 4.0:
        satisfaction = "strong"
        
    elif avg >= 3.5:
        satisfaction = "moderate"
       
    elif avg >= 3.0:
        satisfaction = "mixed"
       
    else:
        satisfaction = "poor"
        
    
    # Build insights dynamically
    insights = [
        f"Average rating: **{avg:.2f}** (Median: **{median:.1f}**, Most common: **{mode}**⭐)",
        f"**{high_pct:.1f}%** positive (4-5⭐) | **{mid_pct:.1f}%** neutral (3⭐) | **{low_pct:.1f}%** negative (1-2⭐)"
    ]
    
    # Add specific distribution insights
    if five_star_pct > 50:
        insights.append(f"**{five_star_pct:.1f}%** are perfect 5-star reviews")
    if one_star_pct > 15:
        insights.append(f" **{one_star_pct:.1f}%** are 1-star reviews (quality concerns)")
    
    # Interpretation logic
    interpretations = []
    
    if is_polarized:
        interpretations.append("**Polarized feedback** - products are loved or hated with few middle opinions")
    elif is_consistent:
        interpretations.append("**Consistent ratings** - stable customer sentiment across reviews")
    else:
        interpretations.append("**Mixed feedback** - varied customer experiences")
    
    if avg >= 4.0 and high_pct >= 70:
        interpretations.append(f"**{satisfaction.capitalize()} satisfaction** - customers are generally happy")
    elif avg >= 3.5 and low_pct < 15:
        interpretations.append("**Acceptable performance** but room for improvement")
    elif low_pct > 25:
        interpretations.append("**Significant dissatisfaction** - urgent quality issues need attention")
    else:
        interpretations.append(f"**{satisfaction.capitalize()} satisfaction** - customer experience needs work")
    
    # Recommendations based on data patterns
    recommendations = []
    
    if avg >= 4.5:
        recommendations.append(" **Leverage social proof** - prominently display ratings in marketing")
        recommendations.append(" **Identify top products** - create bestseller collections based on ratings")
        if five_star_pct > 60:
            recommendations.append(" **Harvest testimonials** - convert 5-star reviews into marketing content")
    elif avg >= 4.0:
        recommendations.append(" **Promote high-rated items** while improving 3-star products")
        recommendations.append(" **Address neutral reviews** - convert 3-star feedback into actionable improvements")
    else:
        recommendations.append(" **Urgent quality review** - investigate and fix root causes of low ratings")
        recommendations.append(" **Customer service intervention** - reach out to dissatisfied customers")
    
    if low_pct > 20:
        recommendations.append(f" **Critical issue**: {low_pct:.1f}% negative reviews - conduct deep-dive analysis")
        recommendations.append(" **Implement quality control** - prevent poor products from reaching customers")
    
    if is_polarized:
        recommendations.append(" **Segment analysis needed** - identify why some love it and others hate it")
        recommendations.append(" **Improve product descriptions** - set accurate expectations to reduce polarization")
    
    if mid_pct > 25:
        recommendations.append(" **Convert neutrals to promoters** - analyze 3-star feedback for quick wins")
    
    # Format output
    insight_text = f"""
**Insights** 
{chr(10).join(f"- {i}" for i in insights)}

**Interpretation**
{chr(10).join(f"- {i}" for i in interpretations)}

**Recommendations**
{chr(10).join(f"- {r}" for r in recommendations)}
"""
    return insight_text


def review_trend_insight(yearly_df: pd.DataFrame) -> str:
    """Generate dynamic insights based on review trends over time."""
    if len(yearly_df) < 2:
        return """
**Insight**
- Insufficient data to determine trend

**Recommendation**
- Collect more historical data for trend analysis
"""
    
    # Calculate various trend metrics
    recent_year = yearly_df["review_year"].max()
    oldest_year = yearly_df["review_year"].min()
    recent_reviews = yearly_df[yearly_df["review_year"] == recent_year]["reviews"].iloc[0]
    oldest_reviews = yearly_df[yearly_df["review_year"] == oldest_year]["reviews"].iloc[0]
    
    # Year-over-year growth
    avg_growth = yearly_df["reviews"].pct_change().mean() * 100
    last_year_growth = yearly_df["reviews"].pct_change().iloc[-1] * 100 if len(yearly_df) > 1 else 0
    
    # Peak analysis
    peak_year = yearly_df.loc[yearly_df["reviews"].idxmax(), "review_year"]
    peak_reviews = yearly_df["reviews"].max()
    
    # Trend direction
    total_growth = ((recent_reviews - oldest_reviews) / oldest_reviews * 100) if oldest_reviews > 0 else 0
    
    # Calculate momentum (last 2 years vs previous period)
    if len(yearly_df) >= 4:
        recent_avg = yearly_df.tail(2)["reviews"].mean()
        previous_avg = yearly_df.head(len(yearly_df) - 2)["reviews"].mean()
        momentum = ((recent_avg - previous_avg) / previous_avg * 100) if previous_avg > 0 else 0
    else:
        momentum = avg_growth
    
    # Determine trend strength
    if avg_growth > 20:
        trend = "rapidly growing "
        strength = "strong"
    elif avg_growth > 10:
        trend = "growing "
        strength = "moderate"
    elif avg_growth > 0:
        trend = "slowly increasing "
        strength = "weak"
    elif avg_growth > -10:
        trend = "slowly declining "
        strength = "weak"
    elif avg_growth > -20:
        trend = "declining "
        strength = "moderate"
    else:
        trend = "rapidly declining "
        strength = "severe"
    
    # Build insights
    insights = [
        f"Review activity is **{trend}**",
        f"Average year-over-year change: **{avg_growth:+.1f}%**",
        f"Recent year ({recent_year}): **{recent_reviews:,}** reviews"
    ]
    
    if len(yearly_df) > 2:
        insights.append(f"Last year change: **{last_year_growth:+.1f}%**")
    
    if peak_year != recent_year:
        insights.append(f"Peak was in **{peak_year}** with **{peak_reviews:,}** reviews")
    
    # Interpretations
    interpretations = []
    
    if avg_growth > 10:
        interpretations.append(f"**{strength.capitalize()} growth momentum** - platform engagement is expanding")
        if momentum > avg_growth:
            interpretations.append("**Accelerating trend** - recent growth is faster than historical average")
        interpretations.append("Indicates increasing customer base, product popularity, or review campaigns")
    elif avg_growth > 0:
        interpretations.append(f"**{strength.capitalize()} positive trend** but growth is slowing")
        interpretations.append("May indicate market saturation or need for fresh engagement strategies")
    elif avg_growth > -10:
        interpretations.append(f"**{strength.capitalize()} decline** - engagement is dropping")
        interpretations.append("Could signal reduced product relevance, market competition, or review fatigue")
    else:
        interpretations.append(f"**{strength.capitalize()} decline** - critical engagement loss")
        interpretations.append("Urgent intervention needed to understand and reverse trend")
    
    if peak_year != recent_year:
        years_since_peak = recent_year - peak_year
        interpretations.append(f" Activity has not recovered to {peak_year} levels ({years_since_peak} years ago)")
    
    # Recommendations
    recommendations = []
    
    if avg_growth > 15:
        recommendations.append(" **Scale up** - expand product catalog to capitalize on momentum")
        recommendations.append(" **Invest in marketing** - high engagement justifies increased ad spend")
        recommendations.append(" **Improve infrastructure** - ensure platform can handle continued growth")
    elif avg_growth > 5:
        recommendations.append(" **Maintain momentum** - continue current strategies while testing new approaches")
        recommendations.append(" **Analyze growth drivers** - identify what's working and double down")
    elif avg_growth > -5:
        recommendations.append(" **Re-engage users** - launch review incentive programs and campaigns")
        recommendations.append(" **Refresh product lineup** - introduce new products to spark interest")
        recommendations.append(" **Survey customers** - understand barriers to leaving reviews")
    else:
        recommendations.append(" **Emergency intervention** - conduct comprehensive engagement audit")
        recommendations.append(" **Investigate root causes** - analyze product quality, competition, and market shifts")
        recommendations.append(" **Relaunch strategy** - consider platform redesign or major campaign")
    
    if momentum < avg_growth and avg_growth > 0:
        recommendations.append(" **Address slowdown** - recent growth is declining despite positive trend")
    
    if peak_year != recent_year and avg_growth < 0:
        recommendations.append(f" **Recovery plan needed** - aim to return to {peak_year} performance levels")
    
    # Format output
    insight_text = f"""
**Insights**
{chr(10).join(f"- {i}" for i in insights)}

**Interpretation**
{chr(10).join(f"- {i}" for i in interpretations)}

**Recommendations**
{chr(10).join(f"- {r}" for r in recommendations)}
"""
    return insight_text