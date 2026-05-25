from __future__ import annotations
 
import pandas as pd
import streamlit as st
 
from utils import pct, top_share, human_int
 
 
def show_users_tab(
    filtered_reviews: pd.DataFrame,
    users: pd.DataFrame,
) -> None:
    """
    Render the Users tab.
 
    Signature simplified: helpers (pct, top_share, human_int) are
    imported from utils so callers don't need to pass them.
    """
    st.markdown("### User analysis")
 
    # ── Summary KPIs ──────────────────────────────────────────────────────────
    u1, u2, u3 = st.columns(3)
 
    reviews_per_user = filtered_reviews.groupby("user_id").size()
    
    median_reviews = reviews_per_user.median()
    top_10_pct_share = top_share(reviews_per_user, 0.1)
    single_review_pct = (reviews_per_user == 1).mean()
 
    u1.metric("Median reviews / user",  f"{median_reviews:.0f}")
    u2.metric("Top 10 % users share",   pct(top_10_pct_share))
    u3.metric("Single-review users",    pct(single_review_pct))
 
    # ── Dynamic User Engagement Insights ──────────────────────────────────────
    total_users = reviews_per_user.nunique()
    total_reviews = len(filtered_reviews)
    avg_reviews_per_user = reviews_per_user.mean()
    
    # Segment users by activity level
    power_users = (reviews_per_user >= 20).sum()
    active_users = ((reviews_per_user >= 5) & (reviews_per_user < 20)).sum()
    casual_users = ((reviews_per_user >= 2) & (reviews_per_user < 5)).sum()
    single_review_users = (reviews_per_user == 1).sum()
    
    power_users_pct = (power_users / total_users * 100) if total_users > 0 else 0
    single_review_pct_value = (single_review_users / total_users * 100) if total_users > 0 else 0
    
    # Calculate contribution of power users
    if power_users > 0:
        power_user_reviews = reviews_per_user[reviews_per_user >= 20].sum()
        power_user_contribution = (power_user_reviews / total_reviews * 100)
    else:
        power_user_contribution = 0
    
    # Engagement quality assessment
    if top_10_pct_share >= 0.6:
        engagement_concentration = "very high"
       
    elif top_10_pct_share >= 0.4:
        engagement_concentration = "high"
        
    elif top_10_pct_share >= 0.25:
        engagement_concentration = "moderate"
        
    else:
        engagement_concentration = "balanced"
        
    
    if median_reviews >= 5:
        typical_engagement = "strong"
    elif median_reviews >= 3:
        typical_engagement = "moderate"
    else:
        typical_engagement = "low"
    
    insights = [
        f"**Total Active Users:** {human_int(total_users)}",
        f"**Median Reviews/User:** {median_reviews:.0f} (Average: {avg_reviews_per_user:.1f})",
        f"**Top 10% Control:** {top_10_pct_share*100:.1f}% of all reviews",
        f"**Power Users (20+ reviews):** {power_users} ({power_users_pct:.1f}%) contribute {power_user_contribution:.1f}% of reviews",
        f"**Single-Review Users:** {single_review_users} ({single_review_pct_value:.1f}%)"
    ]
    
    insights.append(f"**User Segments:** {power_users} power | {active_users} active | {casual_users} casual | {single_review_users} one-time")
    
    interpretations = []
    
    # Concentration analysis
    if top_10_pct_share >= 0.5:
        interpretations.append(f"**{engagement_concentration.capitalize()} concentration** - Top 10% dominate engagement (dependency risk)")
        interpretations.append("Platform heavily relies on small user group")
    elif top_10_pct_share >= 0.3:
        interpretations.append(f"**{engagement_concentration.capitalize()} concentration**  - Significant but not extreme dependency")
    else:
        interpretations.append(f"**{engagement_concentration.capitalize()} distribution** - Healthy engagement spread across users")
    
    # Typical user behavior
    if median_reviews <= 2:
        interpretations.append(f"**{typical_engagement.capitalize()} typical engagement** - Most users write few reviews")
        interpretations.append("Engagement strategies needed to activate casual users")
    elif median_reviews >= 5:
        interpretations.append(f"**{typical_engagement.capitalize()} typical engagement** - Users are generally active")
    
    # Single-review problem
    if single_review_pct_value >= 50:
        interpretations.append(f" **High churn risk** - {single_review_pct_value:.0f}% of users only write once and never return")
    elif single_review_pct_value >= 30:
        interpretations.append(f" **Retention challenge** - {single_review_pct_value:.0f}% are one-time contributors")
    
    # Power user impact
    if power_user_contribution >= 40:
        interpretations.append(f"**Critical dependency** - {power_users} power users ({power_users_pct:.1f}%) drive {power_user_contribution:.0f}% of content")
    
    recommendations = []
    
    # Address concentration
    if top_10_pct_share >= 0.5:
        recommendations.append(" **Diversify engagement** - Reduce dependency on top 10% through broad user activation")
        recommendations.append(" **Activate middle tier** - Convert casual users (2-4 reviews) into active contributors")
    
    # Power user strategy
    if power_users_pct >= 5:
        recommendations.append(f" **Reward power users** - Implement VIP program for {power_users} super contributors")
        recommendations.append(" **Gamification:** Badges, leaderboards, exclusive perks for top reviewers")
    elif power_users_pct < 2:
        recommendations.append("📈 **Cultivate power users** - Create incentives to develop more highly engaged users")
    
    # Single-review problem
    if single_review_pct_value >= 40:
        recommendations.append(f" **Retention crisis:** {single_review_pct_value:.0f}% never return - implement onboarding and follow-up campaigns")
        recommendations.append(" **Re-engagement:** Email users after first review to encourage continued participation")
    elif single_review_pct_value >= 25:
        recommendations.append(" **Improve retention:** Launch programs to convert one-time users into regular contributors")
    
    # General engagement
    if median_reviews < 3:
        recommendations.append(" **Simplify review process:** Remove friction to encourage more frequent participation")
        recommendations.append("**Incentivize reviews:** Rewards, discounts, or points for each review submitted")
    
    recommendations.append("**Segment campaigns:** Target power users differently than casual contributors")
    recommendations.append("**Progression system:** Create clear path from casual → active → power user")
    
    if avg_reviews_per_user < 5:
        recommendations.append(" **Engagement push:** Platform-wide campaign to increase review frequency")
    
    # Risk mitigation
    if power_user_contribution >= 50:
        recommendations.append(" **Risk mitigation:** Losing a few power users could devastate content volume")

    st.markdown(f"""
**Insights** 
{chr(10).join(f"- {i}" for i in insights)}

**Interpretation**
{chr(10).join(f"- {i}" for i in interpretations)}

**Recommendations**
{chr(10).join(f"- {r}" for r in recommendations)}
    """)
 
    # ── User stats table (merged) ─────────────────────────────────────────────
    st.subheader("User statistics")
 
    active_ids = filtered_reviews["user_id"].unique()
    user_subset = users[users["user_id"].isin(active_ids)].copy()
 
    st.dataframe(user_subset.head(500), use_container_width=True)
    
    # ── Dynamic Top User Insights ─────────────────────────────────────────────
    best_user = reviews_per_user.idxmax()
    max_reviews = reviews_per_user.max()
    second_best_reviews = reviews_per_user.nlargest(2).iloc[1] if len(reviews_per_user) > 1 else 0
    
    top_user_share = (max_reviews / total_reviews * 100) if total_reviews > 0 else 0
    gap_to_second = max_reviews - second_best_reviews
    
    # Get top 10 users
    top_10_users = reviews_per_user.nlargest(10)
    top_10_total = top_10_users.sum()
    top_10_contribution = (top_10_total / total_reviews * 100) if total_reviews > 0 else 0
    
    if max_reviews >= 100:
        user_level = "super power user"
      
    elif max_reviews >= 50:
        user_level = "power user"
     
    elif max_reviews >= 20:
        user_level = "active contributor"
        
    else:
        user_level = "regular user"
        
    
    insights = [
        f"**Most Active User:** {best_user}",
        f"**Total Reviews:** {human_int(max_reviews)} ({user_level}) ",
        f"**Contribution:** {top_user_share:.2f}% of all platform reviews",
    ]
    
    if len(reviews_per_user) > 1:
        insights.append(f"**Gap to #2:** {human_int(gap_to_second)} reviews ({(gap_to_second/max_reviews*100):.1f}%)")
    
    insights.append(f"**Top 10 Users:** Contribute {top_10_contribution:.1f}% of all reviews")
    
    interpretations = []
    
    if max_reviews >= 50:
        interpretations.append(f"**Exceptional engagement**  - This user is a platform champion")
        interpretations.append("Represents ideal user behavior and commitment")
    elif max_reviews >= 20:
        interpretations.append(f"**Strong engagement**  - Highly active and valuable contributor")
    else:
        interpretations.append(f"**Moderate engagement** - Active but not exceptional")
    
    if top_user_share >= 5:
        interpretations.append(f" **Single-user dependency** - One person contributes {top_user_share:.1f}% of all content")
    elif top_user_share >= 2:
        interpretations.append("**Significant individual impact** - Single user drives meaningful volume")
    
    if gap_to_second >= max_reviews * 0.5:
        interpretations.append("**Dominant leader** - Far ahead of second-most-active user")
    
    if top_10_contribution >= 30:
        interpretations.append(f"**High concentration risk** - Just 10 users generate {top_10_contribution:.0f}% of content")
    
    recommendations = []
    
    if max_reviews >= 50:
        recommendations.append(f" **VIP treatment:** Give user {best_user} exclusive perks, early access, or special recognition")
        recommendations.append(" **Case study:** Understand their motivation and replicate it across user base")
        recommendations.append(" **Community leadership:** Invite them to beta testing, advisory groups, or ambassador programs")
    elif max_reviews >= 20:
        recommendations.append(f" **Recognize publicly:** Feature user {best_user} in newsletters or leaderboards")
        recommendations.append(" **Reward loyalty:** Special badges, discounts, or rewards")
    
    if top_user_share >= 3:
        recommendations.append(" **Diversification urgent:** Platform too dependent on single user - scale engagement broadly")
    
    if max_reviews >= 20:
        recommendations.append(" **Personalized recommendations:** Use for testing new recommendation algorithms")
        recommendations.append(" **Feedback channel:** Their behavior provides insights into platform usage patterns")
    
    recommendations.append(" **Identify patterns:** Analyze what makes top users so engaged")
    recommendations.append(" **Engagement ladder:** Create program to move users from casual → active → power status")
    recommendations.append(" **Retention strategy:** Ensure power users remain satisfied and engaged")
    
    if top_10_contribution >= 40:
        recommendations.append("⚡ **Risk management:** Platform stability depends on retaining these 10 users")

    st.markdown(f"""
**Top User Insights** 
{chr(10).join(f"- {i}" for i in insights)}

**Interpretation**
{chr(10).join(f"- {i}" for i in interpretations)}

**Recommendations**
{chr(10).join(f"- {r}" for r in recommendations)}
    """)