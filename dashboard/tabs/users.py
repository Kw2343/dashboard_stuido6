from __future__ import annotations
 
import pandas as pd
import streamlit as st
 
from utils import pct, top_share, human_int
from llm_insights import show_llm_insights
 
 
# ── Shared static-insights renderer ──────────────────────────────────────────
 
def _render_static_insights(insights, interpretations, recommendations):
    st.markdown(
        "**Insights**\n" +
        "".join(f"\n* {i}" for i in insights) +
        "\n\n**Interpretation**\n" +
        "".join(f"\n* {i}" for i in interpretations) +
        "\n\n**Recommendations**\n" +
        "".join(f"\n* {i}" for i in recommendations)
    )
 
 
# ── Static insight builders ───────────────────────────────────────────────────
 
def _engagement_static(reviews_per_user: pd.Series, total_reviews: int) -> None:
    n         = len(reviews_per_user)
    one_time  = int((reviews_per_user == 1).sum())
    power     = int((reviews_per_user >= 20).sum())
    active    = int(((reviews_per_user >= 5) & (reviews_per_user < 20)).sum())
    casual    = int(((reviews_per_user >= 2) & (reviews_per_user < 5)).sum())
    t10c      = reviews_per_user.nlargest(max(1, int(n * 0.1))).sum() / total_reviews * 100
    sp        = one_time / n * 100
    avg_r     = reviews_per_user.mean()
    med_r     = reviews_per_user.median()
    power_contrib = reviews_per_user[reviews_per_user >= 20].sum() / total_reviews * 100 if power > 0 else 0
 
    insights = [
        f"**Total unique users:** {human_int(n)}",
        f"**Avg reviews/user:** {avg_r:.1f} | **Median:** {med_r:.0f}",
        f"**Top 10% users contribute:** {t10c:.1f}% of all reviews",
        f"**Single-review (one-time) users:** {human_int(one_time)} ({sp:.1f}%)",
        f"**Power users (≥20 reviews):** {human_int(power)} ({power/n*100:.1f}%) — contribute {power_contrib:.1f}% of all reviews",
        f"**Active (5–19):** {human_int(active)} | **Casual (2–4):** {human_int(casual)} | **One-time:** {human_int(one_time)}",
    ]
 
    interpretations = []
    if sp > 70:
        interpretations.append(f"**One-time reviewer majority** — {sp:.0f}% of users reviewed only once; platform lacks sticky reviewers")
    elif sp < 40:
        interpretations.append(f"**Strong repeat engagement** — only {sp:.0f}% one-time reviewers; healthy loyal reviewer base")
    if t10c > 50:
        interpretations.append(f"**Highly concentrated** — top 10% drive {t10c:.0f}% of reviews; losing key users would be damaging")
    if power_contrib > 30:
        interpretations.append(f"**Power users are critical** — {power/n*100:.1f}% of users ({human_int(power)}) generate {power_contrib:.0f}% of all content")
 
    recommendations = []
    if sp > 60:
        recommendations.append(f"**Convert one-timers** — target the {human_int(one_time)} single-review users with a follow-up purchase incentive")
    recommendations.append(f"**Power user retention** — {human_int(power)} power users drive {power_contrib:.0f}% of reviews; give them VIP perks")
    if t10c > 50:
        recommendations.append("**Broaden contributor base** — reduce concentration risk by activating mid-tier (5–19 review) users")
    recommendations.append("**Segment email campaigns** — casual vs active vs power users need different messaging and incentives")
 
    _render_static_insights(insights, interpretations, recommendations)
 
 
def _top_user_static(reviews_per_user: pd.Series, total_reviews: int) -> None:
    best_user   = reviews_per_user.idxmax()
    max_reviews = int(reviews_per_user.max())
    second_best = int(reviews_per_user.nlargest(2).iloc[1]) if len(reviews_per_user) > 1 else 0
    gap         = max_reviews - second_best
    user_share  = max_reviews / total_reviews * 100 if total_reviews else 0
    top10       = reviews_per_user.nlargest(10)
    top10_sum   = int(top10.sum())
    top10_share = top10_sum / total_reviews * 100 if total_reviews else 0
 
    if max_reviews >= 100:
        level = "Super Power User (100+ reviews)"
    elif max_reviews >= 50:
        level = "Power User (50–99 reviews)"
    elif max_reviews >= 20:
        level = "Active Contributor (20–49 reviews)"
    else:
        level = "Regular User (<20 reviews)"
 
    insights = [
        f"**Most active user:** {best_user} — [{level}]",
        f"**Their review count:** {human_int(max_reviews)} ({user_share:.2f}% of all reviews)",
        f"**Gap to #2 user:** {human_int(gap)} reviews ({gap/max_reviews*100:.1f}% ahead)",
        f"**Top-10 users combined:** {human_int(top10_sum)} reviews ({top10_share:.1f}% of platform total)",
        f"**Top-10 review counts:** {', '.join(str(v) for v in top10.values)}",
    ]
 
    interpretations = []
    if user_share > 1:
        interpretations.append(f"**Disproportionate influence** — one user accounts for {user_share:.1f}% of all reviews")
    if gap > max_reviews * 0.3:
        interpretations.append(f"**Clear top contributor** — #1 user is {gap/max_reviews*100:.0f}% ahead of #2; no close rival")
    if top10_share > 20:
        interpretations.append(f"**Top-10 outsized impact** — just 10 users drive {top10_share:.0f}% of all reviews; high dependency")
 
    recommendations = []
    recommendations.append(f"**Protect your #1 reviewer** — {human_int(max_reviews)} reviews from one user; ensure they stay engaged")
    if top10_share > 15:
        recommendations.append(f"**Churn risk monitoring** — top-10 users represent {top10_share:.0f}% of content; track their activity monthly")
    recommendations.append("**Community recognition** — publicly acknowledge top contributors to reinforce their behaviour")
    recommendations.append("**Seed next tier** — identify users at rank 11–50 and nudge them toward the top 10 with targeted rewards")
 
    _render_static_insights(insights, interpretations, recommendations)
 
 
# ── Context builders ──────────────────────────────────────────────────────────
 
def _build_engagement_context(reviews_per_user, total_reviews):
    total_users    = len(reviews_per_user)
    median_rev     = reviews_per_user.median()
    avg_rev        = reviews_per_user.mean()
    top10_share    = top_share(reviews_per_user, 0.1)
    single_rev_pct = (reviews_per_user == 1).mean() * 100
    power   = int((reviews_per_user >= 20).sum())
    active  = int(((reviews_per_user >= 5) & (reviews_per_user < 20)).sum())
    casual  = int(((reviews_per_user >= 2) & (reviews_per_user < 5)).sum())
    one_time = int((reviews_per_user == 1).sum())
    power_contrib = 0.0
    if power > 0:
        power_contrib = reviews_per_user[reviews_per_user >= 20].sum() / total_reviews * 100
    return "\n".join([
        "USER ENGAGEMENT ANALYSIS", "",
        "AUDIENCE SIZE:",
        f"  Total unique users : {human_int(total_users)}",
        f"  Total reviews      : {human_int(total_reviews)}", "",
        "REVIEW FREQUENCY:",
        f"  Median reviews/user: {median_rev:.0f}",
        f"  Average reviews/user: {avg_rev:.1f}",
        f"  Top 10% users contribute: {top10_share*100:.1f}% of all reviews",
        f"  Single-review (one-time) users: {one_time} ({single_rev_pct:.1f}%)", "",
        "USER SEGMENTS:",
        f"  Power users  (≥20 reviews): {power:,}  ({power/total_users*100:.1f}%)"
        f"  — contribute {power_contrib:.1f}% of all reviews",
        f"  Active users (5–19 reviews): {active:,}  ({active/total_users*100:.1f}%)",
        f"  Casual users (2–4  reviews): {casual:,}  ({casual/total_users*100:.1f}%)",
        f"  One-time     (1 review)    : {one_time:,} ({one_time/total_users*100:.1f}%)",
    ])
 
 
def _build_top_user_context(reviews_per_user, total_reviews):
    best_user    = reviews_per_user.idxmax()
    max_reviews  = int(reviews_per_user.max())
    second_best  = int(reviews_per_user.nlargest(2).iloc[1]) if len(reviews_per_user) > 1 else 0
    gap          = max_reviews - second_best
    user_share   = max_reviews / total_reviews * 100 if total_reviews else 0
    top10        = reviews_per_user.nlargest(10)
    top10_contrib = top10.sum() / total_reviews * 100 if total_reviews else 0
    if max_reviews >= 100:
        level = "super power user (100+ reviews)"
    elif max_reviews >= 50:
        level = "power user (50–99 reviews)"
    elif max_reviews >= 20:
        level = "active contributor (20–49 reviews)"
    else:
        level = "regular user (<20 reviews)"
    return "\n".join([
        "TOP USER ANALYSIS", "",
        "MOST ACTIVE USER:",
        f"  User ID         : {best_user}",
        f"  Total reviews   : {human_int(max_reviews)}  [{level}]",
        f"  % of all reviews: {user_share:.2f}%",
        f"  Gap to #2 user  : {human_int(gap)} reviews ({gap/max_reviews*100:.1f}% ahead)", "",
        "TOP-10 USERS COMBINED:",
        f"  Total reviews by top 10: {human_int(top10.sum())}",
        f"  Share of platform       : {top10_contrib:.1f}%",
        f"  Their review counts     : {', '.join(str(v) for v in top10.values)}",
    ])
 
 
# ── Tab entry point ───────────────────────────────────────────────────────────
 
def show_users_tab(
    filtered_reviews: pd.DataFrame,
    users: pd.DataFrame,
) -> None:
    st.markdown("### User analysis")
 
    # ── Summary KPIs ──────────────────────────────────────────────────────────
    u1, u2, u3 = st.columns(3)
 
    reviews_per_user  = filtered_reviews.groupby("user_id").size()
    median_reviews    = reviews_per_user.median()
    top_10_pct_share  = top_share(reviews_per_user, 0.1)
    single_review_pct = (reviews_per_user == 1).mean()
 
    u1.metric("Median reviews / user", f"{median_reviews:.0f}")
    u2.metric("Top 10 % users share",  pct(top_10_pct_share))
    u3.metric("Single-review users",   pct(single_review_pct))
 
    _engagement_static(reviews_per_user, len(filtered_reviews))
    show_llm_insights(
        context   = _build_engagement_context(reviews_per_user, len(filtered_reviews)),
        cache_key = "users_engagement",
        title     = "User Engagement Analysis",
        chart_type= "KPI metrics and user segment breakdown showing review frequency distribution across power, active, casual, and one-time users",
    )
 
    # ── User stats table ──────────────────────────────────────────────────────
    st.subheader("User statistics")
    active_ids  = filtered_reviews["user_id"].unique()
    user_subset = users[users["user_id"].isin(active_ids)].copy()
    st.dataframe(user_subset.head(500), use_container_width=True)
 
    _top_user_static(reviews_per_user, len(filtered_reviews))
    show_llm_insights(
        context   = _build_top_user_context(reviews_per_user, len(filtered_reviews)),
        cache_key = "users_top",
        title     = "Top User Deep-Dive",
        chart_type= "user statistics table showing the most active reviewers and their contribution to total review volume",
    )