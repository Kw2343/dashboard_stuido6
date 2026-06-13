from __future__ import annotations
 
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
 
from utils import style_bar_chart, section_header, human_int
from llm_insights import show_llm_insights
 
 
# ── Shared static-insights renderer ──────────────────────────────────────────
 
def _render_static_insights(insights: list[str], interpretations: list[str], recommendations: list[str]) -> None:
    st.markdown(
        "**Insights**\n" +
        "".join(f"\n* {i}" for i in insights) +
        "\n\n**Interpretation**\n" +
        "".join(f"\n* {i}" for i in interpretations) +
        "\n\n**Recommendations**\n" +
        "".join(f"\n* {i}" for i in recommendations)
    )
 
 
# ── Static insight builders ───────────────────────────────────────────────────
 
def _yearly_static_insights(reviews: pd.DataFrame, filtered_reviews: pd.DataFrame, yearly: pd.DataFrame) -> None:
    peak_year = int(yearly.loc[yearly["reviews"].idxmax(), "review_year"])
    peak_rev  = int(yearly.loc[yearly["reviews"].idxmax(), "reviews"])
    last_year = int(yearly.iloc[-1]["review_year"])
    last_rev  = int(yearly.iloc[-1]["reviews"])
    first_rev = int(yearly.iloc[0]["reviews"])
    first_year= int(yearly.iloc[0]["review_year"])
    total_fil = len(filtered_reviews)
    total_all = len(reviews)
    vals      = yearly["reviews"].tolist()
    yoy       = [(vals[i] - vals[i-1]) / vals[i-1] * 100 for i in range(1, len(vals)) if vals[i-1]]
    avg_yoy   = sum(yoy) / len(yoy) if yoy else 0
    change    = (last_rev - first_rev) / first_rev * 100 if first_rev else 0
 
    insights = [
        f"**Peak year:** {peak_year} with {human_int(peak_rev)} reviews",
        f"**Latest year ({last_year}):** {human_int(last_rev)} reviews",
        f"**Filtered dataset:** {human_int(total_fil)} of {human_int(total_all)} total reviews",
        f"**Overall change {first_year}→{last_year}:** {change:+.1f}%",
        f"**Avg year-on-year change:** {avg_yoy:+.1f}%",
    ]
 
    interpretations = []
    if avg_yoy > 10:
        interpretations.append(f"**Growing platform** — reviews growing ~{avg_yoy:.0f}% per year on average")
    elif avg_yoy < -10:
        interpretations.append(f"**Declining engagement** — review volume shrinking ~{abs(avg_yoy):.0f}% per year")
    else:
        interpretations.append("**Stable volume** — review activity is relatively flat year-on-year")
    if peak_year != last_year:
        interpretations.append(f"**Post-peak decline** — volume has fallen since the {peak_year} peak of {human_int(peak_rev)}")
    else:
        interpretations.append(f"**{last_year} is the strongest year on record** — momentum is building")
    if change > 50:
        interpretations.append("**Long-term growth story** — review volume has more than doubled over the period")
 
    recommendations = []
    if avg_yoy < 0:
        recommendations.append("**Investigate drop-off** — identify which product categories lost reviews and why")
    recommendations.append("**Align marketing calendar** to peak review periods identified in the trend")
    recommendations.append("**Post-purchase email sequences** — prompt reviews within 7–14 days of delivery to sustain volume")
    if last_rev < peak_rev * 0.7:
        recommendations.append(f"**Re-engagement campaign** — volume is {(1 - last_rev/peak_rev)*100:.0f}% below the {peak_year} peak")
 
    _render_static_insights(insights, interpretations, recommendations)
 
 
def _rating_static_insights(filtered_reviews: pd.DataFrame) -> None:
    avg  = filtered_reviews["rating"].mean()
    rd   = filtered_reviews["rating"].value_counts().sort_index()
    p5   = (filtered_reviews["rating"] == 5).mean() * 100
    p1   = (filtered_reviews["rating"] == 1).mean() * 100
    pos  = (filtered_reviews["rating"] >= 4).mean() * 100
    neg  = (filtered_reviews["rating"] <= 2).mean() * 100
    ver  = filtered_reviews["verified_purchase"].mean() * 100
    tot  = len(filtered_reviews)
 
    insights = [
        f"**Average rating:** {avg:.2f}⭐ across {human_int(tot)} filtered reviews",
        f"**5-star share:** {p5:.1f}% ({human_int(rd.get(5, 0))} reviews)",
        f"**1-star share:** {p1:.1f}% ({human_int(rd.get(1, 0))} reviews)",
        f"**Positive (4–5★):** {pos:.1f}% | **Negative (1–2★):** {neg:.1f}%",
        f"**Verified purchases:** {ver:.1f}% of reviews",
    ]
 
    interpretations = []
    if avg >= 4.3:
        interpretations.append("**Strong overall sentiment** — customers are broadly satisfied")
    elif avg >= 3.8:
        interpretations.append("**Moderate sentiment** — solid but with room for improvement")
    else:
        interpretations.append("**Weak sentiment** — below-average ratings signal product or fulfilment issues")
 
    if p1 > 15:
        interpretations.append(f"**High dissatisfaction signal** — {p1:.1f}% 1-star reviews warrants investigation")
    elif p5 > 60:
        interpretations.append(f"**5-star dominant** — {p5:.1f}% of reviews are 5-star, a strong quality indicator")
 
    if ver < 50:
        interpretations.append(f"**Low verification rate ({ver:.0f}%)** — ratings may include non-purchasers, reducing signal quality")
 
    recommendations = []
    if p1 > 10:
        recommendations.append(f"**Review negative feedback** — audit top themes in the {p1:.1f}% 1-star reviews")
    if avg >= 4.3:
        recommendations.append("**Showcase ratings** — use average score in ads and product listings as social proof")
    recommendations.append("**Verified-purchase filter** — surface verified ratings more prominently to build buyer trust")
    recommendations.append("**Monitor rating drift** — set alerts if average drops below 4.0 after filter changes")
 
    _render_static_insights(insights, interpretations, recommendations)
 
 
def _user_dist_static_insights(filtered_reviews: pd.DataFrame) -> None:
    rpu       = filtered_reviews.groupby("user_id").size()
    n         = len(rpu)
    tot       = len(filtered_reviews)
    one_time  = int((rpu == 1).sum())
    power     = int((rpu >= 20).sum())
    t10c      = rpu.nlargest(max(1, int(n * 0.1))).sum() / tot * 100
    sp        = one_time / n * 100
    sup       = int((rpu > 50).sum())
 
    insights = [
        f"**Total unique users:** {human_int(n)}",
        f"**Single-review users:** {human_int(one_time)} ({sp:.1f}% of all users)",
        f"**Power users (≥20 reviews):** {human_int(power)} ({power/n*100:.1f}%)",
        f"**Top 10% of users contribute:** {t10c:.1f}% of all reviews",
        f"**Super users (>50 reviews):** {human_int(sup)}",
    ]
 
    interpretations = []
    if sp > 70:
        interpretations.append(f"**One-time reviewer dominated** — {sp:.0f}% of users reviewed only once, limiting repeat signal")
    elif sp < 40:
        interpretations.append(f"**High repeat engagement** — only {sp:.0f}% one-time reviewers; strong loyal reviewer base")
    if t10c > 50:
        interpretations.append(f"**Heavy concentration** — top 10% of users drive {t10c:.0f}% of reviews; platform is reliant on a small group")
    if power > 0:
        interpretations.append(f"**{human_int(power)} power users** are your most valuable reviewers — protecting this group matters")
 
    recommendations = []
    if sp > 60:
        recommendations.append("**Re-engagement flow** — email one-time reviewers after repeat purchase to grow multi-review users")
    recommendations.append("**Power user programme** — reward the top 10% contributors with early access or recognition")
    if t10c > 50:
        recommendations.append(f"**Reduce concentration risk** — diversify review sources; top 10% driving {t10c:.0f}% is fragile")
    recommendations.append("**Onboarding nudge** — first-time buyers who haven't reviewed should receive a single gentle reminder")
 
    _render_static_insights(insights, interpretations, recommendations)
 
 
def _product_dist_static_insights(filtered_reviews: pd.DataFrame) -> None:
    rpp   = filtered_reviews.groupby("parent_asin").size()
    n     = len(rpp)
    tot   = len(filtered_reviews)
    lone  = int((rpp == 1).sum())
    w20   = int((rpp > 20).sum())
    t20v  = rpp.nlargest(max(1, int(n * 0.2))).sum() / tot * 100
    avg_r = rpp.mean()
    med_r = rpp.median()
 
    insights = [
        f"**Total unique products:** {human_int(n)}",
        f"**Products with only 1 review:** {human_int(lone)} ({lone/n*100:.1f}%)",
        f"**Products with 20+ reviews:** {human_int(w20)} ({w20/n*100:.1f}%)",
        f"**Top 20% products capture:** {t20v:.1f}% of all reviews",
        f"**Avg reviews per product:** {avg_r:.1f} | **Median:** {med_r:.0f}",
    ]
 
    interpretations = []
    if lone / n > 0.5:
        interpretations.append(f"**Long tail problem** — {lone/n*100:.0f}% of products have just 1 review; too sparse for reliable ranking")
    if t20v > 60:
        interpretations.append(f"**Pareto concentration** — top 20% of products account for {t20v:.0f}% of reviews; catalogue tail is underserved")
    if w20 / n < 0.1:
        interpretations.append(f"**Few well-reviewed products** — only {w20/n*100:.0f}% have 20+ reviews; discovery is limited")
 
    recommendations = []
    recommendations.append(f"**Boost thin products** — the {human_int(lone)} single-review products need review seeding campaigns")
    if t20v > 60:
        recommendations.append("**Promote long-tail** — feature products with 5–19 reviews to help them cross the social-proof threshold")
    recommendations.append("**Minimum review threshold** — set 5 reviews as the floor for recommendation engine inclusion")
    recommendations.append("**Category audit** — identify which categories have the most 1-review products and target them first")
 
    _render_static_insights(insights, interpretations, recommendations)
 
 
# ── Context builders ──────────────────────────────────────────────────────────
 
def _build_yearly_context(reviews, filtered_reviews, yearly):
    peak_year = int(yearly.loc[yearly["reviews"].idxmax(), "review_year"])
    peak_rev  = int(yearly.loc[yearly["reviews"].idxmax(), "reviews"])
    first_rev = int(yearly.iloc[0]["reviews"])
    last_rev  = int(yearly.iloc[-1]["reviews"])
    change    = (last_rev - first_rev) / first_rev * 100 if first_rev else 0
    vals = yearly["reviews"].tolist()
    yoy  = [(vals[i]-vals[i-1])/vals[i-1]*100 for i in range(1,len(vals)) if vals[i-1]]
    avg_yoy = sum(yoy)/len(yoy) if yoy else 0
    lines = [
        "YEARLY REVIEW VOLUME TREND", "",
        f"  Total reviews (unfiltered) : {human_int(len(reviews))}",
        f"  Filtered reviews           : {human_int(len(filtered_reviews))}",
        f"  Year range : {int(yearly.iloc[0]['review_year'])} – {int(yearly.iloc[-1]['review_year'])}",
        f"  Peak year  : {peak_year} ({human_int(peak_rev)} reviews)",
        f"  First→last year change : {change:+.1f}%",
        f"  Avg YoY change         : {avg_yoy:+.1f}%", "",
        "YEAR-BY-YEAR:",
    ]
    for _, row in yearly.iterrows():
        lines.append(f"  {int(row['review_year'])}: {human_int(row['reviews'])} reviews")
    return "\n".join(lines)
 
 
def _build_rating_context(filtered_reviews):
    avg   = filtered_reviews["rating"].mean()
    med   = filtered_reviews["rating"].median()
    rd    = filtered_reviews["rating"].value_counts().sort_index()
    p5    = (filtered_reviews["rating"] == 5).mean() * 100
    p4    = (filtered_reviews["rating"] == 4).mean() * 100
    p3    = (filtered_reviews["rating"] == 3).mean() * 100
    p2    = (filtered_reviews["rating"] == 2).mean() * 100
    p1    = (filtered_reviews["rating"] == 1).mean() * 100
    pos   = (filtered_reviews["rating"] >= 4).mean() * 100
    neg   = (filtered_reviews["rating"] <= 2).mean() * 100
    ver   = filtered_reviews["verified_purchase"].mean() * 100
    return "\n".join([
        "RATING DISTRIBUTION", "",
        f"  Total reviews      : {human_int(len(filtered_reviews))}",
        f"  Average rating     : {avg:.2f} ⭐",
        f"  Median rating      : {med:.0f} ⭐",
        f"  Verified purchases : {ver:.1f}%", "",
        "BREAKDOWN:",
        f"  5★ : {human_int(rd.get(5,0))} ({p5:.1f}%)",
        f"  4★ : {human_int(rd.get(4,0))} ({p4:.1f}%)",
        f"  3★ : {human_int(rd.get(3,0))} ({p3:.1f}%)",
        f"  2★ : {human_int(rd.get(2,0))} ({p2:.1f}%)",
        f"  1★ : {human_int(rd.get(1,0))} ({p1:.1f}%)", "",
        f"  Positive (4–5★) : {pos:.1f}%",
        f"  Negative (1–2★) : {neg:.1f}%",
        f"  Neutral  (3★)   : {p3:.1f}%",
    ])
 
 
def _build_user_dist_context(filtered_reviews):
    rpu   = filtered_reviews.groupby("user_id").size()
    n     = len(rpu)
    tot   = len(filtered_reviews)
    heavy = int((rpu > 10).sum())
    sup   = int((rpu > 50).sum())
    sp    = (rpu == 1).mean() * 100
    t10c  = rpu.nlargest(max(1,int(n*0.1))).sum() / tot * 100
    bins   = [0,1,5,10,20,50,np.inf]
    labels = ["1","2–5","6–10","11–20","21–50","51+"]
    counts = pd.cut(rpu,bins=bins,labels=labels,include_lowest=True).value_counts().sort_index()
    lines = [
        "USER REVIEW DISTRIBUTION", "",
        f"  Total unique users          : {human_int(n)}",
        f"  Avg reviews/user            : {rpu.mean():.1f}",
        f"  Median reviews/user         : {rpu.median():.0f}",
        f"  Top 10% users contribute    : {t10c:.1f}% of all reviews",
        f"  Users with >10 reviews      : {heavy:,} ({heavy/n*100:.1f}%)",
        f"  Super users (>50 reviews)   : {sup:,}",
        f"  Single-review users         : {sp:.1f}%", "",
        "BUCKETS:",
    ]
    for lbl, cnt in counts.items():
        lines.append(f"  {lbl:>6} reviews: {cnt:>6,} users ({cnt/n*100:.1f}%)")
    return "\n".join(lines)
 
 
def _build_product_dist_context(filtered_reviews):
    rpp = filtered_reviews.groupby("parent_asin").size()
    n   = len(rpp)
    tot = len(filtered_reviews)
    w20 = int((rpp > 20).sum())
    lon = int((rpp == 1).sum())
    t20v = rpp.nlargest(max(1,int(n*0.2))).sum() / tot * 100
    bins   = [0,1,5,10,20,50,np.inf]
    labels = ["1","2–5","6–10","11–20","21–50","51+"]
    counts = pd.cut(rpp,bins=bins,labels=labels,include_lowest=True).value_counts().sort_index()
    lines = [
        "PRODUCT REVIEW DISTRIBUTION", "",
        f"  Total unique products        : {human_int(n)}",
        f"  Avg reviews/product          : {rpp.mean():.1f}",
        f"  Median reviews/product       : {rpp.median():.0f}",
        f"  Products with 20+ reviews    : {w20:,} ({w20/n*100:.1f}%)",
        f"  Products with only 1 review  : {lon:,} ({lon/n*100:.1f}%)",
        f"  Top 20% products capture     : {t20v:.1f}% of all reviews", "",
        "BUCKETS:",
    ]
    for lbl, cnt in counts.items():
        lines.append(f"  {lbl:>6} reviews: {cnt:>6,} products ({cnt/n*100:.1f}%)")
    return "\n".join(lines)
 
 
# ── Tab entry point ───────────────────────────────────────────────────────────
 
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
    o4.metric("Avg words / filtered review",
              f"{filtered_reviews['review_length_words'].mean():.1f}")
 
    # ── Row 1: charts ─────────────────────────────────────────────────────────
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
 
    with right:
        rc = filtered_reviews["rating"].value_counts().sort_index().reset_index()
        rc.columns = ["rating", "count"]
        fig = px.bar(rc, x="rating", y="count", title="Rating distribution")
        fig.update_layout(height=420)
        st.plotly_chart(style_bar_chart(fig), use_container_width=True)
 
    # ── Row 1: static insights + AI side by side ───────────────────────────────
    left, right = st.columns(2)
    with left:
        _yearly_static_insights(reviews, filtered_reviews, yearly)
        show_llm_insights(
            context   = _build_yearly_context(reviews, filtered_reviews, yearly),
            cache_key = "overview_yearly",
            title     = "Review Trend Analysis",
            chart_type= "bar chart showing review volume by year",
        )
    with right:
        _rating_static_insights(filtered_reviews)
        show_llm_insights(
            context   = _build_rating_context(filtered_reviews),
            cache_key = "overview_ratings",
            title     = "Rating Distribution Analysis",
            chart_type= "bar chart showing count of reviews at each star rating (1–5)",
        )
 
    # ── Row 2: distribution charts ────────────────────────────────────────────
    bins   = [0, 1, 5, 10, 20, 50, np.inf]
    labels = ["1", "2–5", "6–10", "11–20", "21–50", "51+"]
 
    def _bin_chart(series, x_title, y_col, chart_title):
        binned = pd.cut(series, bins=bins, labels=labels, include_lowest=True)
        counts = binned.value_counts().sort_index().reset_index()
        counts.columns = ["reviews_range", y_col]
        counts["pct"] = (counts[y_col]/counts[y_col].sum()*100).round(1).astype(str)+"%"
        fig = px.bar(counts, x="reviews_range", y=y_col, title=chart_title, text="pct")
        fig.update_traces(textposition="outside")
        fig.update_layout(height=420, xaxis_title=x_title,
                          yaxis_title=f"Number of {y_col.split('_')[0]}s")
        return style_bar_chart(fig)
 
    left2, right2 = st.columns(2)
    with left2:
        st.plotly_chart(
            _bin_chart(filtered_reviews.groupby("user_id").size(),
                       "Reviews per user", "users", "Reviews written per user"),
            use_container_width=True,
        )
    with right2:
        st.plotly_chart(
            _bin_chart(filtered_reviews.groupby("parent_asin").size(),
                       "Reviews per product", "products", "Reviews received per product"),
            use_container_width=True,
        )
 
    # ── Row 2: static insights + AI side by side ─────────────────────────────
    left2, right2 = st.columns(2)
    with left2:
        _user_dist_static_insights(filtered_reviews)
        show_llm_insights(
            context   = _build_user_dist_context(filtered_reviews),
            cache_key = "overview_users",
            title     = "User Engagement Distribution",
            chart_type= "bar chart showing how many users fall into each review-count bucket (1, 2–5, 6–10, 11–20, 21–50, 51+)",
        )
    with right2:
        _product_dist_static_insights(filtered_reviews)
        show_llm_insights(
            context   = _build_product_dist_context(filtered_reviews),
            cache_key = "overview_products",
            title     = "Product Review Distribution",
            chart_type= "bar chart showing how many products fall into each review-count bucket (1, 2–5, 6–10, 11–20, 21–50, 51+)",
        )