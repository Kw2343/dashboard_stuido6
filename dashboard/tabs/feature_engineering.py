from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st
from pathlib import Path

from feature_engineering import feature_engineering, product_feature_importance
from utils import style_bar_chart, human_int
from llm_insights import show_llm_insights


# ── CBF scores loader ─────────────────────────────────────────────────────────

_CBF_PATH = Path(r"C:\Users\kelvi\Downloads\dashboard - Copy (2) - Copy\dashboard_data\popular_products.csv")

@st.cache_data(show_spinner=False)
def _load_cbf_scores() -> pd.DataFrame:
    if not _CBF_PATH.exists():
        return pd.DataFrame(columns=["parent_asin"])
    df = pd.read_csv(_CBF_PATH, low_memory=False)
    keep = ["parent_asin"] + [
        c for c in ["tfidf_popularity_score", "dl_content_score"]
        if c in df.columns
    ]
    return df[keep].drop_duplicates("parent_asin")
 
 
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
 
 
# ── Progress bar helper ───────────────────────────────────────────────────────
 
def _progress_row(label: str, value: float) -> None:
    st.markdown(f"""
    <div style="margin-bottom:14px;">
        <div style="font-weight:600;margin-bottom:6px;">{label} — {value:.1f}%</div>
        <div style="width:100%;height:14px;background:#e5e7eb;border-radius:999px;overflow:hidden;">
            <div style="width:{value}%;height:100%;background:#3b82f6;border-radius:999px;"></div>
        </div>
    </div>""", unsafe_allow_html=True)
 
 
# ── Static insight builders ───────────────────────────────────────────────────
 
def _metadata_completeness_static(importance_df: pd.DataFrame, total_products: int) -> None:
    avg_c    = importance_df["value"].mean()
    lowest   = importance_df.sort_values("value").iloc[0]
    highest  = importance_df.sort_values("value").iloc[-1]
    critical = importance_df[importance_df["value"] < 70]
    excellent= importance_df[importance_df["value"] >= 90]
 
    insights = [
        f"**Total products:** {human_int(total_products)}",
        f"**Average completeness:** {avg_c:.1f}%",
        f"**Best field:** {highest['feature']} at {highest['value']:.1f}%",
        f"**Worst field:** {lowest['feature']} at {lowest['value']:.1f}% ({int(total_products*(100-lowest['value'])/100):,} products missing)",
        f"**Fields ≥90% complete:** {len(excellent)} | **Fields <70% (critical):** {len(critical)}",
    ]
 
    interpretations = []
    if avg_c >= 90:
        interpretations.append(f"**Excellent data quality** — {avg_c:.0f}% average completeness means the catalogue is well-maintained")
    elif avg_c >= 70:
        interpretations.append(f"**Moderate completeness** — {avg_c:.0f}% average; gaps will affect recommendation quality")
    else:
        interpretations.append(f"**Poor data quality** — only {avg_c:.0f}% average completeness; major gaps hurt search and ML features")
    if len(critical) > 0:
        interpretations.append(f"**{len(critical)} critical gap(s)** — fields below 70% need urgent attention")
    if lowest["value"] < 50:
        interpretations.append(f"**{lowest['feature']} is severely incomplete** — only {lowest['value']:.0f}% filled in")
 
    recommendations = []
    recommendations.append(f"**Fix '{lowest['feature']}' first** — lowest field at {lowest['value']:.0f}%; biggest impact per effort")
    if len(critical) > 0:
        recommendations.append(f"**Data enrichment sprint** — address {len(critical)} fields below 70% in the next catalogue update cycle")
    recommendations.append("**Automated completeness checks** — add validation gates to product upload pipeline")
    recommendations.append("**Supplier data push** — request missing metadata from suppliers for top-selling products first")
 
    _render_static_insights(insights, interpretations, recommendations)
 
 
def _feature_quality_static(features_df: pd.DataFrame) -> None:
    total    = len(features_df)
    avg_s    = features_df["score"].mean()
    std_s    = features_df["score"].std()
    high_s   = int((features_df["score"] >= 4.0).sum())
    low_s    = int((features_df["score"] < 2.0).sum())
    avg_freq = features_df["purchase_frequency"].mean()
    avg_u    = features_df["unique_users"].mean()
    avg_r    = features_df["avg_rating"].mean()
    high_r   = int((features_df["avg_rating"] >= 4.5).sum())
    low_eng  = int((features_df["unique_users"] < 10).sum())
 
    insights = [
        f"**Total products in feature set:** {human_int(total)}",
        f"**Avg popularity score:** {avg_s:.3f} / 5.00 (std dev: {std_s:.3f})",
        f"**High performers (score ≥4.0):** {high_s} ({high_s/total*100:.1f}%) | **Low (<2.0):** {low_s} ({low_s/total*100:.1f}%)",
        f"**Avg purchase frequency:** {avg_freq:.1f} | **Avg unique users:** {avg_u:.1f}",
        f"**Well-rated (≥4.5★):** {high_r} ({high_r/total*100:.1f}%) | **Low-engagement (<10 users):** {low_eng} ({low_eng/total*100:.1f}%)",
    ]
 
    interpretations = []
    if avg_s >= 3.5:
        interpretations.append(f"**Strong overall catalogue** — avg score of {avg_s:.2f} suggests most products are performing well")
    elif avg_s < 2.5:
        interpretations.append(f"**Weak catalogue performance** — avg score of only {avg_s:.2f}; most products need attention")
    if low_eng / total > 0.3:
        interpretations.append(f"**Engagement gap** — {low_eng/total*100:.0f}% of products have fewer than 10 unique users; poor visibility")
    if high_r / total > 0.5:
        interpretations.append(f"**Quality-rich catalogue** — {high_r/total*100:.0f}% of products rated ≥4.5★; strong foundation for recommendations")
 
    recommendations = []
    if low_s > 0:
        recommendations.append(f"**Review {low_s} low-scoring products** — scores below 2.0 warrant investigation or delisting")
    if low_eng / total > 0.2:
        recommendations.append(f"**Boost exposure** — {low_eng} products have <10 unique users; run targeted visibility campaigns")
    recommendations.append("**Score-based ranking** — use engineered scores to power homepage, search, and recommendation ranking")
    recommendations.append("**Feature replication** — analyse what drives high scores and apply those patterns to mid-tier products")
 
    _render_static_insights(insights, interpretations, recommendations)
 
 
def _top_products_static(top: pd.DataFrame) -> None:
    top_sorted = top.sort_values("score", ascending=False)
    best       = top_sorted.iloc[0]
    second     = top_sorted.iloc[1] if len(top_sorted) > 1 else None
    gap        = best["score"] - second["score"] if second is not None else 0
    avg_t      = top["score"].mean()
    med_t      = top["score"].median()
    high_r_cnt = int((top["avg_rating"] >= 4.5).sum()) if "avg_rating" in top.columns else 0
    high_r_pct = high_r_cnt / len(top) * 100
 
    insights = [
        f"**#1 Product:** {str(best.get('title', ''))[:60]}",
        f"**Score:** {best['score']:.4f} / 5.00",
        f"**Avg rating:** {best.get('avg_rating', 'N/A')} | **Unique users:** {human_int(best.get('unique_users', 0))}",
        f"**Purchase frequency:** {best.get('purchase_frequency', 0):.1f}",
        f"**Gap to #2:** {gap:.4f} points",
        f"**Average score (Top 20):** {avg_t:.4f} | **Median:** {med_t:.4f}",
        f"**Products with 4.5★+:** {high_r_cnt} ({high_r_pct:.0f}% of top 20)",
    ]
 
    interpretations = []
    if gap > 0.3:
        interpretations.append(f"**Clear market leader** — #1 product's {gap:.3f} point gap over #2 is significant")
    elif gap < 0.1:
        interpretations.append("**Tight competition** — top products are closely matched; small changes could reshuffle rankings")
    if high_r_pct >= 70:
        interpretations.append(f"**Strong top-20 quality** — {high_r_pct:.0f}% of top products have 4.5★+ ratings")
    if best.get("purchase_frequency", 0) > avg_t * 2:
        interpretations.append("**Volume + quality winner** — #1 product leads on both purchase frequency and rating")
 
    recommendations = []
    if best["score"] >= 4.0:
        recommendations.append(f"**Hero product strategy** — feature '{str(best.get('title',''))[:40]}' on homepage and in campaigns")
    else:
        recommendations.append("**No clear standout** — consider promoting top 3–5 products equally until a leader emerges")
    if high_r_pct >= 70:
        recommendations.append("**Leverage social proof** — display ratings prominently; most top products back it up")
    recommendations.append("**Cross-sell pairings** — bundle top-scored products with complementary items")
    recommendations.append("**Replication analysis** — identify which features drive top scores and apply to mid-tier products")
 
    _render_static_insights(insights, interpretations, recommendations)
 
 
def _recency_static(recency: pd.DataFrame) -> None:
    recent   = recency.iloc[0]
    days_ago = recent["days_since_last_purchase"]
    very_r   = int((recency["days_since_last_purchase"] <= 7).sum())
    last_30  = int((recency["days_since_last_purchase"] <= 30).sum())
    avg_rec  = recency["days_since_last_purchase"].mean()
    max_rec  = recency["days_since_last_purchase"].max()
    total    = len(recency)
 
    insights = [
        f"**Most recently purchased:** {str(recent.get('title', ''))[:60]}",
        f"**Days since last purchase:** {days_ago:.0f} days",
        f"**Purchased within 7 days:** {very_r} of {total} shown",
        f"**Purchased within 30 days:** {last_30} of {total} shown",
        f"**Average recency (Top 30):** {avg_rec:.0f} days",
        f"**Least recent (of Top 30):** {max_rec:.0f} days ago",
    ]
 
    interpretations = []
    if very_r >= 5:
        interpretations.append(f"**Active demand** — {very_r} products purchased within the last 7 days; strong current momentum")
    elif very_r == 0:
        interpretations.append("**No very recent activity** — no purchases in the last 7 days; possible seasonal slowdown")
    if last_30 / total >= 0.6:
        interpretations.append(f"**Catalogue is live** — {last_30/total*100:.0f}% of top-30 products sold within 30 days")
    if avg_rec > 60:
        interpretations.append(f"**Ageing demand** — average recency of {avg_rec:.0f} days suggests purchases are slowing down")
 
    recommendations = []
    if very_r >= 3:
        recommendations.append(f"**Promote trending items** — the {very_r} products with purchases in the last 7 days should get 'trending' badges")
    if last_30 < total * 0.5:
        recommendations.append(f"**Re-activate stale products** — over half the top-30 haven't sold in 30+ days; run flash promotions")
    recommendations.append("**Recency weighting** — factor days-since-purchase into recommendation ranking to surface fresh products")
    recommendations.append("**Restocking alerts** — for the most recently purchased items, verify stock levels are sufficient")
 
    _render_static_insights(insights, interpretations, recommendations)
 
 
# ── Context builders ──────────────────────────────────────────────────────────
 
def _build_metadata_context(importance_df, total_products):
    avg_c    = importance_df["value"].mean()
    lowest   = importance_df.sort_values("value").iloc[0]
    highest  = importance_df.sort_values("value").iloc[-1]
    critical = importance_df[importance_df["value"] < 70]
    excellent= importance_df[importance_df["value"] >= 90]
    lines = [
        "PRODUCT METADATA COMPLETENESS", "",
        f"  Total products           : {human_int(total_products)}",
        f"  Average completeness     : {avg_c:.1f}%",
        f"  Best field               : {highest['feature']} ({highest['value']:.1f}%)",
        f"  Weakest field            : {lowest['feature']} ({lowest['value']:.1f}%)"
        f"  — {int(total_products*(100-lowest['value'])/100):,} products missing",
        f"  Fields ≥90%              : {len(excellent)}",
        f"  Fields <70% (critical)   : {len(critical)}", "",
        "ALL FIELDS:",
    ]
    for _, row in importance_df.sort_values("value", ascending=False).iterrows():
        missing = int(total_products * (100 - row["value"]) / 100)
        lines.append(f"  {row['feature']:15s}: {row['value']:5.1f}%  ({missing:,} missing)")
    return "\n".join(lines)
 
 
def _build_feature_quality_context(features_df):
    total    = len(features_df)
    avg_s    = features_df["score"].mean()
    std_s    = features_df["score"].std()
    high_s   = int((features_df["score"] >= 4.0).sum())
    low_s    = int((features_df["score"] < 2.0).sum())
    avg_freq = features_df["purchase_frequency"].mean()
    avg_u    = features_df["unique_users"].mean()
    avg_r    = features_df["avg_rating"].mean()
    high_r   = int((features_df["avg_rating"] >= 4.5).sum())
    low_eng  = int((features_df["unique_users"] < 10).sum())
    return "\n".join([
        "ENGINEERED FEATURE QUALITY OVERVIEW", "",
        f"  Total products in feature set  : {human_int(total)}",
        f"  Avg popularity score           : {avg_s:.3f} / 5.00",
        f"  Score std dev                  : {std_s:.3f}",
        f"  High performers (score ≥4.0)   : {high_s} ({high_s/total*100:.1f}%)",
        f"  Low performers  (score <2.0)   : {low_s} ({low_s/total*100:.1f}%)", "",
        f"  Avg purchase frequency         : {avg_freq:.1f}",
        f"  Avg unique users               : {avg_u:.1f}",
        f"  Avg product rating             : {avg_r:.2f} ⭐",
        f"  Well-rated products (≥4.5★)    : {high_r} ({high_r/total*100:.1f}%)",
        f"  Low-engagement (<10 users)     : {low_eng} ({low_eng/total*100:.1f}%)",
    ])
 
 
def _build_top_products_context(top):
    best   = top.iloc[-1]
    second = top.iloc[-2] if len(top) > 1 else None
    gap    = best["score"] - second["score"] if second is not None else 0
    avg_t  = top["score"].mean()
    lines = [
        "TOP 20 PRODUCTS BY POPULARITY SCORE", "",
        f"  #1 product        : {str(best.get('title',''))[:70]}",
        f"  Score             : {best['score']:.4f} / 5.00",
        f"  Avg rating        : {best.get('avg_rating', 'N/A')}",
        f"  Unique users      : {human_int(best.get('unique_users', 0))}",
        f"  Purchase frequency: {best.get('purchase_frequency', 0):.1f}",
        f"  Gap to #2         : {gap:.4f}",
        f"  Average (Top 20)  : {avg_t:.4f}", "",
        "TOP 20 SCORES:",
    ]
    for _, row in top.sort_values("score", ascending=False).iterrows():
        lines.append(f"  {str(row.get('title',''))[:50]:50s}  score={row['score']:.3f}")
    return "\n".join(lines)
 
 
def _build_recency_context(recency):
    recent   = recency.iloc[0]
    days_ago = recent["days_since_last_purchase"]
    very_r   = int((recency["days_since_last_purchase"] <= 7).sum())
    last_30  = int((recency["days_since_last_purchase"] <= 30).sum())
    avg_rec  = recency["days_since_last_purchase"].mean()
    max_rec  = recency["days_since_last_purchase"].max()
    return "\n".join([
        "PRODUCT PURCHASE RECENCY (Top 30 most recent)", "",
        f"  Most recently purchased  : {str(recent.get('title',''))[:70]}",
        f"  Days since last purchase : {days_ago:.0f} days",
        f"  Purchased within 7 days  : {very_r} products",
        f"  Purchased within 30 days : {last_30} products",
        f"  Average recency (Top 30) : {avg_rec:.0f} days",
        f"  Least recent (of Top 30) : {max_rec:.0f} days ago",
    ])


# ── CBF score comparison helpers ──────────────────────────────────────────────

def _cbf_comparison_static(merged: pd.DataFrame, score_col: str, score_label: str) -> None:
    """Static insights for the engineered score vs CBF score scatter."""
    corr        = merged[["score", score_col]].corr().iloc[0, 1]
    both_high   = int(((merged["score"] >= 4.0) & (merged[score_col] >= 0.7)).sum())
    eng_hi_cbf_lo = int(((merged["score"] >= 4.0) & (merged[score_col] < 0.3)).sum())
    cbf_hi_eng_lo = int(((merged[score_col] >= 0.7) & (merged["score"] < 2.0)).sum())
    total       = len(merged)

    insights = [
        f"**Products with both scores:** {human_int(total)}",
        f"**Pearson correlation (engineered vs {score_label}):** {corr:.3f}",
        f"**Agree on high quality** (eng ≥4.0 & CBF ≥0.7): {both_high} products",
        f"**Engineered high, CBF low** (eng ≥4.0 & CBF <0.3): {eng_hi_cbf_lo} products",
        f"**CBF high, Engineered low** (CBF ≥0.7 & eng <2.0): {cbf_hi_eng_lo} products",
    ]

    interpretations = []
    if abs(corr) >= 0.5:
        interpretations.append(
            f"**Strong agreement ({corr:.2f})** — engineered formula and CBF content scoring "
            "rank products similarly; content richness correlates with purchase behaviour."
        )
    elif abs(corr) >= 0.25:
        interpretations.append(
            f"**Moderate agreement ({corr:.2f})** — the two scores overlap but diverge for "
            "some products; each captures something the other misses."
        )
    else:
        interpretations.append(
            f"**Weak agreement ({corr:.2f})** — engineered popularity and content richness are "
            "independent signals; combining them could improve recommendation quality."
        )
    if both_high > 0:
        interpretations.append(
            f"**{both_high} consensus winners** — these products rank highly on both behavioural "
            "and content signals; they are the safest items to promote."
        )
    if eng_hi_cbf_lo > 0:
        interpretations.append(
            f"**{eng_hi_cbf_lo} popularity-only leaders** — high purchase volume but thin content; "
            "improving descriptions could make CBF recommend them more often."
        )
    if cbf_hi_eng_lo > 0:
        interpretations.append(
            f"**{cbf_hi_eng_lo} content-rich hidden gems** — strong text descriptions but low "
            "purchase score; these may be under-promoted despite being a good fit for CBF users."
        )

    recommendations = []
    recommendations.append(
        "**Promote consensus winners** — products scoring high on both signals are "
        "your most reliable recommendations; use them as hero items."
    )
    if eng_hi_cbf_lo > 0:
        recommendations.append(
            f"**Enrich {eng_hi_cbf_lo} popular-but-sparse products** — better titles and "
            "descriptions will let the CBF pipeline surface them to more users."
        )
    if cbf_hi_eng_lo > 0:
        recommendations.append(
            f"**Investigate {cbf_hi_eng_lo} hidden gems** — strong CBF score but low purchase "
            "count; these may benefit from a visibility boost or targeted promotion."
        )
    recommendations.append(
        "**Blend signals** — consider a combined score (e.g. 0.6 × engineered + 0.4 × CBF) "
        "for a more robust product ranking that uses both behavioural and content evidence."
    )

    _render_static_insights(insights, interpretations, recommendations)


def _build_cbf_comparison_context(merged: pd.DataFrame, score_col: str, score_label: str) -> str:
    corr          = merged[["score", score_col]].corr().iloc[0, 1]
    both_high     = int(((merged["score"] >= 4.0) & (merged[score_col] >= 0.7)).sum())
    eng_hi_cbf_lo = int(((merged["score"] >= 4.0) & (merged[score_col] < 0.3)).sum())
    cbf_hi_eng_lo = int(((merged[score_col] >= 0.7) & (merged["score"] < 2.0)).sum())
    top_consensus = (
        merged[(merged["score"] >= 4.0) & (merged[score_col] >= 0.7)]
        .sort_values("score", ascending=False)
        .head(5)
    )
    lines = [
        f"ENGINEERED SCORE vs {score_label.upper()} COMPARISON", "",
        f"  Products compared         : {human_int(len(merged))}",
        f"  Pearson correlation       : {corr:.4f}",
        f"  Agree on high quality     : {both_high}  (eng≥4.0 & CBF≥0.7)",
        f"  Engineered high, CBF low  : {eng_hi_cbf_lo}  (eng≥4.0 & CBF<0.3)",
        f"  CBF high, Engineered low  : {cbf_hi_eng_lo}  (CBF≥0.7 & eng<2.0)", "",
        "TOP CONSENSUS PRODUCTS (high on both):",
    ]
    for _, row in top_consensus.iterrows():
        lines.append(
            f"  {str(row.get('title',''))[:50]:50s}  "
            f"eng={row['score']:.3f}  cbf={row[score_col]:.4f}"
        )
    return "\n".join(lines)
 
 
# ── Tab entry point ───────────────────────────────────────────────────────────
 
def show_feature_tab(
    reviews: pd.DataFrame,
    products: pd.DataFrame,
    features_df: pd.DataFrame,
) -> None:
 
    # ── Section 1: Metadata completeness ─────────────────────────────────────
    st.markdown("## Feature Engineering Overview")
    st.markdown("### Product Metadata Completeness")
 
    importance_df = product_feature_importance(products)
    for _, row in importance_df.iterrows():
        _progress_row(row["feature"], row["value"])
 
    _metadata_completeness_static(importance_df, len(products))
    show_llm_insights(
        context   = _build_metadata_context(importance_df, len(products)),
        cache_key = "feature_metadata",
        title     = "Metadata Completeness Analysis",
        chart_type= "progress bar view showing % completeness for each product metadata field",
    )
 
    st.divider()
 
    # ── Section 2: Feature quality KPIs ──────────────────────────────────────
    st.header("Engineered Product Features")
 
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Avg Popularity Score",   f"{features_df['score'].mean():.2f}")
    c2.metric("Avg Purchase Frequency", f"{features_df['purchase_frequency'].mean():.1f}")
    c3.metric("Avg Unique Users",       f"{features_df['unique_users'].mean():.1f}")
    c4.metric("Avg Product Rating",     f"{features_df['avg_rating'].mean():.2f}")
 
    _feature_quality_static(features_df)
    show_llm_insights(
        context   = _build_feature_quality_context(features_df),
        cache_key = "feature_quality",
        title     = "Feature Quality Analysis",
        chart_type= "KPI metric cards showing average popularity score, purchase frequency, unique users, and product rating across all engineered features",
    )
 
    st.divider()
 
    # ── Section 3: Top products by score ─────────────────────────────────────
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
 
    _top_products_static(top)
    show_llm_insights(
        context   = _build_top_products_context(top),
        cache_key = "feature_top_products",
        title     = "Top Products Analysis",
        chart_type= "horizontal bar chart showing the top 20 products ranked by engineered popularity score (0–5)",
    )

    # ── Section 3b: CBF score comparison ─────────────────────────────────────
    st.divider()
    st.subheader("Engineered Score vs CBF Content Score")
    st.caption(
        "Compares the hand-crafted popularity formula (purchase frequency + ratings) "
        "against the CBF pipeline's content-based scores. "
        "Products in the top-right quadrant are strong on both signals — your safest recommendations."
    )

    cbf_scores = _load_cbf_scores()
    has_tfidf  = "tfidf_popularity_score" in cbf_scores.columns
    has_dl     = "dl_content_score"       in cbf_scores.columns

    if not (has_tfidf or has_dl):
        st.info(
            f"CBF scores not found at `{_CBF_PATH}`. "
            "Run the CBF pipeline (`cbf_and_significance.py`) to unlock this comparison."
        )
    else:
        # Pick the best available score
        score_col   = "dl_content_score"       if has_dl    else "tfidf_popularity_score"
        score_label = "DL Content Score"        if has_dl    else "TF-IDF Content Score"

        # Toggle between scores when both are present
        if has_tfidf and has_dl:
            choice = st.radio(
                "CBF score to compare against",
                ["DL Content Score (neural)", "TF-IDF Content Score (classic)"],
                horizontal=True,
                key="fe_cbf_score_choice",
            )
            if choice.startswith("TF-IDF"):
                score_col, score_label = "tfidf_popularity_score", "TF-IDF Content Score"

        # Merge engineered features with CBF scores
        merged = (
            features_df
            .merge(products[["parent_asin", "title"]], on="parent_asin", how="left")
            .merge(cbf_scores[["parent_asin", score_col]], on="parent_asin", how="inner")
            .dropna(subset=["score", score_col])
        )

        if len(merged) == 0:
            st.warning("No products matched between the feature set and CBF scores — check that `parent_asin` values align.")
        else:
            fig_cbf = px.scatter(
                merged.head(1000),
                x=score_col,
                y="score",
                hover_data=[c for c in ["title", "avg_rating", "purchase_frequency",
                                        "unique_users", "tfidf_popularity_score",
                                        "dl_content_score"]
                            if c in merged.columns],
                labels={
                    score_col: f"{score_label} (0–1, CBF pipeline)",
                    "score":   "Engineered Popularity Score (0–5)",
                },
                title=f"Engineered Popularity Score vs {score_label}",
                opacity=0.55,
                color="avg_rating" if "avg_rating" in merged.columns else None,
                color_continuous_scale="RdYlGn",
            )
            # Quadrant reference lines
            fig_cbf.add_hline(y=4.0,  line_dash="dot", line_color="grey",
                              annotation_text="eng ≥ 4.0", annotation_position="right")
            fig_cbf.add_vline(x=0.7,  line_dash="dot", line_color="grey",
                              annotation_text="CBF ≥ 0.7", annotation_position="top")
            fig_cbf.update_layout(height=480)
            st.plotly_chart(fig_cbf, use_container_width=True)

            _cbf_comparison_static(merged, score_col, score_label)
            show_llm_insights(
                context   = _build_cbf_comparison_context(merged, score_col, score_label),
                cache_key = f"feature_cbf_comparison_{score_col}",
                title     = f"Engineered Score vs {score_label} Analysis",
                chart_type= (
                    f"scatter plot comparing the hand-crafted engineered popularity score (y-axis, 0–5) "
                    f"against the CBF {score_label} (x-axis, 0–1). "
                    "Quadrant lines at eng=4.0 and CBF=0.7 highlight consensus winners (top-right), "
                    "hidden gems (top-left), and content-rich under-performers (bottom-right)."
                ),
            )

    # ── Section 4: Recency ────────────────────────────────────────────────────
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
 
    _recency_static(recency)
    show_llm_insights(
        context   = _build_recency_context(recency),
        cache_key = "feature_recency",
        title     = "Purchase Recency Analysis",
        chart_type= "horizontal bar chart showing the 30 most recently purchased products sorted by days since last purchase (shortest bar = most recent)",
    )
 
    # ── Section 5: Raw table ──────────────────────────────────────────────────
    st.subheader("Engineered Features Table")
    st.dataframe(
        top[[
            "parent_asin", "title", "purchase_count", "unique_users",
            "avg_rating", "purchase_frequency", "days_since_last_purchase", "score",
        ]],
        use_container_width=True,
    )