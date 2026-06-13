from __future__ import annotations
 
from typing import Optional
 
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
 
from config import SCATTER_FILE, TOP_ORDER
from data_loader import load_scatter
from utils import human_int
from llm_insights import show_llm_insights
 
 
# ── Chart builder ─────────────────────────────────────────────────────────────
 
def _build_scatter(df: pd.DataFrame) -> go.Figure:
    top  = df[df["Group"].isin(TOP_ORDER)].copy()
    near = df[df["Group"] == "Near"]
    far  = df[df["Group"] == "Far"]
    rand = df[df["Group"] == "Random"]
 
    top["_order"] = top["Group"].map({g: i for i, g in enumerate(TOP_ORDER)})
    top = top.sort_values("_order")
 
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=rand["MaxCosine"], y=rand["Predicted_Rating"],
        mode="markers", name="All",
        marker=dict(size=6, color="rgba(120,120,120,0.25)"),
        hoverinfo="skip",
    ))
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
    fig.add_trace(go.Scatter(
        x=top["MaxCosine"], y=top["Predicted_Rating"],
        mode="markers", name="Top glow",
        marker=dict(size=26, color="rgba(59,130,246,0.22)"),
        hoverinfo="skip", showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=top["MaxCosine"], y=top["Predicted_Rating"],
        mode="lines+markers+text",
        text=top["DisplayLabel"], textposition="top center",
        name="Top 5",
        line=dict(color="#3b82f6", width=3),
        marker=dict(size=14, color="#3b82f6"),
    ))
    fig.update_layout(
        title="Recommendation Scatter Plot", height=650,
        xaxis_title="Cosine Similarity", yaxis_title="Predicted Rating",
    )
    return fig
 
 
# ── ASIN resolution ───────────────────────────────────────────────────────────
 
def _find_asin_by_title(label: str, products_lookup: pd.DataFrame) -> Optional[str]:
    """
    Find parent_asin by matching DisplayLabel against product titles.
    Tries three progressively looser strategies.
    """
    label_clean = str(label).lower().strip()
 
    # Strategy 1: exact title match
    mask = products_lookup["title"].str.lower().str.strip() == label_clean
    if mask.any():
        return str(products_lookup[mask].iloc[0]["parent_asin"])
 
    # Strategy 2: first 25 chars of label appears in title
    prefix = label_clean[:25]
    if len(prefix) > 5:
        mask = products_lookup["title"].str.lower().str.contains(
            prefix, na=False, regex=False
        )
        if mask.any():
            return str(products_lookup[mask].iloc[0]["parent_asin"])
 
    # Strategy 3: first 25 chars of title appears in label
    for _, prow in products_lookup.iterrows():
        title_prefix = str(prow["title"]).lower()[:25]
        if len(title_prefix) > 5 and title_prefix in label_clean:
            return str(prow["parent_asin"])
 
    return None
 
 
def _resolve_asin(row: pd.Series, asin_col: Optional[str],
                  products_lookup: pd.DataFrame) -> Optional[str]:
    """Return ASIN from column if present, else fall back to title matching."""
    if asin_col and pd.notna(row.get(asin_col)):
        return str(row[asin_col]).strip()
    label = str(row.get("DisplayLabel", ""))
    return _find_asin_by_title(label, products_lookup)
 
 
# ── User profile ──────────────────────────────────────────────────────────────
 
def _get_user_profile(user_id: str, reviews: pd.DataFrame,
                      products_lookup: pd.DataFrame) -> dict:
    user_rev = reviews[reviews["user_id"] == user_id].copy()
    profile  = {"found": False, "total_reviews": 0}
    if user_rev.empty:
        return profile
 
    profile["found"]            = True
    profile["total_reviews"]    = len(user_rev)
    profile["avg_rating_given"] = user_rev["rating"].mean()
    profile["pct_5star"]        = (user_rev["rating"] == 5).mean() * 100
    profile["pct_1star"]        = (user_rev["rating"] == 1).mean() * 100
    profile["verified_pct"]     = user_rev["verified_purchase"].mean() * 100
    profile["avg_words"]        = user_rev["review_length_words"].mean()
    profile["disliked_count"]   = int((user_rev["rating"] <= 2).sum())
 
    top_rated = (
        user_rev[user_rev["rating"] >= 4]
        .merge(products_lookup[["parent_asin", "title"]], on="parent_asin", how="left")
        .sort_values("rating", ascending=False)
        .head(5)
    )
    profile["top_rated"] = [
        {"title": str(r.get("title", r["parent_asin"]))[:60], "rating": int(r["rating"])}
        for _, r in top_rated.iterrows()
    ]
    return profile
 
 
# ── Product review stats ──────────────────────────────────────────────────────
 
def _get_product_stats(asin: str, reviews: pd.DataFrame,
                       products_lookup: pd.DataFrame) -> dict:
    prod_rev  = reviews[reviews["parent_asin"] == asin]
    prod_info = products_lookup[products_lookup["parent_asin"] == asin]
    stats: dict = {"asin": asin, "found": False}
 
    if not prod_info.empty:
        r = prod_info.iloc[0]
        stats.update({
            "found":        True,
            "title":        str(r.get("title", "")),
            "avg_rating":   r.get("average_rating"),
            "rating_count": r.get("rating_number", 0),
            "store":        r.get("store_clean", ""),
            "price":        r.get("price"),
        })
 
    if not prod_rev.empty:
        stats.update({
            "found":             True,
            "review_count":      len(prod_rev),
            "actual_avg_rating": prod_rev["rating"].mean(),
            "pct_5star":         (prod_rev["rating"] == 5).mean() * 100,
            "pct_1star":         (prod_rev["rating"] == 1).mean() * 100,
            "pct_positive":      (prod_rev["rating"] >= 4).mean() * 100,
            "pct_negative":      (prod_rev["rating"] <= 2).mean() * 100,
            "verified_pct":      prod_rev["verified_purchase"].mean() * 100,
            "helpful_votes":     int(prod_rev["helpful_vote"].sum()),
            "avg_words":         prod_rev["review_length_words"].mean(),
        })
 
    return stats
 
 
# ── Context builder ───────────────────────────────────────────────────────────
 
def _build_recommendation_context(
    top: pd.DataFrame,
    user_id: str,
    reviews: pd.DataFrame,
    products_lookup: pd.DataFrame,
) -> str:
    profile  = _get_user_profile(user_id, reviews, products_lookup)
    asin_col = next((c for c in top.columns if "asin" in c.lower()), None)
    lines    = [
        f"PERSONALISED PRODUCT RECOMMENDATION REPORT — USER: {user_id}",
        "=" * 60, "",
    ]
 
    # ── User profile section ──────────────────────────────────────────────────
    if profile["found"]:
        lines += [
            "USER PREFERENCE PROFILE:",
            f"  Total reviews written       : {profile['total_reviews']}",
            f"  Average rating they give    : {profile['avg_rating_given']:.2f} ⭐",
            f"  % 5-star ratings given      : {profile['pct_5star']:.1f}%",
            f"  % 1-star ratings given      : {profile['pct_1star']:.1f}%",
            f"  Verified purchase rate      : {profile['verified_pct']:.1f}%",
            f"  Avg review length (words)   : {profile['avg_words']:.0f}",
            f"  Products disliked (≤2★)     : {profile['disliked_count']}",
        ]
        if profile.get("top_rated"):
            lines.append("  Products they rated highly  :")
            for p in profile["top_rated"]:
                lines.append(f"    {p['rating']}★  {p['title']}")
    else:
        lines.append("USER PREFERENCE PROFILE: No review history found for this user.")
    lines.append("")
 
    # ── Per-product sections ──────────────────────────────────────────────────
    lines += ["TOP 5 RECOMMENDED PRODUCTS — REVIEW & RATING DATA:", "-" * 60, ""]
 
    matched_any = False
 
    for rank, (_, row) in enumerate(top.iterrows(), start=1):
        label = str(row.get("DisplayLabel", f"Product {rank}"))
        sim   = float(row.get("MaxCosine", 0))
        pred  = float(row.get("Predicted_Rating", 0))
 
        lines.append(f"PRODUCT #{rank}: {label}")
        lines.append(f"  Cosine Similarity (taste match): {sim:.4f}")
        lines.append(f"  Predicted Rating  (model)      : {pred:.2f} ⭐")
 
        asin  = _resolve_asin(row, asin_col, products_lookup)
        stats = _get_product_stats(asin, reviews, products_lookup) if asin else {"found": False}
 
        if stats["found"]:
            matched_any = True
            if stats.get("avg_rating") is not None:
                lines.append(
                    f"  Catalogue Avg Rating           : {stats['avg_rating']:.2f} ⭐"
                    f"  ({human_int(stats.get('rating_count', 0))} ratings)"
                )
            if stats.get("actual_avg_rating") is not None:
                actual = stats["actual_avg_rating"]
                diff   = pred - actual
                tag    = ("over-estimated" if diff > 0.3
                          else "under-estimated" if diff < -0.3 else "accurate")
                lines.append(f"  Actual Avg Rating (reviews)    : {actual:.2f} ⭐  → model is {tag} ({diff:+.2f})")
            if stats.get("review_count"):
                lines.append(f"  Number of Reviews              : {human_int(stats['review_count'])}")
            if stats.get("pct_5star") is not None:
                lines.append(f"  5★ reviews                     : {stats['pct_5star']:.1f}%")
            if stats.get("pct_1star") is not None:
                lines.append(f"  1★ reviews                     : {stats['pct_1star']:.1f}%")
            if stats.get("pct_positive") is not None:
                lines.append(f"  Positive (4–5★)                : {stats['pct_positive']:.1f}%")
            if stats.get("pct_negative") is not None:
                lines.append(f"  Negative (1–2★)                : {stats['pct_negative']:.1f}%")
            if stats.get("verified_pct") is not None:
                lines.append(f"  Verified Purchases             : {stats['verified_pct']:.1f}%")
            if stats.get("helpful_votes"):
                lines.append(f"  Total Helpful Votes            : {human_int(stats['helpful_votes'])}")
            if stats.get("store") and stats["store"] not in ("", "(missing store)"):
                lines.append(f"  Seller                         : {stats['store']}")
            if stats.get("price") and pd.notna(stats["price"]) and float(stats["price"]) > 0:
                lines.append(f"  Price                          : ${float(stats['price']):.2f}")
        else:
            lines.append(f"  Review data                    : not found in dataset")
            lines.append(f"  (only model score available)")
 
        lines.append("")
 
    if not matched_any:
        lines += [
            "NOTE: No products could be matched to the reviews dataset.",
            "The scatter file's DisplayLabel values do not match product titles.",
            "The LLM should base its analysis on the model scores alone",
            "and clearly state that actual review data was unavailable.",
            "",
        ]
 
    # ── Instructions ─────────────────────────────────────────────────────────
    lines += [
        "=" * 60,
        "YOUR TASK — respond with exactly three sections:",
        "",
        "**Insights**",
        "  One bullet per product covering: actual rating (if available),",
        "  review volume, positive/negative split, and model accuracy.",
        "  If no review data: note this and use model score only.",
        "",
        "**Interpretation**",
        "  Based on this user's preference profile, explain which products",
        "  best match their demonstrated taste. Call out any risky picks.",
        "",
        "**Recommendations**",
        "  Rank #1 to #5 using EXACTLY this markdown format (blank line between each):",
        "  ",
        "  **#1 [Product Name]**",
        "  Why: [one sentence — model score + actual rating + user taste fit]",
        "  ",
        "  **#2 [Product Name]**",
        "  Why: [one sentence]",
        "  ",
        "  **#3 [Product Name]**",
        "  Why: [one sentence]",
        "  ",
        "  **#4 [Product Name]**",
        "  Why: [one sentence]",
        "  ",
        "  **#5 [Product Name]**",
        "  Why: [one sentence]",
        "  ",
        "  ---",
        "  🏆 **Top pick: [Product Name]** — [key reason in one sentence]",
    ]
 
    return "\n".join(lines)
 
 
def _build_system_context(plot_df: pd.DataFrame) -> str:
    avg_sim = plot_df["MaxCosine"].mean()
    med_sim = plot_df["MaxCosine"].median()
    avg_rat = plot_df["Predicted_Rating"].mean()
    total   = len(plot_df)
    high_q  = int((plot_df["MaxCosine"] >= 0.6).sum())
    med_q   = int(((plot_df["MaxCosine"] >= 0.4) & (plot_df["MaxCosine"] < 0.6)).sum())
    low_q   = int((plot_df["MaxCosine"] < 0.4).sum())
    near_c  = int((plot_df["Group"] == "Near").sum())
    far_c   = int((plot_df["Group"] == "Far").sum())
    return "\n".join([
        "OVERALL RECOMMENDATION SYSTEM ANALYSIS", "",
        f"  Total candidate products       : {human_int(total)}",
        f"  Average cosine similarity      : {avg_sim:.4f}  (median: {med_sim:.4f})",
        f"  Average predicted rating       : {avg_rat:.2f} ⭐", "",
        "QUALITY DISTRIBUTION:",
        f"  High quality  (≥0.6 similarity): {high_q} ({high_q/total*100:.1f}%)",
        f"  Medium quality (0.4–0.6)       : {med_q} ({med_q/total*100:.1f}%)",
        f"  Low quality   (<0.4 similarity): {low_q} ({low_q/total*100:.1f}%)", "",
        "CLUSTER COUNTS:",
        f"  Near matches  : {near_c}",
        f"  Far mismatches: {far_c}",
        f"  Random/other  : {total - near_c - far_c}",
    ])
 
 
# ── Tab entry point ───────────────────────────────────────────────────────────
 
def show_scatter_tab(
    reviews: pd.DataFrame,
    products_lookup: pd.DataFrame,
) -> None:
    st.header("Product Recommendation Scatter Plot")
 
    df: Optional[pd.DataFrame] = load_scatter()
    if df is None:
        st.warning(f"Scatter data not found at `{SCATTER_FILE}`.")
        return
 
    # Debug expander — shows what columns the scatter file has
    with st.expander("🔍 Scatter file columns (debug)", expanded=False):
        st.write(list(df.columns))
        st.caption(
            "If you see an ASIN/product ID column above, make sure it contains "
            "the word 'asin' (case-insensitive) so it is picked up automatically. "
            "Otherwise title-matching is used as a fallback."
        )
 
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
                "DisplayLabel":     "Product",
                "MaxCosine":        "Cosine Similarity",
                "Predicted_Rating": "Predicted Rating",
            })
            .assign(**{
                "Cosine Similarity": lambda d: d["Cosine Similarity"].round(3),
                "Predicted Rating":  lambda d: d["Predicted Rating"].round(2),
            }),
            use_container_width=True, hide_index=True, height=220,
        )
 
    # ── Scatter plot ──────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.plotly_chart(_build_scatter(plot_df), use_container_width=True)
 
    # ── LLM blocks BELOW chart ────────────────────────────────────────────────
    if not top.empty:
        show_llm_insights(
            context   = _build_recommendation_context(
                top, user_id.strip(), reviews, products_lookup),
            cache_key = f"scatter_top5_{user_id}",
            title     = "Top 5 Recommendations & User Preference Analysis",
            user_id   = user_id.strip(),
        )
 
    show_llm_insights(
        context   = _build_system_context(plot_df),
        cache_key = f"scatter_system_{user_id}",
        title     = "Recommendation System Analysis",
        user_id   = user_id.strip(),
    )