from __future__ import annotations
 
import pandas as pd
import plotly.express as px
import streamlit as st
from pathlib import Path
 
from utils import style_bar_chart, shorten, human_int
from llm_insights import show_llm_insights
 
 
# ── Helpers ───────────────────────────────────────────────────────────────────
 
def _norm(series: pd.Series) -> pd.Series:
    mn, mx = series.min(), series.max()
    if mx == mn:
        return pd.Series([0.5] * len(series), index=series.index)
    return (series - mn) / (mx - mn)
 
 
def _make_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    rating_col = "average_rating" if "average_rating" in df.columns else None
    store_col  = "store_clean"    if "store_clean"    in df.columns else "store" if "store" in df.columns else None
 
    def _label(row):
        title = shorten(str(row.get("title", row.get("parent_asin", ""))), 40)
        if rating_col and pd.notna(row.get(rating_col)):
            title = f"{title} ⭐ {float(row[rating_col]):.1f}"
        if store_col and pd.notna(row.get(store_col)) and str(row.get(store_col, "")) not in ("", "(missing store)", "nan"):
            title = f"{title} ({row[store_col]})"
        return title
 
    df["short_label"] = df.apply(_label, axis=1)
    return df
 
 
# ── Score computation ─────────────────────────────────────────────────────────
 
def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["rating_number", "average_rating", "purchase_frequency"]:
        df[col] = pd.to_numeric(df.get(col, 0), errors="coerce").fillna(0)
    return df
 
 
def _compute_popular_score(products: pd.DataFrame) -> pd.DataFrame:
    """
    Popular Score = 0.50 × Popularity + 0.30 × Customer Satisfaction + 0.20 × Trending
      Popularity           → purchase_frequency  (normalised)
      Customer Satisfaction→ average_rating       (normalised)
      Trending             → rating_number        (normalised)
    """
    df = _prepare(products)
    pop_norm   = _norm(df["purchase_frequency"])
    sat_norm   = _norm(df["average_rating"])
    trend_norm = _norm(df["rating_number"])
 
    df["popular_score"] = ((0.50 * pop_norm + 0.30 * sat_norm + 0.20 * trend_norm) * 5).round(3)
    df["_pop_norm"]     = pop_norm
    return df
 
 
def _compute_discovery_score(products: pd.DataFrame) -> pd.DataFrame:
    """
    Discovery Score = 0.40 × Customer Satisfaction + 0.30 × Trending + 0.30 × Low Exposure
      Low Exposure = 1 − normalised_popularity
    Only products that pass the quality gate (min rating + min reviews) are eligible.
    """
    df = _prepare(products)
    pop_norm   = _norm(df["purchase_frequency"])
    sat_norm   = _norm(df["average_rating"])
    trend_norm = _norm(df["rating_number"])
    low_exp    = 1 - pop_norm
 
    df["discovery_score"] = ((0.40 * sat_norm + 0.30 * trend_norm + 0.30 * low_exp) * 5).round(3)
    df["_low_exposure"]   = low_exp
    df["_pop_norm"]       = pop_norm
    return df
 
 
def _split_buckets(
    products: pd.DataFrame,
    k: int,
    d: float,
    min_disc_rating: float,
    min_disc_reviews: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    popular_k   = max(1, round(k * (1 - d)))
    discovery_k = k - popular_k
 
    pop_df  = _compute_popular_score(products)
    disc_df = _compute_discovery_score(products)
 
    top_popular = pop_df.sort_values("popular_score", ascending=False).head(popular_k).copy()
    top_popular["rec_type"] = "Popular"
 
    popular_ids = set(top_popular.index)
 
    qualified = disc_df[
        (~disc_df.index.isin(popular_ids)) &
        (disc_df["average_rating"] >= min_disc_rating) &
        (disc_df["rating_number"]  >= min_disc_reviews)
    ]
    top_discovery = qualified.sort_values("discovery_score", ascending=False).head(discovery_k).copy()
    top_discovery["rec_type"] = "Discovery"
 
    return top_popular, top_discovery
 
 
# ── Section renderers ─────────────────────────────────────────────────────────
 
def _render_popular_section(top_popular: pd.DataFrame) -> None:
    st.subheader(f" Popular Products — {len(top_popular)} slots")
    st.caption(
        "**Popular Score** = 50% Popularity + 30% Customer Satisfaction + 20% Trending  \n"
        "These are proven, safe recommendations with strong customer interaction."
    )
 
    df = _make_labels(top_popular)
    fig = px.bar(
        df.sort_values("popular_score"),
        x="popular_score", y="short_label", orientation="h",
        hover_data=[c for c in ["average_rating", "purchase_frequency", "rating_number"] if c in df.columns],
        title="Popular Score (0–5)",
        color="popular_score",
        color_continuous_scale="Teal",
    )
    fig.update_layout(height=max(400, len(df) * 40), yaxis_title="", coloraxis_showscale=False)
    fig.update_xaxes(range=[0, 5])
    st.plotly_chart(style_bar_chart(fig), use_container_width=True)
 
    # Insight metrics
    top1 = top_popular.iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("#1 Score",      f"{top1['popular_score']:.2f} / 5")
    c2.metric("Avg Score",     f"{top_popular['popular_score'].mean():.2f}")
    c3.metric("Avg Rating ★",  f"{top_popular['average_rating'].mean():.2f}" if "average_rating" in top_popular.columns else "N/A")
    high_q = (top_popular["average_rating"] >= 4.5).sum() if "average_rating" in top_popular.columns else 0
    c4.metric("Products ★4.5+", str(high_q))
 
    # Table + export
    st.markdown("**Popular Products Table**")
    table_cols = [c for c in ["parent_asin", "title", "average_rating", "rating_number", "purchase_frequency", "popular_score"] if c in top_popular.columns]
    st.dataframe(top_popular[table_cols].reset_index(drop=True), use_container_width=True, hide_index=True)
 
    csv = top_popular[table_cols].to_csv(index=False).encode("utf-8")
    st.download_button(
        label=f"⬇ Export {len(top_popular)} Popular Products as CSV",
        data=csv,
        file_name=f"popular_products_top{len(top_popular)}.csv",
        mime="text/csv",
        key="dl_popular",
    )
 
 
def _render_discovery_section(top_discovery: pd.DataFrame, min_disc_rating: float, min_disc_reviews: int) -> None:
    st.subheader(f" Discovery Products — {len(top_discovery)} slots")
    st.caption(
        "**Discovery Score** = 40% Customer Satisfaction + 30% Trending + 30% Low Exposure  \n"
        f"Qualified gate: avg rating ≥ {min_disc_rating:.1f}★ and ≥ {min_disc_reviews} reviews.  \n"
        "These are *under-exposed but promising* products — not just unpopular ones."
    )
 
    if top_discovery.empty:
        st.warning(
            "⚠️ No products passed the discovery qualification gate with the current thresholds. "
            "Try lowering the minimum rating or minimum reviews sliders."
        )
        return
 
    df = _make_labels(top_discovery)
    fig = px.bar(
        df.sort_values("discovery_score"),
        x="discovery_score", y="short_label", orientation="h",
        hover_data=[c for c in ["average_rating", "rating_number", "purchase_frequency", "_low_exposure"] if c in df.columns],
        title="Discovery Score (0–5)",
        color="discovery_score",
        color_continuous_scale="Sunset",
    )
    fig.update_layout(height=max(400, len(df) * 40), yaxis_title="", coloraxis_showscale=False)
    fig.update_xaxes(range=[0, 5])
    st.plotly_chart(style_bar_chart(fig), use_container_width=True)
 
    top1 = top_discovery.iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("#1 Score",        f"{top1['discovery_score']:.2f} / 5")
    c2.metric("Avg Score",       f"{top_discovery['discovery_score'].mean():.2f}")
    c3.metric("Avg Rating ★",    f"{top_discovery['average_rating'].mean():.2f}" if "average_rating" in top_discovery.columns else "N/A")
    avg_low_exp = top_discovery["_low_exposure"].mean() if "_low_exposure" in top_discovery.columns else None
    c4.metric("Avg Low Exposure", f"{avg_low_exp:.2f}" if avg_low_exp is not None else "N/A",
              help="1.0 = completely unexposed; higher = more novel")
 
 
def _render_blended_table(top_popular: pd.DataFrame, top_discovery: pd.DataFrame, k: int) -> None:
    st.subheader(" Unified Blended Recommendation List")
    st.caption("Popular products first, then Discovery products. Discovery rows are highlighted.")
 
    top_popular   = top_popular.copy()
    top_discovery = top_discovery.copy() if not top_discovery.empty else top_discovery
 
    top_popular["slot_rank"]   = range(1, len(top_popular) + 1)
    if not top_discovery.empty:
        top_discovery["slot_rank"] = range(len(top_popular) + 1, len(top_popular) + len(top_discovery) + 1)
 
    blended = pd.concat([top_popular, top_discovery], ignore_index=True) if not top_discovery.empty else top_popular.copy()
 
    table_cols = [c for c in [
        "slot_rank", "rec_type", "parent_asin", "title",
        "average_rating", "rating_number", "purchase_frequency",
        "popular_score", "discovery_score",
    ] if c in blended.columns]
 
    def _highlight(row):
        if row.get("rec_type") == "Discovery":
            return ["background-color: #fff7e6"] * len(row)
        return [""] * len(row)
 
    st.dataframe(
        blended[table_cols].style.apply(_highlight, axis=1),
        use_container_width=True,
        hide_index=True,
    )
 
    csv = blended[table_cols].to_csv(index=False).encode("utf-8")
    st.download_button(
        label=f"⬇ Export Full Blended List (Top {k}) as CSV",
        data=csv,
        file_name=f"blended_recommendations_top{k}.csv",
        mime="text/csv",
        key="dl_blended",
    )
 
 
# ── Main tab ──────────────────────────────────────────────────────────────────
 
def show_popularity_tab(products: pd.DataFrame) -> None:
    st.header("Popular + Discovery Products")
    st.caption(
        "The baseline recommender covers both proven popular products and under-exposed discovery "
        "products — improving catalogue coverage and product discoverability."
    )
 
    st.info(
        "**Baseline Formula:**  `Final@K = (1 − d) × Popular + d × Discovery`\n\n"
        "**Popular Score** = 50% Popularity + 30% Customer Satisfaction + 20% Trending\n\n"
        "**Discovery Score** = 40% Customer Satisfaction + 30% Trending + 30% Low Exposure  \n"
        "*(Low Exposure = 1 − normalised popularity)*"
    )
 
    # ── Controls ──────────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        k = st.slider("Total recommendations (K)", 5, 50, 20, key="blend_k")
    with c2:
        d = st.slider("Discovery share (d %)", 0, 50, 20, key="blend_d") / 100
    with c3:
        min_disc_rating = st.slider("Min discovery rating ★", 1.0, 5.0, 3.5, 0.5, key="disc_min_rating")
    with c4:
        min_disc_reviews = st.slider("Min discovery reviews", 1, 50, 5, key="disc_min_reviews")
 
    popular_k   = max(1, round(k * (1 - d)))
    discovery_k = k - popular_k
 
    st.caption(f"Slots: **{popular_k} Popular** + **{discovery_k} Discovery** = {k} total  |  d = {d:.0%}")
    st.divider()
 
    top_popular, top_discovery = _split_buckets(
        products, k=k, d=d,
        min_disc_rating=min_disc_rating,
        min_disc_reviews=min_disc_reviews,
    )
 
    # ── Side-by-side charts ───────────────────────────────────────────────────
    col_left, col_right = st.columns(2)
    with col_left:
        _render_popular_section(top_popular)
    with col_right:
        _render_discovery_section(top_discovery, min_disc_rating, min_disc_reviews)
 
    st.divider()
 
    # ── Full-width Popular table ──────────────────────────────────────────────
    st.subheader(" Popular Products Table")
    st.caption("Use the slider to control how many popular products to show and export — independent of the recommendation K above.")
 
    max_pop = len(_compute_popular_score(products))
    pop_table_n = st.slider(
        "Number of popular products to show",
        10, min(500, max_pop), min(100, max_pop),
        key="pop_table_n",
    )
    pop_full = _compute_popular_score(products).nlargest(pop_table_n, "popular_score")
    pop_table_cols = [c for c in [
        "parent_asin", "title", "average_rating", "rating_number",
        "purchase_frequency", "popular_score",
    ] if c in pop_full.columns]
    st.dataframe(pop_full[pop_table_cols].reset_index(drop=True), use_container_width=True, hide_index=True)
    st.download_button(
        label=f"⬇ Download {pop_table_n} Popular Products as CSV",
        data=pop_full[pop_table_cols].to_csv(index=False).encode("utf-8"),
        file_name=f"popular_products_top{pop_table_n}.csv",
        mime="text/csv",
        key="dl_popular_table",
    )
 
    st.divider()
 
    # ── Full-width Discovery table ────────────────────────────────────────────
    st.subheader(" Discovery (Less Popular) Products Table")
    st.caption(
        "Under-exposed products with good customer signals — qualified by minimum rating and review count. "
        "Use the slider to control how many to show and export — independent of the recommendation K above."
    )
 
    # Compute full discovery pool (not capped by discovery_k)
    disc_full_df = _compute_discovery_score(products)
    pop_ids_full = set(_compute_popular_score(products).nlargest(popular_k, "popular_score").index)
    disc_pool = disc_full_df[
        (~disc_full_df.index.isin(pop_ids_full)) &
        (disc_full_df["average_rating"] >= min_disc_rating) &
        (disc_full_df["rating_number"]  >= min_disc_reviews)
    ].sort_values("discovery_score", ascending=False)
 
    if disc_pool.empty:
        st.warning("No discovery products qualified. Lower the minimum rating or reviews threshold to see results.")
    else:
        disc_table_n = st.slider(
            "Number of discovery products to show",
            10, min(500, len(disc_pool)), min(100, len(disc_pool)),
            key="disc_table_n",
        )
        disc_show = disc_pool.head(disc_table_n)
        disc_table_cols = [c for c in [
            "parent_asin", "title", "average_rating", "rating_number",
            "purchase_frequency", "discovery_score", "_low_exposure",
        ] if c in disc_show.columns]
        disc_display = disc_show[disc_table_cols].rename(
            columns={"_low_exposure": "low_exposure_score"}
        ).reset_index(drop=True)
        st.dataframe(disc_display, use_container_width=True, hide_index=True)
        st.download_button(
            label=f"⬇ Download {disc_table_n} Discovery Products as CSV",
            data=disc_display.to_csv(index=False).encode("utf-8"),
            file_name=f"discovery_products_top{disc_table_n}.csv",
            mime="text/csv",
            key="dl_discovery_table",
        )
 
    st.divider()
 
    # ── Unified blended list ──────────────────────────────────────────────────
    _render_blended_table(top_popular, top_discovery, k)
 
    st.divider()
 
    # ── LLM insights ─────────────────────────────────────────────────────────
    top1_pop  = top_popular.iloc[0]
    top1_disc = top_discovery.iloc[0] if not top_discovery.empty else None
 
    show_llm_insights(
        context=(
            f"Blended recommender: {popular_k} popular + {discovery_k} discovery slots (K={k}, d={d:.0%}). "
            f"Top popular: {shorten(str(top1_pop.get('title','')), 60)} "
            f"(popular_score={top1_pop['popular_score']:.2f}, "
            f"avg_rating={top1_pop.get('average_rating','N/A')}, "
            f"purchase_frequency={top1_pop.get('purchase_frequency','N/A')}). "
            + (
                f"Top discovery: {shorten(str(top1_disc.get('title','')), 60)} "
                f"(discovery_score={top1_disc['discovery_score']:.2f}, "
                f"avg_rating={top1_disc.get('average_rating','N/A')}, "
                f"low_exposure={top1_disc.get('_low_exposure', 'N/A')}). "
                if top1_disc is not None else "No qualifying discovery products. "
            ) +
            f"Popular avg score: {top_popular['popular_score'].mean():.2f}. "
            + (f"Discovery avg score: {top_discovery['discovery_score'].mean():.2f}." if not top_discovery.empty else "")
        ),
        cache_key="blended_popularity",
        title="Popular + Discovery Baseline Analysis",
        chart_type=(
            f"two side-by-side horizontal bar charts: {popular_k} popular products by popular score "
            f"and {discovery_k} discovery products by discovery score, both on a 0-5 scale"
        ),
    )