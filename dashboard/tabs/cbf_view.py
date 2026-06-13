from __future__ import annotations
 
import pandas as pd
import plotly.express as px
import streamlit as st
from pathlib import Path
 
from utils import style_bar_chart, shorten, human_int
from llm_insights import show_llm_insights
 
 
# ── File paths ────────────────────────────────────────────────────────────────
 
_DASHBOARD_DATA = Path("dashboard_data")
_POPULAR_PRODUCTS_CSV = _DASHBOARD_DATA / "popular_products.csv"
 
 
# ── Cache loader ──────────────────────────────────────────────────────────────
 
@st.cache_data(show_spinner=False)
def _load_cbf_popular() -> pd.DataFrame | None:
    if not _POPULAR_PRODUCTS_CSV.exists():
        return None
    return pd.read_csv(_POPULAR_PRODUCTS_CSV)
 
 
# ── Popularity score (original live computation) ──────────────────────────────
#
# Simple weighted score:
#   score = (0.6 × avg_rating_normalised) + (0.4 × purchase_count_normalised)
#
# Both inputs are min-max normalised to [0, 1] then scaled to [0, 5] so the
# final score sits on a familiar 0–5 scale.
# Weights: quality (rating) counts more than volume (purchases) — 60 / 40.
 
def _compute_popularity(products: pd.DataFrame) -> pd.DataFrame:
    df = products.copy()
 
    df["rating_number"]      = pd.to_numeric(df["rating_number"],      errors="coerce").fillna(0)
    df["average_rating"]     = pd.to_numeric(df["average_rating"],     errors="coerce").fillna(0)
    df["purchase_frequency"] = pd.to_numeric(
        df.get("purchase_frequency", 0), errors="coerce"
    ).fillna(0)
 
    def _norm(series: pd.Series) -> pd.Series:
        mn, mx = series.min(), series.max()
        if mx == mn:
            return pd.Series([0.5] * len(series), index=series.index)
        return (series - mn) / (mx - mn)
 
    rating_norm   = _norm(df["average_rating"])
    purchase_norm = _norm(df["purchase_frequency"])
 
    df["popularity_score"] = (
        (0.6 * rating_norm + 0.4 * purchase_norm) * 5
    ).round(3)
 
    return df.sort_values("popularity_score", ascending=False)
 
 
def _make_labels(df: pd.DataFrame, title_col: str = "title") -> pd.DataFrame:
    df = df.copy()
 
    rating_col = "average_rating" if "average_rating" in df.columns else "avg_rating_raw"
    store_col  = "store_clean"    if "store_clean"    in df.columns else "store"
 
    def _label(row):
        title  = shorten(str(row[title_col]), 40)
        rating = row.get(rating_col)
        store  = row.get(store_col, "")
        if pd.notna(rating):
            title = f"{title} ⭐ {float(rating):.1f}"
        if pd.notna(store) and str(store) not in ("", "(missing store)", "nan"):
            title = f"{title} ({store})"
        return title
 
    df["short_label"] = df.apply(_label, axis=1)
    return df
 
 
# ── CBF popular-products view ─────────────────────────────────────────────────
 
def _show_cbf_popular(cbf_df: pd.DataFrame) -> None:
    st.info(
        "**How CBF popularity is scored:**  \n"
        "Score = **Bayesian Rating** blended with **purchase frequency**, **helpful votes**, "
        "and **verified purchase %** — pre-computed by the content-based filtering pipeline.  \n"
        "Rankings reflect product quality signals that are robust to low review counts."
    )
 
    # Normalise column names to consistent aliases
    col_map = {
        "avg_rating_raw":   "average_rating",
        "purchase_freq":    "purchase_frequency",
        "bayesian_rating":  "bayesian_rating",
        "popularity_rank":  "popularity_rank",
        "display_name":     "display_name",
        "short_title":      "short_title",
        "rating_tier":      "rating_tier",
    }
    cbf_df = cbf_df.rename(columns={k: v for k, v in col_map.items() if k in cbf_df.columns})
 
    # Sort by popularity_score descending (already ranked, but make sure)
    cbf_df = cbf_df.sort_values("popularity_score", ascending=False).reset_index(drop=True)
 
    top_n = st.slider("Top N products", 10, min(100, len(cbf_df)), 20, key="cbf_pop_top_n")
    top   = cbf_df.head(top_n).copy()
 
    # Use short_title if available, otherwise title
    label_col = "short_title" if "short_title" in top.columns else "title"
    top = _make_labels(top, title_col=label_col)
 
    # ── Chart ────────────────────────────────────────────────────────────────
    hover_cols = [c for c in ["average_rating", "bayesian_rating", "purchase_frequency",
                               "helpful_votes", "verified_pct", "rating_tier", "store"]
                  if c in top.columns]
 
    fig = px.bar(
        top.sort_values("popularity_score"),
        x="popularity_score", y="short_label", orientation="h",
        hover_data=hover_cols,
        title=f"Top {top_n} Popular Products — CBF Pipeline  (score out of 5)",
        color="popularity_score",
        color_continuous_scale="Teal",
    )
    fig.update_layout(height=500, yaxis_title="", coloraxis_showscale=False)
    fig.update_xaxes(range=[0, 5])
    st.plotly_chart(style_bar_chart(fig), use_container_width=True)
 
    # ── Static insight block ──────────────────────────────────────────────────
    top_product  = top.iloc[0]
    score_gap    = (
        top.iloc[0]["popularity_score"] - top.iloc[1]["popularity_score"]
        if len(top) > 1 else 0
    )
    avg_score    = top["popularity_score"].mean()
    median_score = top["popularity_score"].median()
    avg_rating   = top_product.get("average_rating", None)
    bay_rating   = top_product.get("bayesian_rating", None)
    purch_freq   = top_product.get("purchase_frequency", None)
 
    high_rating_products = (
        (top["average_rating"] >= 4.5).sum() if "average_rating" in top.columns else 0
    )
    high_rating_pct = (high_rating_products / len(top) * 100) if len(top) > 0 else 0
 
    insights = [
        f"**#1 Product:** {shorten(str(top_product.get('title', top_product.get('short_title', ''))), 60)}",
        f"**Popularity Score:** {top_product['popularity_score']:.2f} / 5.00",
    ]
    if bay_rating is not None and pd.notna(bay_rating):
        insights.append(f"**Bayesian Rating:** {float(bay_rating):.3f}")
    if avg_rating is not None and pd.notna(avg_rating):
        insights.append(f"**Avg Raw Rating:** {float(avg_rating):.2f}⭐")
    if purch_freq is not None and pd.notna(purch_freq) and float(purch_freq) > 0:
        insights.append(f"**Purchase Frequency:** {human_int(int(purch_freq))}")
    if len(top) > 1:
        insights.append(
            f"**Gap to #2:** {score_gap:.2f} pts "
            f"({(score_gap / top_product['popularity_score'] * 100):.1f}%)"
        )
    insights.append(f"**Average score (Top {top_n}):** {avg_score:.2f} | **Median:** {median_score:.2f}")
    insights.append(f"**{high_rating_products}** products ({high_rating_pct:.0f}%) have 4.5+ star ratings")
 
    interpretations, recommendations = [], []
 
    if score_gap > 0.5:
        interpretations.append("**Clear market leader** — #1 product significantly outperforms the rest")
    elif score_gap > 0.2:
        interpretations.append("**Strong leader** — #1 product holds a solid advantage")
    else:
        interpretations.append("**Competitive market** — top products are closely matched")
 
    if avg_rating is not None and pd.notna(avg_rating) and float(avg_rating) >= 4.5:
        interpretations.append("**Quality-driven** — high raw rating is the primary score driver")
    if high_rating_pct >= 70:
        interpretations.append(f"**Strong catalogue quality** — {high_rating_pct:.0f}% of top products are 4.5★+")
    elif high_rating_pct < 40:
        interpretations.append(f"**Quality gap** — only {high_rating_pct:.0f}% of popular products are 4.5★+")
 
    interpretations.append(
        "**Bayesian smoothing** reduces bias from products with few reviews, "
        "making rankings more reliable than raw averages alone"
    )
 
    if top_product["popularity_score"] >= 4.0:
        recommendations.append(
            f"**Hero product strategy** — feature '{shorten(str(top_product.get('title', '')), 40)}' "
            "on the homepage and in campaigns"
        )
    else:
        recommendations.append("**No standout winner** — consider promoting the top 3–5 products equally")
 
    recommendations.append("**Cross-sell** — bundle top-scoring products with complementary items")
    recommendations.append("**Replication** — analyse what drives top Bayesian scores and apply learnings to mid-tier products")
    recommendations.append("**Compare with NCF/CBF recs** — check overlap between popularity-based and personalised outputs")
 
    st.markdown(f"""
**Insights**
{chr(10).join(f"- {i}" for i in insights)}
 
**Interpretation**
{chr(10).join(f"- {i}" for i in interpretations)}
 
**Recommendations**
{chr(10).join(f"- {r}" for r in recommendations)}
    """)
 
    show_llm_insights(
        context=(
            f"CBF popularity pipeline top product: {top_product.get('title', '')}, "
            f"score: {top_product['popularity_score']:.2f}, "
            f"bayesian rating: {bay_rating:.3f if bay_rating is not None and pd.notna(bay_rating) else 'N/A'}, "
            f"avg score across top {len(top)}: {avg_score:.2f}"
        ),
        cache_key="cbf_popularity",
        title="CBF Popularity Analysis",
        chart_type=f"horizontal bar chart ranking the top {top_n} products by CBF-pipeline popularity score (0–5 scale, Bayesian-rating weighted)",
    )
 
    # ── Table + download ──────────────────────────────────────────────────────
    st.subheader("Top Products Table — CBF Pipeline")
    table_cols = [c for c in [
        "parent_asin", "title", "short_title", "popularity_rank",
        "average_rating", "bayesian_rating", "purchase_frequency",
        "helpful_votes", "verified_pct", "popularity_score", "rating_tier", "store",
    ] if c in top.columns]
    st.data_editor(top[table_cols], use_container_width=True, hide_index=True, num_rows="fixed")
 
    csv = top[table_cols].to_csv(index=False).encode("utf-8")
    st.download_button(
        label=f"⬇ Download Top {top_n} CBF Popular Products as CSV",
        data=csv,
        file_name=f"top_{top_n}_cbf_popular_products.csv",
        mime="text/csv",
    )
 
 
# ── Original popularity view ──────────────────────────────────────────────────
 
def _show_original_popular(products: pd.DataFrame) -> None:
    st.info(
        "**How the popularity score works:**  \n"
        "Score = **(0.6 × Average Rating) + (0.4 × Purchase Count)** — both normalised to a 0–5 scale.  \n"
        "Rating is weighted slightly higher than volume because quality matters more than quantity."
    )
 
    pop_df = _make_labels(_compute_popularity(products))
 
    top_n = st.slider("Top N products", 10, 100, 20, key="popularity_top_n")
    top   = pop_df.head(top_n)
 
    fig = px.bar(
        top.sort_values("popularity_score"),
        x="popularity_score", y="short_label", orientation="h",
        hover_data=["average_rating", "rating_number", "purchase_frequency"],
        title=f"Top {top_n} Popular Products  (score out of 5)",
    )
    fig.update_layout(height=500, yaxis_title="")
    fig.update_xaxes(range=[0, 5])
    st.plotly_chart(style_bar_chart(fig), use_container_width=True)
 
    # ── Static insight block ──────────────────────────────────────────────────
    top_product   = top.iloc[0]
    score_gap     = (
        top.iloc[0]["popularity_score"] - top.iloc[1]["popularity_score"]
        if len(top) > 1 else 0
    )
    avg_score     = top["popularity_score"].mean()
    median_score  = top["popularity_score"].median()
    avg_rating    = top_product.get("average_rating", 0)
    rating_count  = top_product.get("rating_number", 0)
    purchase_freq = top_product.get("purchase_frequency", 0)
 
    high_rating_products = (
        (top["average_rating"] >= 4.5).sum() if "average_rating" in top.columns else 0
    )
    high_rating_pct = (high_rating_products / len(top) * 100) if len(top) > 0 else 0
 
    insights = [
        f"**#1 Product:** {shorten(top_product['title'], 60)}",
        f"**Popularity Score:** {top_product['popularity_score']:.2f} / 5.00",
    ]
    if pd.notna(avg_rating):
        insights.append(f"**Rating:** {avg_rating:.2f}⭐ from {human_int(rating_count)} reviews")
    if pd.notna(purchase_freq) and purchase_freq > 0:
        insights.append(f"**Purchase Count:** {human_int(purchase_freq)}")
    if len(top) > 1:
        insights.append(
            f"**Gap to #2:** {score_gap:.2f} points "
            f"({(score_gap / top_product['popularity_score'] * 100):.1f}%)"
        )
    insights.append(f"**Average score (Top {top_n}):** {avg_score:.2f} | **Median:** {median_score:.2f}")
    insights.append(f"**{high_rating_products}** products ({high_rating_pct:.0f}%) have 4.5+ star ratings")
 
    interpretations = []
    if score_gap > 0.5:
        interpretations.append("**Clear market leader** — #1 product significantly outperforms the rest")
    elif score_gap > 0.2:
        interpretations.append("**Strong leader** — #1 product has a solid advantage")
    else:
        interpretations.append("**Competitive market** — top products are closely matched")
 
    if pd.notna(avg_rating) and avg_rating >= 4.5 and rating_count > 100:
        interpretations.append("**Quality + Volume winner** — excellent ratings backed by strong purchase count")
    elif pd.notna(avg_rating) and avg_rating >= 4.5:
        interpretations.append("**Quality-driven** — high rating is the main driver of the score")
    elif rating_count > 200:
        interpretations.append("**Volume-driven** — high purchase count compensates for moderate rating")
 
    if high_rating_pct >= 70:
        interpretations.append(f"**Strong catalogue quality** — {high_rating_pct:.0f}% of top products are 4.5★+")
    elif high_rating_pct < 40:
        interpretations.append(f"**Quality gap** — only {high_rating_pct:.0f}% of popular products are 4.5★+")
 
    recommendations = []
    if top_product["popularity_score"] >= 4.0:
        recommendations.append(
            f"**Hero product strategy** — feature '{shorten(top_product['title'], 40)}' "
            "on homepage and in campaigns"
        )
    else:
        recommendations.append("**No standout winner** — consider promoting the top 3–5 products equally")
 
    if pd.notna(avg_rating) and avg_rating >= 4.5:
        recommendations.append("**Leverage social proof** — display ratings prominently on product pages")
    elif pd.notna(avg_rating) and avg_rating < 4.0:
        recommendations.append("**Quality review needed** — investigate root causes of lower ratings")
 
    recommendations.append("**Cross-sell** — bundle top-scoring products with complementary items")
    recommendations.append("**Replication** — analyse what drives top scores and apply to mid-tier products")
 
    st.markdown(f"""
**Insights**
{chr(10).join(f"- {i}" for i in insights)}
 
**Interpretation**
{chr(10).join(f"- {i}" for i in interpretations)}
 
**Recommendations**
{chr(10).join(f"- {r}" for r in recommendations)}
    """)
 
    show_llm_insights(
        context=(
            f"Top product: {top.iloc[0]['title']}, score: {top.iloc[0]['popularity_score']:.2f}, "
            f"avg rating: {top.iloc[0].get('average_rating', 'N/A')}, "
            f"avg score across top {len(top)}: {top['popularity_score'].mean():.2f}"
        ),
        cache_key="popularity",
        title="Popularity Analysis",
        chart_type=f"horizontal bar chart ranking the top {top_n} products by popularity score (0–5 scale, combining 60% rating + 40% purchase volume)",
    )
 
    # ── Top Products Table ────────────────────────────────────────────────────
    st.subheader("Top Products Table")
    table_cols = [
        "parent_asin", "title", "average_rating",
        "rating_number", "purchase_frequency", "popularity_score",
    ]
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
        _compute_popularity(products).to_csv("popularity_table.csv", index=False)
        st.success("Saved to popularity_table.csv ✓")
 
 
# ── Tab entry point ───────────────────────────────────────────────────────────
 
def show_popularity_tab(products: pd.DataFrame) -> None:
    st.header("Most Popular Products")
 
    # ── View toggle ───────────────────────────────────────────────────────────
    cbf_df = _load_cbf_popular()
 
    view_options = ["📊 Collaborative Filtering (Live)", "🤖 Content-Based Filtering (CBF Pipeline)"]
    if cbf_df is None:
        st.warning(
            f"CBF popular products file not found at `{_POPULAR_PRODUCTS_CSV}`. "
            "Run your CBF pipeline to generate it, or check the path in `popularity.py`."
        )
        view_options = view_options[:1]  # only show live view if file missing
 
    selected_view = st.radio(
        "Select popularity view",
        view_options,
        horizontal=True,
        key="popularity_view_toggle",
    )
 
    st.divider()
 
    if selected_view == view_options[0]:
        _show_original_popular(products)
    else:
        _show_cbf_popular(cbf_df)