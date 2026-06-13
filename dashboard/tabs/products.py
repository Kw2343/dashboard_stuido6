from __future__ import annotations
 
import pandas as pd
import plotly.express as px
import streamlit as st
from pathlib import Path
 
from utils import style_bar_chart, human_int
from llm_insights import show_llm_insights
 
 
# ── CBF popular_products loader ───────────────────────────────────────────────
 
_CBF_PATH = Path(r"C:\Users\kelvi\Downloads\dashboard - Copy (2) - Copy\dashboard_data\popular_products.csv")
 
@st.cache_data(show_spinner=False)
def _load_cbf_scores() -> pd.DataFrame:
    """Load tfidf_popularity_score and dl_content_score from the CBF pipeline output."""
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
 
 
# ── Static insight builders ───────────────────────────────────────────────────
 
def _product_chart_static(product_counts: pd.DataFrame, top_n: int) -> None:
    top         = product_counts.head(top_n)
    top_product = top.sort_values("filtered_review_count").iloc[-1]
    total       = product_counts["filtered_review_count"].sum()
    top_share   = top_product["filtered_review_count"] / total * 100 if total else 0
    top10_share = top.head(10)["filtered_review_count"].sum() / total * 100 if total else 0
    median_rev  = product_counts["filtered_review_count"].median()
    avg_rev     = product_counts["filtered_review_count"].mean()
    lone        = int((product_counts["filtered_review_count"] == 1).sum())
    w20         = int((product_counts["filtered_review_count"] >= 20).sum())
    title_str   = str(top_product.get("title", "N/A"))[:60]
    avg_rating  = top_product.get("average_rating", None)
    store       = top_product.get("store_clean", "")
    tfidf_score = top_product.get("tfidf_popularity_score", None)
    dl_score    = top_product.get("dl_content_score", None)
 
    insights = [
        f"**#1 Product:** {title_str}",
        f"**Review count:** {human_int(top_product['filtered_review_count'])} ({top_share:.1f}% of all reviews)",
    ]
    if pd.notna(avg_rating):
        insights.append(f"**Rating:** {avg_rating:.2f}⭐")
    if pd.notna(store) and store not in ("", "(missing store)"):
        insights.append(f"**Store:** {store}")
    if tfidf_score is not None and pd.notna(tfidf_score):
        insights.append(f"**TF-IDF Content Score:** {float(tfidf_score):.4f}")
    if dl_score is not None and pd.notna(dl_score):
        insights.append(f"**DL Content Score:** {float(dl_score):.4f}")
    insights += [
        f"**Top 10 products capture:** {top10_share:.1f}% of all reviews",
        f"**Avg reviews/product:** {avg_rev:.1f} | **Median:** {median_rev:.0f}",
        f"**Products with 20+ reviews:** {human_int(w20)} | **With only 1:** {human_int(lone)}",
    ]
 
    interpretations = []
    if top_share > 10:
        interpretations.append(f"**Dominant product** — #1 alone accounts for {top_share:.1f}% of all reviews")
    if top10_share > 50:
        interpretations.append(f"**Concentrated engagement** — top 10 products drive {top10_share:.0f}% of reviews; long tail is sparse")
    if lone / len(product_counts) > 0.4:
        interpretations.append(f"**Long tail gap** — {lone/len(product_counts)*100:.0f}% of products have just 1 review; unreliable for ranking")
    if pd.notna(avg_rating) and avg_rating >= 4.5:
        interpretations.append("**Quality leader** — top-reviewed product also carries excellent ratings")
 
    recommendations = []
    recommendations.append(f"**Feature #1 product** — '{title_str[:40]}' has the most social proof; use it in homepage and ads")
    if top10_share > 50:
        recommendations.append("**Diversify promotion** — spread visibility to products ranked 11–25 to reduce concentration")
    if lone > 50:
        recommendations.append(f"**Review seeding** — run campaigns targeting the {human_int(lone)} products with only 1 review")
    recommendations.append("**Price-rating analysis** — cross-reference review volume with price to identify value sweet spots")
 
    _render_static_insights(insights, interpretations, recommendations)
 
 
def _metadata_static(products: pd.DataFrame, completeness: pd.DataFrame) -> None:
    total     = len(products)
    avg_cov   = completeness["Coverage"].mean() * 100
    lowest    = completeness.sort_values("Coverage").iloc[0]
    highest   = completeness.sort_values("Coverage").iloc[-1]
    critical  = completeness[completeness["Coverage"] < 0.8]
 
    insights = [
        f"**Total products:** {human_int(total)}",
        f"**Average field coverage:** {avg_cov:.1f}%",
        f"**Best field:** {highest['Field']} at {highest['Coverage']*100:.1f}%",
        f"**Worst field:** {lowest['Field']} at {lowest['Coverage']*100:.1f}% ({int(total*(1-lowest['Coverage'])):,} products missing)",
        f"**Fields below 80%:** {len(critical)}" + (f" — {', '.join(critical['Field'].tolist())}" if len(critical) else " — none"),
    ]
 
    interpretations = []
    if avg_cov >= 90:
        interpretations.append(f"**Strong data quality** — average coverage of {avg_cov:.0f}% means the catalogue is well-maintained")
    elif avg_cov >= 70:
        interpretations.append(f"**Moderate completeness** — {avg_cov:.0f}% average coverage; gaps will affect recommendation quality")
    else:
        interpretations.append(f"**Poor data quality** — only {avg_cov:.0f}% average coverage; major gaps will hurt search and recommendations")
    if len(critical) > 0:
        interpretations.append(f"**{len(critical)} critical field(s) below 80%** — missing metadata in key fields reduces discoverability")
    if lowest["Coverage"] < 0.5:
        interpretations.append(f"**{lowest['Field']} is a major gap** — only {lowest['Coverage']*100:.0f}% complete; affecting {int(total*(1-lowest['Coverage'])):,} products")
 
    recommendations = []
    recommendations.append(f"**Fix {lowest['Field']} first** — it's the worst field at {lowest['Coverage']*100:.0f}%; highest impact improvement")
    if len(critical) > 0:
        recommendations.append(f"**Data enrichment sprint** — prioritise the {len(critical)} fields below 80% in next catalogue update")
    recommendations.append("**Automated validation** — add metadata completeness checks to the product upload pipeline")
    recommendations.append("**Supplier data requests** — for missing descriptions/features, go back to suppliers or scrape manufacturer pages")
 
    _render_static_insights(insights, interpretations, recommendations)
 
 
# ── Context builders ──────────────────────────────────────────────────────────
 
def _build_product_context(product_counts, top_n):
    top_product = product_counts.head(top_n).sort_values("filtered_review_count").iloc[-1]
    total       = product_counts["filtered_review_count"].sum()
    top_share   = top_product["filtered_review_count"] / total * 100 if total else 0
    top10       = product_counts.head(10)
    top10_share = top10["filtered_review_count"].sum() / total * 100 if total else 0
    median_rev  = product_counts["filtered_review_count"].median()
    avg_rev     = product_counts["filtered_review_count"].mean()
    avg_rating  = top_product.get("average_rating", None)
    price       = top_product.get("price", None)
    store       = top_product.get("store_clean", "Unknown")
    tfidf_score = top_product.get("tfidf_popularity_score", None)
    dl_score    = top_product.get("dl_content_score", None)
    lines = [
        f"PRODUCT EXPLORATION — Top {top_n} by filtered review count", "",
        "TOP PRODUCT (most reviewed):",
        f"  Title          : {str(top_product.get('chart_title', top_product.get('title', 'N/A')))[:80]}",
        f"  Review count   : {human_int(top_product['filtered_review_count'])} ({top_share:.1f}% of all reviews)",
    ]
    if pd.notna(avg_rating):
        lines.append(f"  Average rating : {avg_rating:.2f} ⭐")
    if pd.notna(price) and price > 0:
        lines.append(f"  Price          : ${price:.2f}")
    if pd.notna(store) and store != "(missing store)":
        lines.append(f"  Seller/Store   : {store}")
    if tfidf_score is not None and pd.notna(tfidf_score):
        lines.append(f"  TF-IDF Content Score : {float(tfidf_score):.4f}  (CBF pipeline, 0–1 scale)")
    if dl_score is not None and pd.notna(dl_score):
        lines.append(f"  DL Content Score     : {float(dl_score):.4f}  (Dual-Encoder neural, 0–1 scale)")
    lines += [
        "", "REVIEW DISTRIBUTION (all products in filtered set):",
        f"  Total reviews           : {human_int(total)}",
        f"  Unique products         : {human_int(len(product_counts))}",
        f"  Top-10 products capture : {top10_share:.1f}% of all reviews",
        f"  Median reviews/product  : {median_rev:.0f}",
        f"  Average reviews/product : {avg_rev:.1f}",
        f"  Products with 1 review  : {(product_counts['filtered_review_count'] == 1).sum():,}",
        f"  Products with 20+ reviews: {(product_counts['filtered_review_count'] >= 20).sum():,}",
    ]
    return "\n".join(lines)
 
 
def _build_metadata_context(products, completeness):
    total     = len(products)
    avg_cov   = completeness["Coverage"].mean() * 100
    lowest    = completeness.sort_values("Coverage").iloc[0]
    highest   = completeness.sort_values("Coverage").iloc[-1]
    critical  = completeness[completeness["Coverage"] < 0.8]
    lines = [
        "PRODUCT METADATA COMPLETENESS", "",
        f"  Total products in catalogue: {human_int(total)}",
        f"  Average coverage across all fields: {avg_cov:.1f}%", "",
        "FIELD-BY-FIELD COVERAGE:",
    ]
    for _, row in completeness.sort_values("Coverage", ascending=False).iterrows():
        missing = int(total * (1 - row["Coverage"]))
        lines.append(f"  {row['Field']:15s}: {row['Coverage']*100:5.1f}%  ({human_int(missing)} products missing)")
    lines += [
        "",
        f"BEST FIELD  : {highest['Field']} at {highest['Coverage']*100:.1f}%",
        f"WORST FIELD : {lowest['Field']} at {lowest['Coverage']*100:.1f}%  ({int(total*(1-lowest['Coverage'])):,} products missing this data)",
        f"CRITICAL GAPS (<80% coverage): {len(critical)} field(s)"
        + (f" — {', '.join(critical['Field'].tolist())}" if len(critical) else ""),
    ]
    return "\n".join(lines)
 
 
# ── Tab entry point ───────────────────────────────────────────────────────────
 
def show_products_tab(
    filtered_reviews: pd.DataFrame,
    products_lookup: pd.DataFrame,
    products: pd.DataFrame,
) -> None:
    st.markdown("### Product exploration")
 
    # ── Load & merge CBF scores ───────────────────────────────────────────────
    cbf_scores  = _load_cbf_scores()
    has_tfidf   = "tfidf_popularity_score" in cbf_scores.columns
    has_dl      = "dl_content_score"       in cbf_scores.columns
    cbf_loaded  = has_tfidf or has_dl
 
    if cbf_loaded:
        st.caption(
            f"✅ CBF content scores loaded — "
            + ("`tfidf_popularity_score`" if has_tfidf else "")
            + (" + `dl_content_score`" if has_dl else "")
            + " columns available."
        )
    else:
        st.caption(
            f"ℹ️ CBF content scores not found at `{_CBF_PATH}`. "
            "Run the CBF pipeline to enrich this tab."
        )
 
    product_counts = (
        filtered_reviews.groupby("parent_asin", as_index=False)
        .size()
        .rename(columns={"size": "filtered_review_count"})
        .merge(products_lookup, on="parent_asin", how="left")
    )
 
    # Deduplicate: if products_lookup has multiple rows per asin (e.g. multiple
    # store variants), keep the one with the highest average_rating so the chart
    # title and hover data are clean — review count stays accurate because it
    # came from the groupby before the merge.
    if product_counts.duplicated("parent_asin").any():
        product_counts = (
            product_counts
            .sort_values(
                ["filtered_review_count", "average_rating"],
                ascending=[False, False],
            )
            .drop_duplicates("parent_asin", keep="first")
        )
 
    product_counts = product_counts.sort_values(
        ["filtered_review_count", "average_rating"], ascending=[False, False]
    )
 
    # Merge CBF scores in if available
    if cbf_loaded and len(cbf_scores) > 0:
        product_counts = product_counts.merge(cbf_scores, on="parent_asin", how="left")
 
    top_n      = st.slider("Top products to show", 10, 100, 25, key="top_products_n")
 
    title_source = "display_title" if "display_title" in product_counts.columns else "title"
    chart_data = product_counts.head(top_n).sort_values("filtered_review_count").copy()
 
    # Use the store-cleaned short name if available, otherwise take only the
    # first word(s) of the title up to the first comma/dash/parenthesis.
    # This produces the same short label Plotly shows at small font, so two
    # rows that look identical on screen share exactly one bar.
    import re as _re
    def _short(val):
        if not isinstance(val, str) or not val.strip():
            return val
        return _re.split(r'[,\-\(]', val)[0].strip()[:30]
 
    chart_data["chart_title"] = chart_data[title_source].fillna(chart_data["parent_asin"]).apply(_short)
 
    # Drop any rows whose short label is now a duplicate — keep the first
    # (highest review count, since chart_data is sorted ascending for the chart
    # but product_counts was descending — use the last occurrence of each label).
    chart_data = chart_data.drop_duplicates("chart_title", keep="last")
 
    # Build hover columns — include CBF scores when present
    hover_cols = ["parent_asin", "store_clean", "average_rating", "price", "title"]
    if has_tfidf:
        hover_cols.append("tfidf_popularity_score")
    if has_dl:
        hover_cols.append("dl_content_score")
    hover_cols = [c for c in hover_cols if c in chart_data.columns]
 
    fig = px.bar(
        chart_data,
        x="filtered_review_count", y="chart_title", orientation="h",
        hover_data=hover_cols,
        title=f"Top {top_n} products by filtered review count",
    )
    fig.update_layout(height=420, yaxis_title="Product")
    fig.update_yaxes(tickfont=dict(size=10))
    st.plotly_chart(style_bar_chart(fig), use_container_width=True)
 
    _product_chart_static(product_counts, top_n)
    show_llm_insights(
        context   = _build_product_context(product_counts, top_n),
        cache_key = "products_exploration",
        title     = "Product Engagement Analysis",
        chart_type= f"horizontal bar chart showing the top {top_n} products ranked by filtered review count",
    )
 
    # ── CBF score comparison (only when scores are available) ─────────────────
    if cbf_loaded:
        st.divider()
        st.subheader("CBF Content Score vs Review Volume")
        st.caption(
            "Do products with rich, distinctive content descriptions also attract more reviews? "
            "A positive correlation suggests content quality drives discoverability."
        )
 
        score_col = "dl_content_score" if has_dl else "tfidf_popularity_score"
        score_label = "DL Content Score" if has_dl else "TF-IDF Content Score"
 
        scatter_df = product_counts.dropna(subset=[score_col, "filtered_review_count"]).copy()
        if len(scatter_df) > 0:
            fig_s = px.scatter(
                scatter_df.head(500),   # cap for performance
                x=score_col,
                y="filtered_review_count",
                hover_data=[c for c in ["title", "average_rating", "store_clean",
                                        "tfidf_popularity_score", "dl_content_score"]
                            if c in scatter_df.columns],
                labels={
                    score_col:               score_label + " (0–1)",
                    "filtered_review_count": "Review Count",
                },
                title=f"{score_label} vs Review Volume (top 500 products)",
                opacity=0.6,
            )
            fig_s.update_layout(height=400)
            st.plotly_chart(fig_s, use_container_width=True)
 
            corr = scatter_df[[score_col, "filtered_review_count"]].corr().iloc[0, 1]
            if abs(corr) >= 0.4:
                interp = "strong" if abs(corr) >= 0.6 else "moderate"
                direction = "positive" if corr > 0 else "negative"
                st.info(
                    f"**Correlation: {corr:.3f}** — {interp} {direction} relationship. "
                    + ("Products with richer content tend to attract more reviews."
                       if corr > 0 else
                       "High-review products don't necessarily have the most distinctive content.")
                )
            else:
                st.info(
                    f"**Correlation: {corr:.3f}** — weak relationship. "
                    "Review volume and content richness are largely independent signals."
                )
 
    # ── Metadata coverage ──────────────────────────────────────────────────────
    st.divider()
    completeness = pd.DataFrame({
        "Field":    ["Price", "Description", "Features", "Store", "Categories"],
        "Coverage": [
            products["has_price"].mean(),
            products["has_description"].mean(),
            products["has_features"].mean(),
            products["has_store"].mean(),
            products["has_categories"].mean(),
        ],
    })
    fig2 = px.bar(
        completeness, x="Field", y="Coverage",
        title="Metadata coverage in products file",
    )
    fig2.update_layout(height=400, yaxis_tickformat=".0%")
    st.plotly_chart(style_bar_chart(fig2), use_container_width=True)
 
    _metadata_static(products, completeness)
    show_llm_insights(
        context   = _build_metadata_context(products, completeness),
        cache_key = "products_metadata",
        title     = "Metadata Quality Analysis",
        chart_type= "bar chart showing % coverage (0–100%) of each metadata field (Price, Description, Features, Store, Categories)",
    )
 
    # ── Search ─────────────────────────────────────────────────────────────────
    st.markdown("#### Search products")
    query = st.text_input("Search by product title or store")
 
    table = product_counts[
        product_counts["store_clean"].notna()
        & (product_counts["store_clean"] != "(missing store)")
    ].copy()
 
    if query.strip():
        q    = query.strip().lower()
        mask = (
            table["title"].fillna("").str.lower().str.contains(q, na=False)
            | table["store_clean"].fillna("").str.lower().str.contains(q, na=False)
        )
        table = table[mask]
 
    # Show CBF columns in the search results table when available
    base_cols  = ["parent_asin", "title", "store_clean", "average_rating",
                  "price", "filtered_review_count"]
    cbf_cols   = [c for c in ["tfidf_popularity_score", "dl_content_score"]
                  if c in table.columns]
    show_cols  = [c for c in base_cols + cbf_cols if c in table.columns]
    st.dataframe(table[show_cols].head(250), use_container_width=True)