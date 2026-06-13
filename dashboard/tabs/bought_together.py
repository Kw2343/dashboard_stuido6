from __future__ import annotations
 
import pandas as pd
import plotly.express as px
import streamlit as st
 
from data_loader import load_bought_together
from utils import style_bar_chart, shorten, human_int
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
 
 
# ── Static insight builder ────────────────────────────────────────────────────
 
def _bought_together_static(pairs_df: pd.DataFrame, top: pd.DataFrame) -> None:
    total_pairs      = len(pairs_df)
    total_copurchases = int(pairs_df["count"].sum())
    top10_sum        = int(top["count"].sum())
    top10_share      = top10_sum / total_copurchases * 100 if total_copurchases else 0
    top_pair         = top.iloc[0]
    count_gap        = int(top.iloc[0]["count"] - top.iloc[1]["count"]) if len(top) > 1 else 0
    strong           = int((top["count"] >= 50).sum())
    moderate         = int(((top["count"] >= 20) & (top["count"] < 50)).sum())
    weak             = int((top["count"] < 20).sum())
    avg_count        = top["count"].mean()
    p1_name          = shorten(str(top_pair.get("Product_1", top_pair.get("parent_asin_1", "N/A"))), 45)
    p2_name          = shorten(str(top_pair.get("Product_2", top_pair.get("parent_asin_2", "N/A"))), 45)
 
    insights = [
        f"**Top pair:** {p1_name} + {p2_name}",
        f"**Co-purchase count (top pair):** {human_int(top_pair['count'])}",
        f"**Gap to #2 pair:** {human_int(count_gap)} ({count_gap/top_pair['count']*100:.1f}% ahead)" if count_gap else f"**Only one pair found**",
        f"**Top-10 pairs share:** {top10_share:.1f}% of all {human_int(total_copurchases)} co-purchases",
        f"**Total unique pairs in dataset:** {human_int(total_pairs)}",
        f"**Strength breakdown — Strong (≥50):** {strong} | **Moderate (20–49):** {moderate} | **Weak (<20):** {weak}",
        f"**Avg co-purchase count (Top 10):** {avg_count:.1f}",
    ]
 
    interpretations = []
    if strong >= 5:
        interpretations.append(f"**Strong bundle signals** — {strong} of top 10 pairs have ≥50 co-purchases; clear bundle opportunities")
    elif weak >= 7:
        interpretations.append(f"**Weak associations** — {weak} of top 10 pairs have <20 co-purchases; limited bundling evidence")
    if top10_share > 30:
        interpretations.append(f"**Concentrated co-purchasing** — top 10 pairs drive {top10_share:.0f}% of all co-purchases; strong affinity clusters")
    if count_gap > top_pair["count"] * 0.3:
        interpretations.append(f"**Clear #1 pair** — top pair is {count_gap/top_pair['count']*100:.0f}% ahead of #2; a natural bundle candidate")
 
    recommendations = []
    recommendations.append(f"**Bundle #1 pair** — '{p1_name[:35]}' + '{p2_name[:35]}' should be featured as a recommended bundle")
    if strong >= 3:
        recommendations.append(f"**'Frequently Bought Together' widget** — activate on PDPs for the {strong} strong pairs (≥50 co-purchases)")
    if weak >= 5:
        recommendations.append(f"**Investigate weak pairs** — {weak} pairs have <20 co-purchases; test if promotion can strengthen these signals")
    recommendations.append("**Cross-category bundles** — check if top pairs span categories; if so, surface them in category navigation")
 
    _render_static_insights(insights, interpretations, recommendations)
 
 
# ── Context builder ───────────────────────────────────────────────────────────
 
def _build_context(pairs_df, top):
    total_pairs       = len(pairs_df)
    total_copurchases = int(pairs_df["count"].sum())
    top10_sum         = int(top["count"].sum())
    top10_share       = top10_sum / total_copurchases * 100 if total_copurchases else 0
    top_pair          = top.iloc[0]
    count_gap         = int(top.iloc[0]["count"] - top.iloc[1]["count"]) if len(top) > 1 else 0
    strong   = int((top["count"] >= 50).sum())
    moderate = int(((top["count"] >= 20) & (top["count"] < 50)).sum())
    weak     = int((top["count"] < 20).sum())
    lines = [
        "CO-PURCHASE / BOUGHT-TOGETHER ANALYSIS", "",
        "DATASET:",
        f"  Total unique product pairs : {human_int(total_pairs)}",
        f"  Total co-purchase events   : {human_int(total_copurchases)}",
        f"  Top-10 pairs share         : {top10_share:.1f}% of all co-purchases", "",
        "TOP PAIR (#1):",
        f"  Product A : {shorten(str(top_pair.get('Product_1', top_pair.get('parent_asin_1', 'N/A'))), 60)}",
        f"  Product B : {shorten(str(top_pair.get('Product_2', top_pair.get('parent_asin_2', 'N/A'))), 60)}",
        f"  Count     : {human_int(top_pair['count'])}",
        f"  Gap to #2 : {human_int(count_gap)} ({count_gap/top_pair['count']*100:.1f}% ahead)" if count_gap else "",
        "",
        "ASSOCIATION STRENGTH (Top 10):",
        f"  Strong   (≥50 co-purchases) : {strong}",
        f"  Moderate (20–49)            : {moderate}",
        f"  Weak     (<20)              : {weak}",
        f"  Count range                 : {int(top['count'].min())} – {int(top['count'].max())}",
        f"  Average count (Top 10)      : {top['count'].mean():.1f}",
        f"  Median  count (Top 10)      : {top['count'].median():.1f}", "",
        "TOP 10 PAIRS:",
    ]
    for i, row in top.iterrows():
        p1 = shorten(str(row.get("Product_1", row.get("parent_asin_1", "?"))), 40)
        p2 = shorten(str(row.get("Product_2", row.get("parent_asin_2", "?"))), 40)
        lines.append(f"  {p1} + {p2} : {human_int(row['count'])}")
    return "\n".join(l for l in lines if l is not None)
 
 
# ── Tab entry point ───────────────────────────────────────────────────────────
 
def show_bought_together_tab(products_lookup: pd.DataFrame) -> None:
    st.header("Top Products Bought Together")
 
    pairs_df = load_bought_together()
 
    if pairs_df is None:
        st.error(
            "Could not load bought-together data. "
            "Check that the Excel file exists in the data folder."
        )
        return
 
    titles = products_lookup[["parent_asin", "title"]].copy()
    top    = pairs_df.sort_values("count", ascending=False).head(10).copy()
 
    top = (
        top
        .merge(titles, left_on="parent_asin_1", right_on="parent_asin", how="left")
        .rename(columns={"title": "Product_1"})
        .drop(columns=["parent_asin"], errors="ignore")
        .merge(titles, left_on="parent_asin_2", right_on="parent_asin", how="left")
        .rename(columns={"title": "Product_2"})
        .drop(columns=["parent_asin"], errors="ignore")
    )
 
    top["Pair"] = (
        top["Product_1"].fillna(top["parent_asin_1"])
        + " + "
        + top["Product_2"].fillna(top["parent_asin_2"])
    )
    top["Pair_short"] = top["Pair"].apply(shorten)
 
    fig = px.bar(
        top.sort_values("count"),
        x="count", y="Pair_short", orientation="h",
        title="Top 10 Frequently Bought Together",
        hover_data={"Pair": True, "Pair_short": False, "count": True},
    )
    fig.update_layout(height=500, yaxis_title="", xaxis_title="Co-purchase count")
    st.plotly_chart(style_bar_chart(fig), use_container_width=True)
 
    _bought_together_static(pairs_df, top)
    show_llm_insights(
        context   = _build_context(pairs_df, top),
        cache_key = "bought_together",
        title     = "Co-Purchase Pattern Analysis",
        chart_type= "horizontal bar chart showing the top 10 most frequently co-purchased product pairs, ranked by co-purchase count",
    )
 
    st.subheader("Details")
    st.dataframe(top[["Pair", "count"]], use_container_width=True, hide_index=True)