import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st


# =========================
# POPULARITY SCORE ENGINE
# =========================
def compute_popularity(products_df, m=50):

    df = products_df.copy()

    C = df["average_rating"].mean()

    df["rating_number"] = pd.to_numeric(df["rating_number"], errors="coerce").fillna(0)
    df["purchase_frequency"] = pd.to_numeric(df["purchase_frequency"], errors="coerce").fillna(0)

    v = df["rating_number"]
    R = df["average_rating"]
    f = df["purchase_frequency"]

    f_norm = np.log1p(f) / np.log1p(f.max() + 1)

    rating_score = (v / (v + m)) * R + (m / (v + m)) * C

    df["popularity_score"] = (0.7 * rating_score) + (0.3 * f_norm * 5)

    return df.sort_values("popularity_score", ascending=False)


# =========================
# SHORT LABEL FOR CHART
# =========================
def make_short_description(df):

    df = df.copy()

    def shorten(row):
        title = str(row["title"])
        store = row.get("store_clean", "")

        short_title = title[:40] + "..." if len(title) > 40 else title

        rating = row.get("average_rating", None)
        if pd.notna(rating):
            short_title = f"{short_title} ⭐ {rating:.1f}"

        if pd.notna(store) and store not in ["", "(missing store)"]:
            return f"{short_title} ({store})"

        return short_title

    df["short_label"] = df.apply(shorten, axis=1)
    return df


# =========================
# BAR CHART STYLE
# =========================
def style_popularity_chart(fig):

    is_dark = st.session_state.get("dark_mode", False)

    bg = "#0f172a" if is_dark else "white"
    grid = "#334155" if is_dark else "#e5e7eb"
    border = "#334155" if is_dark else "#cfcfcf"
    font = "#ffffff" if is_dark else "#111827"

    fig.update_layout(
        height=500,
        plot_bgcolor=bg,
        paper_bgcolor=bg,
        font=dict(color=font),
        xaxis=dict(showgrid=True, gridcolor=grid, showline=True, linecolor=border),
        yaxis=dict(showgrid=True, gridcolor=grid, showline=True, linecolor=border),
    )

    return fig


# =========================
# MAIN TAB FUNCTION
# =========================
def show_popularity_tab(products):

    st.header("🔥 Most Popular Products")

    # ---------- compute ----------
    pop_df = compute_popularity(products, m=50)
    pop_df = make_short_description(pop_df)

    # ---------- UI ----------
    top_n = st.slider("Top products", 10, 100, 20)

    top_pop = pop_df.head(top_n)

    # =========================
    # BAR CHART
    # =========================
    fig = px.bar(
        top_pop.sort_values("popularity_score"),
        x="popularity_score",
        y="short_label",
        orientation="h",
        hover_data=[
            "average_rating",
            "rating_number",
            "purchase_frequency"
        ],
        title="Top Popular Products (Weighted Score)"
    )

    fig = style_popularity_chart(fig)

    st.plotly_chart(fig, use_container_width=True)

    # =========================
    # EXCEL-STYLE TABLE (CSS ONLY)
    # =========================
    st.subheader("📊 Top Products Table")

    pop_table = top_pop[
        [
            "parent_asin",
            "title",
            "average_rating",
            "rating_number",
            "purchase_frequency",
            "popularity_score"
        ]
    ]

    # IMPORTANT: use Streamlit dataframe ONLY (CSS handles styling)
    st.data_editor(
    pop_table,
    use_container_width=True,
    hide_index=True,
    num_rows="fixed"
)
    